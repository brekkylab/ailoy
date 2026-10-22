use virtx::protocol::Error;

use super::largest_fitting;
use crate::{
    agent::{MAX_TOOL_RESULT_CHARS, fits_tool_result},
    datatype::Value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

/// `s`, of `len` characters, cut to `keep` of them, half from each end, around a note of how
/// many were cut; `s` whole when it has no more than `keep`. Finding the cut takes time in
/// `keep`, not in `len`.
fn middle_truncate(s: &str, len: usize, keep: usize) -> String {
    if len <= keep {
        return s.to_owned();
    }
    let head = keep / 2;
    let tail = keep - head;
    let head_end = s.char_indices().nth(head).map_or(s.len(), |(i, _)| i);
    let tail_start = s
        .char_indices()
        .rev()
        .take(tail)
        .last()
        .map_or(s.len(), |(i, _)| i);
    format!(
        "{}\n\n... [{} characters omitted] ...\n\n{}",
        &s[..head_end],
        len - keep,
        &s[tail_start..]
    )
}

/// What a command printed, each stream with its length in characters, and how it ended.
struct Output<'a> {
    stdout: (&'a str, usize),
    stderr: (&'a str, usize),
    exit_code: i64,
    truncated: bool,
}

impl<'a> Output<'a> {
    fn new(stdout: &'a str, stderr: &'a str, exit_code: i64, truncated: bool) -> Self {
        Self {
            stdout: (stdout, stdout.chars().count()),
            stderr: (stderr, stderr.chars().count()),
            exit_code,
            truncated,
        }
    }

    /// The result with each stream cut to at most `keep` characters.
    fn cut_to(&self, keep: usize) -> Value {
        crate::to_value!({
            "stdout": middle_truncate(self.stdout.0, self.stdout.1, keep).as_str(),
            "stderr": middle_truncate(self.stderr.0, self.stderr.1, keep).as_str(),
            "exit_code": self.exit_code,
            "timed_out": false,
            // The console cut output that would not fit one message. Said out loud,
            // since a model that takes a partial result as whole draws conclusions from it.
            "truncated": self.truncated
        })
    }

    /// The result with each stream cut to at most `keep` characters, where keeping one more
    /// would make the result too long for one tool result, whose middle the agent would cut
    /// out. A stream no longer than `keep` stays whole.
    fn fitted(&self) -> Value {
        // A stream longer than a tool result does not fit, so the search stops there.
        let longest = self.stdout.1.max(self.stderr.1).min(MAX_TOOL_RESULT_CHARS);
        let keep = largest_fitting(longest, |keep| fits_tool_result(&self.cut_to(keep)));
        self.cut_to(keep.unwrap_or(0))
    }
}

pub fn get_shell_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("shell")
        .description(
            "Shell command. Interpreted by `sh` on Linux/macOS and by `powershell` on Windows.",
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds. 0 or omitted means no timeout."
                }
            },
            "required": ["cmd"]
        }))
        .build()
}

pub fn get_shell_tool_func() -> ToolFunc {
    tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value {
        let cmd = match args.pointer("/cmd").and_then(|v| v.as_str()) {
            Some(c) => c.to_string(),
            None => {
                return crate::to_value!({
                    "stdout": "",
                    "stderr": "missing required parameter: cmd",
                    "exit_code": -1,
                    "phase": "validation"
                });
            }
        };

        // Fractional seconds allowed; 0 or absent defers to the console's own timeout.
        let timeout_ms = args
            .pointer("/timeout_secs")
            .and_then(|v| v.as_float().or_else(|| v.as_unsigned().map(|u| u as f64)))
            .filter(|secs| *secs > 0.0)
            .map(|secs| (secs * 1000.0).ceil() as u64);

        // virtx consults no shell, so asking for shell semantics means asking for a
        // shell.
        let out = match console.exec(["sh", "-c", cmd.as_str()], timeout_ms).await {
            Ok(out) => out,
            // A killed command has no result — no exit code, and whatever it wrote is
            // gone with it — so virtx refuses the execution instead of inventing one.
            Err(e) if e.code() == Some(Error::TIMED_OUT) => {
                return crate::to_value!({
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1,
                    "timed_out": true
                });
            }
            Err(e) => {
                return crate::to_value!({
                    "stdout": "",
                    "stderr": e.to_string(),
                    "exit_code": -1,
                    "timed_out": false
                });
            }
        };

        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        Output::new(&stdout, &stderr, out.code as i64, out.truncated).fitted()
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{test_console, to_value};

    /// `kept`, the result's copy of `original`: `None` when it is `original` whole; otherwise
    /// the number of characters it keeps, once its note is checked against what is missing.
    fn kept_of(original: &str, kept: &str) -> Option<usize> {
        if kept == original {
            return None;
        }
        let (head, rest) = kept
            .split_once("\n\n... [")
            .expect("a cut stream has a note");
        let (omitted, tail) = rest.split_once(" characters omitted] ...\n\n").unwrap();
        assert!(original.starts_with(head) && original.ends_with(tail));
        let keep = head.chars().count() + tail.chars().count();
        assert_eq!(head.chars().count(), keep / 2);
        assert_eq!(
            keep + omitted.parse::<usize>().unwrap(),
            original.chars().count()
        );
        Some(keep)
    }

    /// Every output fits in one tool result. A stream is cut only when the result would not
    /// fit whole, one more character kept would not fit, a note counts what the cut leaves
    /// out, and two cut streams keep the same number. `seq 1 6000` is shorter than a tool
    /// result and still cut: JSON escapes its newlines.
    #[test]
    fn output_fits_one_tool_result_and_notes_each_cut() {
        let seq = |n: usize| (1..=n).map(|i| format!("{i}\n")).collect::<String>();
        for (stdout, stderr, stdout_cut, stderr_cut) in [
            (seq(5_000), String::new(), false, false),
            (seq(6_000), String::new(), true, false),
            ("a".repeat(40_000), String::new(), true, false),
            ("a".repeat(20_000), "b".repeat(20_000), true, true),
            ("x".repeat(100), "e".repeat(50_000), false, true),
            ("가".repeat(40_000), String::new(), true, false),
            ("\u{1}".repeat(10_000), String::new(), true, false),
        ] {
            let output = Output::new(&stdout, &stderr, 3, true);
            let value = output.fitted();
            let row = format!("stdout {}, stderr {}", stdout.len(), stderr.len());
            assert!(fits_tool_result(&value), "{row}");
            assert_eq!(
                value.pointer("/exit_code").and_then(|v| v.as_integer()),
                Some(3)
            );
            assert_eq!(
                value.pointer("/truncated").and_then(|v| v.as_bool()),
                Some(true)
            );
            assert_eq!(
                value.pointer("/timed_out").and_then(|v| v.as_bool()),
                Some(false)
            );
            let text = |key: &str| value.pointer(key).and_then(|v| v.as_str()).unwrap();
            let kept = [
                kept_of(&stdout, text("/stdout")),
                kept_of(&stderr, text("/stderr")),
            ];
            assert_eq!(
                [kept[0].is_some(), kept[1].is_some()],
                [stdout_cut, stderr_cut],
                "{row}"
            );
            if let Some(keep) = kept.iter().flatten().max() {
                assert!(kept.iter().flatten().all(|k| k == keep), "{row}: {kept:?}");
                assert!(!fits_tool_result(&output.cut_to(keep + 1)), "{row}");
            }
        }
    }

    /// `seq 1 6000` comes back cut to fit one tool result, its first and last lines kept.
    #[tokio::test]
    async fn test_long_output_fits_one_tool_result() {
        let f = get_shell_tool_func();
        let mut console = test_console().await;
        let msg = f
            .call(to_value!({ "cmd": "seq 1 6000" }), "", &mut console)
            .next()
            .await
            .unwrap()
            .message;
        let value = msg.contents[0].as_value().unwrap();
        assert!(fits_tool_result(value), "{value:?}");
        let stdout = value.pointer("/stdout").and_then(|v| v.as_str()).unwrap();
        assert!(stdout.starts_with("1\n2\n") && stdout.ends_with("5999\n6000\n"));
        assert!(stdout.contains(" characters omitted] "), "{stdout:?}");
    }

    #[tokio::test]
    async fn test_missing_cmd_returns_validation_error() {
        let f = get_shell_tool_func();
        let mut console = test_console().await;
        let msg = f
            .call(to_value!({}), "", &mut console)
            .next()
            .await
            .unwrap()
            .message;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }

    #[tokio::test]
    async fn test_echo_returns_stdout() {
        let f = get_shell_tool_func();
        let mut console = test_console().await;
        let msg = f
            .call(to_value!({ "cmd": "echo ailoy" }), "", &mut console)
            .next()
            .await
            .unwrap()
            .message;
        let stdout = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(stdout.contains("ailoy"), "stdout: {stdout:?}");
    }

    #[tokio::test]
    async fn test_timeout_secs_bounds_the_command() {
        let f = get_shell_tool_func();
        let mut console = test_console().await;
        let msg = f
            .call(
                to_value!({ "cmd": "sleep 30", "timeout_secs": 1 }),
                "",
                &mut console,
            )
            .next()
            .await
            .unwrap()
            .message;
        let timed_out = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/timed_out")
            .and_then(|v| v.as_bool())
            .unwrap();
        assert!(timed_out);
    }

    #[tokio::test]
    async fn test_exit_code_is_captured() {
        let f = get_shell_tool_func();
        let mut console = test_console().await;
        let msg = f
            .call(to_value!({ "cmd": "exit 42" }), "", &mut console)
            .next()
            .await
            .unwrap()
            .message;
        let exit_code = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/exit_code")
            .and_then(|v| v.as_integer())
            .unwrap();
        assert_eq!(exit_code, 42);
    }

    #[tokio::test]
    async fn test_state_persists_across_calls() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let f = get_shell_tool_func();
        let mut console = test_console().await;

        let r1 = f
            .call(
                to_value!({ "cmd": format!("echo persisted > {path}") }),
                "",
                &mut console,
            )
            .next()
            .await
            .unwrap()
            .message;
        assert_eq!(
            r1.contents[0]
                .as_value()
                .unwrap()
                .pointer("/exit_code")
                .and_then(|v| v.as_integer())
                .unwrap_or(-1),
            0
        );

        let r2 = f
            .call(
                to_value!({ "cmd": format!("cat {path}") }),
                "",
                &mut console,
            )
            .next()
            .await
            .unwrap()
            .message;
        let stdout = r2.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            stdout.contains("persisted"),
            "second call should see file from first call, got: {stdout:?}"
        );
    }
}
