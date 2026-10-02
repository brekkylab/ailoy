use virtx::protocol::Error;

use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

const MAX_OUTPUT_CHARS: usize = 30_000; // same as Claude Code

/// Truncate `s` to at most `max_chars` characters, keeping equal-sized head and
/// tail and inserting an omission notice in the middle.
fn middle_truncate(s: String, max_chars: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max_chars {
        return s;
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let omitted = chars.len() - head - tail;
    let head_str: String = chars[..head].iter().collect();
    let tail_str: String = chars[chars.len() - tail..].iter().collect();
    format!("{head_str}\n\n... [{omitted} characters omitted] ...\n\n{tail_str}")
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
                    "description": "Timeout in seconds. 0 or omitted means the default (600)."
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

        // Fractional seconds allowed; 0 or absent means the default. The protocol's expiry
        // is a kill with no output, so a bound has to exist: an agent that hangs a shell
        // forever hangs the run.
        const DEFAULT_TIMEOUT_SECS: f64 = 600.0;
        let timeout_secs = args
            .pointer("/timeout_secs")
            .and_then(|v| v.as_float().or_else(|| v.as_unsigned().map(|u| u as f64)))
            .filter(|secs| *secs > 0.0)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);
        let timeout_ms = (timeout_secs * 1000.0).ceil() as u64;

        // virtx consults no shell, so asking for shell semantics means asking for a
        // shell.
        let out = match console
            .exec(["sh", "-c", cmd.as_str()], Some(timeout_ms))
            .await
        {
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

        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        crate::to_value!({
            "stdout": middle_truncate(stdout, MAX_OUTPUT_CHARS).as_str(),
            "stderr": middle_truncate(stderr, MAX_OUTPUT_CHARS).as_str(),
            "exit_code": out.code as i64,
            "timed_out": false,
            // The console cut output that would not fit one message. Said out loud,
            // since a model that takes a partial result as whole draws conclusions from it.
            "truncated": out.truncated
        })
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{test_console, to_value, tool::ToolProvider};

    async fn provider() -> ToolProvider {
        let mut provider = ToolProvider::new();
        provider.insert_func("shell", get_shell_tool_func());
        provider
    }

    #[tokio::test]
    async fn test_missing_cmd_returns_validation_error() {
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
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
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
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
    async fn test_exit_code_is_captured() {
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
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
    async fn test_timeout_secs_kills_and_reports_timed_out() {
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
        let mut console = test_console().await;
        let started = std::time::Instant::now();
        let msg = f
            .call(
                to_value!({ "cmd": "sleep 10", "timeout_secs": 1 }),
                "",
                &mut console,
            )
            .next()
            .await
            .unwrap()
            .message;
        let v = msg.contents[0].as_value().unwrap();
        assert_eq!(
            v.pointer("/timed_out").and_then(|b| b.as_bool()),
            Some(true),
            "{v:?}"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_state_persists_across_calls() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
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
