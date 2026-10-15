use std::collections::VecDeque;

use futures::StreamExt as _;
use virtx::protocol::{Error, Failure};

use crate::{
    console::ConsoleClient,
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

const MAX_OUTPUT_CHARS: usize = 30_000; // same as Claude Code

/// One stream's output as the model sees it: the first and last `MAX_OUTPUT_CHARS / 2`
/// characters, with an omission notice between them when there were more.
#[derive(Default)]
struct Kept {
    head: String,
    head_chars: usize,
    tail: VecDeque<char>,
    total: usize,
}

impl Kept {
    fn push(&mut self, text: &str) {
        let head_max = MAX_OUTPUT_CHARS / 2;
        let tail_max = MAX_OUTPUT_CHARS - head_max;
        for c in text.chars() {
            self.total += 1;
            if self.head_chars < head_max {
                self.head.push(c);
                self.head_chars += 1;
            } else {
                self.tail.push_back(c);
                if self.tail.len() > tail_max {
                    self.tail.pop_front();
                }
            }
        }
    }

    fn finish(self) -> String {
        let tail: String = self.tail.iter().collect();
        let omitted = self.total - self.head_chars - self.tail.len();
        if omitted == 0 {
            return self.head + &tail;
        }
        format!(
            "{}\n\n... [{omitted} characters omitted] ...\n\n{tail}",
            self.head
        )
    }
}

/// UTF-8 decoding across chunks: a character split between two polls is held until its last
/// byte arrives, rather than becoming two replacement characters.
#[derive(Default)]
struct Utf8Chunks {
    pending: Vec<u8>,
}

impl Utf8Chunks {
    fn decode(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut text = String::new();
        let mut at = 0;
        while at < self.pending.len() {
            match std::str::from_utf8(&self.pending[at..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    at = self.pending.len();
                }
                Err(e) => {
                    let valid = &self.pending[at..at + e.valid_up_to()];
                    text.push_str(std::str::from_utf8(valid).unwrap_or_default());
                    at += e.valid_up_to();
                    match e.error_len() {
                        Some(len) => {
                            text.push(char::REPLACEMENT_CHARACTER);
                            at += len;
                        }
                        // The start of a character whose rest is still to come.
                        None => break,
                    }
                }
            }
        }
        self.pending.drain(..at);
        text
    }

    /// What is left when the output ends: a character that never got its last byte.
    fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

/// Both streams of one command, as it runs.
#[derive(Default)]
struct Output {
    stdout: (Utf8Chunks, Kept),
    stderr: (Utf8Chunks, Kept),
    truncated: bool,
}

impl Output {
    /// Take in a piece of output, and return it as text.
    fn take(&mut self, stdout: &[u8], stderr: &[u8]) -> (String, String) {
        let out = self.stdout.0.decode(stdout);
        let err = self.stderr.0.decode(stderr);
        self.stdout.1.push(&out);
        self.stderr.1.push(&err);
        (out, err)
    }

    fn finish(mut self) -> (String, String) {
        let out = self.stdout.0.finish();
        let err = self.stderr.0.finish();
        self.stdout.1.push(&out);
        self.stderr.1.push(&err);
        (self.stdout.1.finish(), self.stderr.1.finish())
    }
}

fn tool_output(id: &str, value: Value, depth: Option<u8>) -> MessageOutput {
    MessageOutput {
        message: Message::new(Role::Tool)
            .with_contents([Part::value(value)])
            .with_id(id),
        finish_reason: FinishReason::Stop {},
        usage: None,
        depth,
        source_agent: None,
    }
}

/// A piece of a running command's output, for whoever iterates the turn; not the model's.
fn progress(id: &str, stdout: &str, stderr: &str) -> MessageOutput {
    let value = crate::to_value!({ "stdout": stdout, "stderr": stderr });
    tool_output(id, value, Some(0))
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

/// Run a command in the console, yielding its output while it runs and then one result.
///
/// Each piece of output is an output of its own (`{stdout, stderr}`), which reaches whoever
/// iterates the turn but not the model; the result is what the model reads. Dropping the
/// stream kills the command before the console's next call.
pub fn get_shell_tool_func() -> ToolFunc {
    ToolFunc::new_with_console(|args: Value, id: String, console: &mut ConsoleClient| {
        async_stream::stream! {
            let Some(cmd) = args.pointer("/cmd").and_then(|v| v.as_str()) else {
                let value = crate::to_value!({
                    "stdout": "",
                    "stderr": "missing required parameter: cmd",
                    "exit_code": -1,
                    "phase": "validation"
                });
                yield tool_output(&id, value, None);
                return;
            };

            // Fractional seconds allowed; 0 or absent defers to the console's own timeout.
            let timeout_ms = args
                .pointer("/timeout_secs")
                .and_then(|v| v.as_float().or_else(|| v.as_unsigned().map(|u| u as f64)))
                .filter(|secs| *secs > 0.0)
                .map(|secs| (secs * 1000.0).ceil() as u64);

            let mut output = Output::default();
            // virtx consults no shell, so asking for shell semantics means asking for a
            // shell.
            let ended: Result<i32, Failure> = match console
                .exec_stream(["sh", "-c", cmd], timeout_ms)
                .await
            {
                Err(failure) => Err(failure),
                Ok(mut stream) => loop {
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            output.truncated |= chunk.truncated;
                            let (out, err) = output.take(&chunk.stdout, &chunk.stderr);
                            if !out.is_empty() || !err.is_empty() {
                                yield progress(&id, &out, &err);
                            }
                            if let Some(code) = chunk.code {
                                break Ok(code);
                            }
                        }
                        Some(Err(failure)) => break Err(failure),
                        None => {
                            break Err(Failure::Broken(anyhow::anyhow!(
                                "the command's output ended without its exit code"
                            )));
                        }
                    }
                },
            };

            // A command killed at its timeout has no exit code, but what it wrote until the
            // kill is kept: it is what says where it hung.
            let timed_out = matches!(&ended, Err(f) if f.code() == Some(Error::TIMED_OUT));
            let mut failure = None;
            if let Err(f) = &ended {
                if let Failure::Refused(refusal) = f
                    && let Some(last) = refusal.output()
                {
                    output.truncated |= last.truncated;
                    let (out, err) = output.take(&last.stdout, &last.stderr);
                    if !out.is_empty() || !err.is_empty() {
                        yield progress(&id, &out, &err);
                    }
                }
                if !timed_out {
                    failure = Some(f.to_string());
                }
            }

            let truncated = output.truncated;
            let (stdout, mut stderr) = output.finish();
            if let Some(failure) = failure {
                if !stderr.is_empty() && !stderr.ends_with('\n') {
                    stderr.push('\n');
                }
                stderr.push_str(&failure);
            }
            let value = crate::to_value!({
                "stdout": stdout.as_str(),
                "stderr": stderr.as_str(),
                "exit_code": ended.as_ref().map_or(-1, |code| *code as i64),
                "timed_out": timed_out,
                // The console dropped output it could not hold. Said out loud, since a model
                // that takes a partial result as whole draws conclusions from it.
                "truncated": truncated
            });
            yield tool_output(&id, value, None);
        }
        .boxed()
    })
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::{test_console, to_value, tool::ToolProvider};

    fn shell() -> ToolFunc {
        let mut provider = ToolProvider::new();
        provider.insert_func("shell", get_shell_tool_func());
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        funcs.get("shell").unwrap().clone()
    }

    /// Every output of one call, each with how long after the call it arrived.
    async fn run(args: Value, console: &mut ConsoleClient) -> Vec<(Duration, MessageOutput)> {
        let started = Instant::now();
        let f = shell();
        let mut stream = f.call(args, "call-1", console);
        let mut outputs = Vec::new();
        while let Some(output) = stream.next().await {
            outputs.push((started.elapsed(), output));
        }
        outputs
    }

    /// What the model reads: the call's last output.
    async fn result(args: Value, console: &mut ConsoleClient) -> Value {
        let (_, last) = run(args, console).await.pop().unwrap();
        assert_eq!(last.depth, None, "a result is the tool's own output");
        last.message.contents[0].as_value().unwrap().clone()
    }

    fn text<'a>(value: &'a Value, key: &str) -> &'a str {
        value
            .pointer(&format!("/{key}"))
            .and_then(|v| v.as_str())
            .unwrap()
    }

    #[test]
    fn kept_output_is_cut_in_the_middle() {
        let mut kept = Kept::default();
        kept.push("short");
        assert_eq!(kept.finish(), "short");

        let long: String = (0..MAX_OUTPUT_CHARS + 10)
            .map(|i| char::from(b'a' + (i % 26) as u8))
            .collect();
        let mut kept = Kept::default();
        // In pieces, as a command's output arrives.
        for piece in long.as_bytes().chunks(7) {
            kept.push(std::str::from_utf8(piece).unwrap());
        }
        let half = MAX_OUTPUT_CHARS / 2;
        assert_eq!(
            kept.finish(),
            format!(
                "{}\n\n... [10 characters omitted] ...\n\n{}",
                &long[..half],
                &long[long.len() - half..]
            )
        );
    }

    #[test]
    fn a_character_split_between_chunks_is_decoded_whole() {
        let bytes = "가나".as_bytes();
        let mut chunks = Utf8Chunks::default();
        assert_eq!(chunks.decode(&bytes[..2]), "");
        assert_eq!(chunks.decode(&bytes[2..4]), "가");
        assert_eq!(chunks.decode(&bytes[4..]), "나");
        assert_eq!(chunks.decode(b"\xffok"), "\u{FFFD}ok");
        assert_eq!(chunks.decode(&bytes[..1]), "");
        assert_eq!(chunks.finish(), "\u{FFFD}");
    }

    #[tokio::test]
    async fn test_missing_cmd_returns_validation_error() {
        let mut console = test_console().await;
        let value = result(to_value!({}), &mut console).await;
        assert_eq!(text(&value, "phase"), "validation");
    }

    #[tokio::test]
    async fn test_echo_returns_stdout() {
        let mut console = test_console().await;
        let value = result(to_value!({ "cmd": "echo ailoy" }), &mut console).await;
        assert!(text(&value, "stdout").contains("ailoy"), "{value:?}");
    }

    #[tokio::test]
    async fn test_timeout_secs_bounds_the_command() {
        let mut console = test_console().await;
        let value = result(
            to_value!({ "cmd": "sleep 30", "timeout_secs": 1 }),
            &mut console,
        )
        .await;
        assert_eq!(
            value.pointer("/timed_out").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    /// What a command wrote before its timeout killed it is what says where it hung.
    #[tokio::test]
    async fn a_timed_out_command_keeps_its_output() {
        let mut console = test_console().await;
        let value = result(
            to_value!({ "cmd": "echo started; echo err >&2; sleep 30", "timeout_secs": 1 }),
            &mut console,
        )
        .await;
        assert_eq!(
            value.pointer("/timed_out").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(text(&value, "stdout"), "started\n");
        assert_eq!(text(&value, "stderr"), "err\n");
    }

    /// Output is yielded as it is written, at a depth of its own, before the result.
    #[tokio::test]
    async fn output_arrives_while_the_command_runs() {
        let mut console = test_console().await;
        let outputs = run(
            to_value!({ "cmd": "for i in 1 2 3; do echo tick $i; sleep 1; done" }),
            &mut console,
        )
        .await;

        let (progress, result) = outputs.split_at(outputs.len() - 1);
        let (ended, result) = &result[0];
        assert!(progress.iter().all(|(_, o)| o.depth == Some(0)));
        let streamed: String = progress
            .iter()
            .map(|(_, o)| text(o.message.contents[0].as_value().unwrap(), "stdout").to_string())
            .collect();
        assert_eq!(streamed, "tick 1\ntick 2\ntick 3\n");

        let (first, _) = &progress[0];
        assert!(
            *ended - *first > Duration::from_millis(1500),
            "the first tick came {first:?} in, the end {ended:?}"
        );

        let value = result.message.contents[0].as_value().unwrap();
        assert_eq!(text(value, "stdout"), "tick 1\ntick 2\ntick 3\n");
        assert_eq!(
            value.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(0)
        );
    }

    /// A call let go of while its command runs, as a turn broken out of lets go of it, ends the
    /// command before the console's next call.
    #[tokio::test]
    async fn dropping_a_call_ends_its_command() {
        let mut console = test_console().await;
        {
            let f = shell();
            let mut stream = f.call(
                to_value!({ "cmd": "echo marker-dropped; sleep 1000" }),
                "call-1",
                &mut console,
            );
            assert_eq!(stream.next().await.unwrap().depth, Some(0));
            // The next poll waits on a silent command; it is let go of while still out.
            let mut next = std::pin::pin!(stream.next());
            let polled =
                std::future::poll_fn(|cx| std::task::Poll::Ready(next.as_mut().poll(cx))).await;
            assert!(polled.is_pending());
        }

        let value = result(
            // The bracket keeps this command's own line from matching.
            to_value!({ "cmd": "grep -l 'marker-dr[o]pped' /proc/[0-9]*/cmdline | wc -l" }),
            &mut console,
        )
        .await;
        assert_eq!(text(&value, "stdout").trim(), "0");
    }

    #[tokio::test]
    async fn test_exit_code_is_captured() {
        let mut console = test_console().await;
        let value = result(to_value!({ "cmd": "exit 42" }), &mut console).await;
        assert_eq!(
            value.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(42)
        );
    }

    #[tokio::test]
    async fn test_state_persists_across_calls() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let mut console = test_console().await;

        let first = result(
            to_value!({ "cmd": format!("echo persisted > {path}") }),
            &mut console,
        )
        .await;
        assert_eq!(
            first.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(0)
        );

        let second = result(to_value!({ "cmd": format!("cat {path}") }), &mut console).await;
        assert!(
            text(&second, "stdout").contains("persisted"),
            "second call should see file from first call, got: {second:?}"
        );
    }
}
