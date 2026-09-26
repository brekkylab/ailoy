use cortex::protocol::Error;

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

        // Seconds, whole or not, as the model gives them; 0 or none is no bound of the call's
        // own, which leaves it to whatever the console was built with.
        let timeout_ms = args
            .pointer("/timeout_secs")
            .and_then(|v| v.as_float().or_else(|| v.as_unsigned().map(|u| u as f64)))
            .filter(|secs| *secs > 0.0)
            .map(|secs| (secs * 1000.0).ceil() as u64);

        // cortex consults no shell, so asking for shell semantics means asking for a
        // shell.
        let out = match console.exec(["sh", "-c", cmd.as_str()], timeout_ms).await {
            Ok(out) => out,
            // A killed command has no result — no exit code, and whatever it wrote is
            // gone with it — so cortex refuses the execution instead of inventing one.
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
            // The console cut the output because it would not fit one message. Said
            // out loud, because a model reading a partial result it believes is whole
            // draws a conclusion from it.
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
    async fn test_timeout_secs_bounds_the_command() {
        let provider = provider().await;
        let funcs = provider.provide(&[get_shell_tool_desc()]).unwrap();
        let f = funcs.get("shell").unwrap();
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
