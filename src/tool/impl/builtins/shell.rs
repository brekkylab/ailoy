use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
    util::truncate::middle_truncate,
};

const MAX_OUTPUT_CHARS: usize = 30_000; // same as Claude Code

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
    tool_func!(async |args: Value, console: &dyn Console| -> Value {
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

        let Ok(out) = console.exec_shell(cmd, None).await else {
            return crate::to_value!({
                "stdout": String::new(),
                "stderr": String::from("Internal error"),
                "exit_code": -1,
                "timed_out": false
            });
        };
        crate::to_value!({
            "stdout": middle_truncate(out.stdout, MAX_OUTPUT_CHARS).as_str(),
            "stderr": middle_truncate(out.stderr, MAX_OUTPUT_CHARS).as_str(),
            "exit_code": out.exit_code as i64,
            "timed_out": out.timed_out
        })
    })
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        runenv::LocalConsole,
        to_value,
        tool::ToolProvider,
    };

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
        let local = LocalConsole::new();
        let console = &local;
        let msg = f
            .call(to_value!({}), "", console)
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
        let local = LocalConsole::new();
        let console = &local;
        let msg = f
            .call(to_value!({ "cmd": "echo ailoy" }), "", console)
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
        let local = LocalConsole::new();
        let console = &local;
        let msg = f
            .call(to_value!({ "cmd": "exit 42" }), "", console)
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
        let local = LocalConsole::new();
        let console = &local;

        let r1 = f
            .call(
                to_value!({ "cmd": format!("echo persisted > {path}") }),
                "",
                console,
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
            .call(to_value!({ "cmd": format!("cat {path}") }), "", console)
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
