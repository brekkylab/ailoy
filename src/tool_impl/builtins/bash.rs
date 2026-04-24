use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

const MAX_OUTPUT_CHARS: usize = 30_000; // same as Claude Code

pub async fn build_bash_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("bash")
        .description("Execute a shell command and return stdout/stderr/exit_code.")
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute (interpreted by sh -c)"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds. 0 or omitted means no timeout."
                }
            },
            "required": ["cmd"]
        }))
        .build();

    let f_sandbox = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
        let sandbox = ctx
            .sandbox
            .expect("sandbox_aware f1 requires sandbox in ToolContext");
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
        let timeout_secs = args
            .pointer("/timeout_secs")
            .and_then(|v| v.as_integer())
            .unwrap_or(0)
            .max(0) as u64;

        let result = if timeout_secs > 0 {
            sandbox.shell_with_timeout(&cmd, timeout_secs).await
        } else {
            sandbox.shell(&cmd).await
        };

        match result {
            Ok(r) => crate::to_value!({
                "stdout": r.stdout.as_str(),
                "stderr": r.stderr.as_str(),
                "exit_code": r.exit_code as i64,
                "timed_out": r.timed_out
            }),
            Err(e) => crate::to_value!({
                "stdout": "",
                "stderr": format!("execution error: {e}").as_str(),
                "exit_code": -1,
                "timed_out": false,
                "phase": "execution"
            }),
        }
    });

    let f_local = ToolFunc::new(|args: Value| async move {
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

        let out = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&cmd)
            .output()
            .await;
        match out {
            Ok(o) => crate::to_value!({
                "stdout": middle_truncate(String::from_utf8_lossy(&o.stdout).into_owned(), MAX_OUTPUT_CHARS).as_str(),
                "stderr": middle_truncate(String::from_utf8_lossy(&o.stderr).into_owned(), MAX_OUTPUT_CHARS).as_str(),
                "exit_code": o.status.code().unwrap_or(-1) as i64,
                "timed_out": false
            }),
            Err(e) => crate::to_value!({
                "stdout": "",
                "stderr": format!("execution error: {e}").as_str(),
                "exit_code": -1,
                "timed_out": false,
                "phase": "execution"
            }),
        }
    });

    Ok(ToolFactory::sandbox_aware(desc, f_sandbox, f_local))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentSpec;

    fn no_sandbox_spec() -> AgentSpec {
        AgentSpec::new("test")
    }

    #[tokio::test]
    async fn test_tool_name_is_bash() {
        let tool = build_bash_tool().await.unwrap().make(&no_sandbox_spec());
        assert_eq!(tool.get_desc().name, "bash");
    }

    #[tokio::test]
    async fn test_tool_schema_requires_cmd() {
        let tool = build_bash_tool().await.unwrap().make(&no_sandbox_spec());
        let required = tool
            .get_desc()
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"cmd"));
    }

    #[cfg(feature = "sandbox")]
    mod sandbox_tests {
        use std::sync::Arc;

        use super::*;
        use crate::{
            sandbox::{Sandbox, SandboxConfig},
            to_value,
        };

        async fn make_sandbox() -> Arc<Sandbox> {
            Arc::new(
                Sandbox::new(SandboxConfig::default())
                    .await
                    .expect("failed to create sandbox"),
            )
        }

        fn sandbox_spec() -> AgentSpec {
            AgentSpec::new("test").sandbox("test")
        }

        #[tokio::test]
        async fn test_missing_cmd_returns_validation_error() {
            let factory = build_bash_tool().await.unwrap();
            let tool = factory.make(&sandbox_spec());
            let args = to_value!({});
            let msg = tool
                .call_next(
                    args,
                    ToolContext {
                        id: String::new(),
                        sandbox: Some(make_sandbox().await),
                    },
                )
                .await;
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
            let factory = build_bash_tool().await.unwrap();
            let tool = factory.make(&sandbox_spec());
            let args = to_value!({ "cmd": "echo ailoy" });
            let msg = tool
                .call_next(
                    args,
                    ToolContext {
                        id: String::new(),
                        sandbox: Some(make_sandbox().await),
                    },
                )
                .await;
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
            let factory = build_bash_tool().await.unwrap();
            let tool = factory.make(&sandbox_spec());
            let args = to_value!({ "cmd": "exit 42" });
            let msg = tool
                .call_next(
                    args,
                    ToolContext {
                        id: String::new(),
                        sandbox: Some(make_sandbox().await),
                    },
                )
                .await;
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
            let sandbox = make_sandbox().await;
            let factory = build_bash_tool().await.unwrap();
            let tool = factory.make(&sandbox_spec());

            let call1 = to_value!({ "cmd": "echo persisted > /workspace/flag.txt" });
            let r1 = tool
                .call_next(
                    call1,
                    ToolContext {
                        id: String::new(),
                        sandbox: Some(sandbox.clone()),
                    },
                )
                .await;
            assert_eq!(
                r1.contents[0]
                    .as_value()
                    .unwrap()
                    .pointer("/exit_code")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(-1),
                0
            );

            let call2 = to_value!({ "cmd": "cat /workspace/flag.txt" });
            let r2 = tool
                .call_next(
                    call2,
                    ToolContext {
                        id: String::new(),
                        sandbox: Some(sandbox.clone()),
                    },
                )
                .await;
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
}
