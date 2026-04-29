use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
    util::truncate::middle_truncate,
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

    let f = ToolFunc::new(|args: Value, ctx: ToolContext| async move {
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

        let Ok(out) = ctx
            .runenv
            .exec("sh".to_string(), vec!["-c".to_string(), cmd], None)
            .await
        else {
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
    });
    Ok(ToolFactory::simple(desc, f))
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{agent::AgentSpec, runenv::Local, to_value, tool::ToolContext};

    fn spec() -> AgentSpec {
        AgentSpec::new("test")
    }

    fn local_ctx() -> ToolContext {
        ToolContext::new(String::new(), Arc::new(Local {}))
    }

    #[tokio::test]
    async fn test_tool_name_is_bash() {
        let tool = build_bash_tool().await.unwrap().make(&spec());
        assert_eq!(tool.get_desc().name, "bash");
    }

    #[tokio::test]
    async fn test_tool_schema_requires_cmd() {
        let tool = build_bash_tool().await.unwrap().make(&spec());
        let required = tool
            .get_desc()
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"cmd"));
    }

    #[tokio::test]
    async fn test_missing_cmd_returns_validation_error() {
        let tool = build_bash_tool().await.unwrap().make(&spec());
        let msg = tool.call_next(to_value!({}), local_ctx()).await;
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
        let tool = build_bash_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "cmd": "echo ailoy" }), local_ctx())
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
        let tool = build_bash_tool().await.unwrap().make(&spec());
        let msg = tool
            .call_next(to_value!({ "cmd": "exit 42" }), local_ctx())
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
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_string_lossy().to_string();
        let tool = build_bash_tool().await.unwrap().make(&spec());
        let runenv = Arc::new(Local {});

        let r1 = tool
            .call_next(
                to_value!({ "cmd": format!("echo persisted > {path}") }),
                ToolContext::new(String::new(), runenv.clone()),
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

        let r2 = tool
            .call_next(
                to_value!({ "cmd": format!("cat {path}") }),
                ToolContext::new(String::new(), runenv.clone()),
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
