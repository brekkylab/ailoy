use std::sync::Arc;

use futures::future::BoxFuture;

use crate::{
    agent::rt::tool::{ToolAsyncFunc, ToolContext, ToolRuntime},
    datatype::Value,
    message::ToolDescBuilder,
};

pub fn build_python_repl_tool() -> ToolRuntime {
    let desc = ToolDescBuilder::new("python_repl")
        .description(
            "Execute a Python script in an isolated sandbox and return its output. \
             The sandbox persists state across calls within a session — installed \
             packages and created files are available in subsequent calls. \
             Use `pip_install` to install packages before execution.",
        )
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "pip_install": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Packages to install before running (e.g. 'numpy>=1.24')."
                }
            },
            "required": ["code"]
        }))
        .build();

    let f: Arc<ToolAsyncFunc> = Arc::new(move |args: Value, ctx: ToolContext| {
        Box::pin(async move {
            let sandbox = match ctx.sandbox {
                Some(s) => s,
                None => {
                    return crate::to_value!({
                        "stdout": "",
                        "stderr": "python_repl requires a sandbox; set AgentState.sandbox",
                        "exit_code": -1,
                        "phase": "setup"
                    });
                }
            };

            let code = match args.pointer("/code").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => {
                    return crate::to_value!({
                        "stdout": "",
                        "stderr": "missing required parameter: code",
                        "exit_code": -1,
                        "phase": "validation"
                    });
                }
            };

            let pip_packages: Vec<String> = args
                .pointer("/pip_install")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            if !pip_packages.is_empty() {
                let pip_cmd = format!("pip install {}", pip_packages.join(" "));
                let install_result = sandbox
                    .exec(crate::sandbox::ExecRequest {
                        command: pip_cmd,
                        timeout_secs: 120,
                    })
                    .await;
                match install_result {
                    Ok(r) if r.exit_code != 0 => {
                        return crate::to_value!({
                            "stdout": "",
                            "stderr": r.stderr.as_str(),
                            "exit_code": r.exit_code as i64,
                            "phase": "pip_install"
                        });
                    }
                    Err(e) => {
                        return crate::to_value!({
                            "stdout": "",
                            "stderr": format!("pip install error: {e}").as_str(),
                            "exit_code": -1,
                            "phase": "pip_install"
                        });
                    }
                    _ => {}
                }
            }

            let py_cmd = format!("python3 -c {}", shell_escape(&code));
            match sandbox
                .exec(crate::sandbox::ExecRequest {
                    command: py_cmd,
                    timeout_secs: 60,
                })
                .await
            {
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
        }) as BoxFuture<'static, Value>
    });

    ToolRuntime::new_async(desc, f)
}

fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        agent::rt::tool::ToolContext,
        message::Part,
        sandbox::{ExecRequest, ExecResult, Sandbox},
    };

    struct FakeSandbox {
        result: ExecResult,
    }

    #[async_trait::async_trait]
    impl Sandbox for FakeSandbox {
        async fn exec(&self, _req: ExecRequest) -> anyhow::Result<ExecResult> {
            Ok(self.result.clone())
        }
        async fn shutdown(&self) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn make_ctx(stdout: &str, exit_code: i32) -> ToolContext {
        ToolContext {
            sandbox: Some(Arc::new(FakeSandbox {
                result: ExecResult {
                    stdout: stdout.to_string(),
                    stderr: String::new(),
                    exit_code,
                    timed_out: false,
                },
            })),
        }
    }

    #[test]
    fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool();
        assert_eq!(tool.desc().name, "python_repl");
    }

    #[test]
    fn test_tool_has_description() {
        let tool = build_python_repl_tool();
        assert!(tool.desc().description.is_some());
    }

    #[test]
    fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool();
        let required = tool
            .desc()
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"code"), "code must be required");
    }

    #[tokio::test]
    async fn test_missing_code_param_returns_validation_error() {
        let tool = build_python_repl_tool();
        let ctx = make_ctx("", 0);
        let call = Part::function_with_id("call-1", "python_repl", crate::to_value!({}));
        let msg = tool.run(call, ctx).await.unwrap();
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }

    #[tokio::test]
    async fn test_no_sandbox_returns_error() {
        let tool = build_python_repl_tool();
        let ctx = ToolContext::empty();
        let call = Part::function_with_id(
            "call-1",
            "python_repl",
            crate::to_value!({ "code": "print('hi')" }),
        );
        let msg = tool.run(call, ctx).await.unwrap();
        let stderr = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stderr")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(
            stderr.contains("sandbox"),
            "expected sandbox error, got: {stderr}"
        );
    }

    #[tokio::test]
    async fn test_with_sandbox_runs_code() {
        let tool = build_python_repl_tool();
        let ctx = make_ctx("hello\n", 0);
        let call = Part::function_with_id(
            "call-1",
            "python_repl",
            crate::to_value!({ "code": "print('hello')" }),
        );
        let msg = tool.run(call, ctx).await.unwrap();
        let stdout = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(stdout, "hello\n");
    }

    #[cfg(feature = "sandbox-krun")]
    mod krun_tests {
        use std::sync::Arc;

        use super::build_python_repl_tool;
        use crate::{
            agent::rt::tool::ToolContext,
            message::Part,
            sandbox::krun::{KrunSandbox, KrunSandboxConfig},
        };

        async fn python_krun_ctx() -> ToolContext {
            let sandbox = Arc::new(
                KrunSandbox::new(KrunSandboxConfig {
                    image: "python:3.12-alpine".to_string(),
                    ..Default::default()
                })
                .await
                .expect("KrunSandbox::new failed"),
            );
            ToolContext {
                sandbox: Some(sandbox),
            }
        }

        #[tokio::test]
        async fn test_python_repl_krun_basic() {
            let tool = build_python_repl_tool();
            let ctx = python_krun_ctx().await;
            let call = Part::function_with_id(
                "call-1",
                "python_repl",
                crate::to_value!({ "code": "print('hello from vm')" }),
            );
            let msg = tool.run(call, ctx).await.unwrap();
            let result = msg.contents[0].as_value().unwrap();
            let stdout = result
                .pointer("/stdout")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let exit_code = result
                .pointer("/exit_code")
                .and_then(|v| v.as_integer())
                .unwrap_or(-1);
            assert_eq!(
                exit_code,
                0,
                "stderr: {}",
                result
                    .pointer("/stderr")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            );
            assert!(stdout.contains("hello from vm"), "stdout: {:?}", stdout);
        }

        #[tokio::test]
        async fn test_python_repl_krun_stdlib_usage() {
            let tool = build_python_repl_tool();
            let ctx = python_krun_ctx().await;

            // Exercise stdlib (json + math) in a single call to verify the tool
            // correctly executes Python code inside the KrunSandbox
            let call = Part::function_with_id(
                "call-1",
                "python_repl",
                crate::to_value!({
                    "code": "import json, math; print(json.dumps({'pi': round(math.pi, 4)}))"
                }),
            );
            let msg = tool.run(call, ctx).await.unwrap();
            let result = msg.contents[0].as_value().unwrap();
            let exit_code = result
                .pointer("/exit_code")
                .and_then(|v| v.as_integer())
                .unwrap_or(-1);
            assert_eq!(
                exit_code,
                0,
                "stderr: {}",
                result
                    .pointer("/stderr")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
            );
            let stdout = result
                .pointer("/stdout")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            assert!(
                stdout.contains("\"pi\""),
                "expected JSON with pi key in stdout, got: {:?}",
                stdout
            );
        }
    }
}
