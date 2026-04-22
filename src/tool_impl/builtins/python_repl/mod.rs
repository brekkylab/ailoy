mod runner;

use std::sync::Arc;

pub(crate) use runner::PythonScriptRunner;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    sandbox::Sandbox,
    tool::{Tool, ToolFunc},
};

pub async fn build_python_repl_tool() -> anyhow::Result<Tool> {
    let desc = ToolDescBuilder::new("python_repl")
        .description(
            "Execute a Python script and return stdout/stderr. \
             When a sandbox is active the VM persists across calls within a session — \
             pip-installed packages and created files survive between calls. \
             Without a sandbox the script runs in the host process; state does not persist. \
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

    let runner = Arc::new(PythonScriptRunner::new(None, 0));

    let f = ToolFunc::new(move |args: Value, sandbox: Option<Arc<Sandbox>>| {
        let runner = runner.clone();
        async move {
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
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();

            let sandbox = sandbox.as_ref();

            if !pip_packages.is_empty() {
                let pkg_refs: Vec<&str> = pip_packages.iter().map(String::as_str).collect();
                match runner.install_packages(sandbox, &pkg_refs).await {
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
                            "exit_code": 1,
                            "phase": "pip_install"
                        });
                    }
                    Ok(_) => {}
                }
            }

            match runner.run(sandbox, &code, &[]).await {
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
        }
    });

    Ok(Tool::new(desc, Arc::new(f)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool().await.unwrap();
        assert_eq!(tool.get_desc().name, "python_repl");
    }

    #[tokio::test]
    async fn test_tool_has_description() {
        let tool = build_python_repl_tool().await.unwrap();
        assert!(tool.get_desc().description.is_some());
    }

    #[tokio::test]
    async fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool().await.unwrap();
        let required = tool
            .get_desc()
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"code"));
    }

    #[tokio::test]
    async fn test_tool_schema_pip_install_is_array_of_strings() {
        let tool = build_python_repl_tool().await.unwrap();
        let item_type = tool
            .get_desc()
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    #[cfg(feature = "sandbox")]
    mod sandbox_tests {
        use std::sync::Arc;

        use super::*;
        use crate::{
            message::Part,
            sandbox::{Sandbox, SandboxConfig},
            to_value,
            tool::test_helpers::call_with_sandbox,
        };

        async fn make_sandbox() -> Arc<Sandbox> {
            Arc::new(
                Sandbox::new(SandboxConfig::default())
                    .await
                    .expect("failed to create sandbox"),
            )
        }

        #[tokio::test]
        async fn test_missing_code_param_returns_validation_error() {
            let tool = build_python_repl_tool().await.unwrap();
            let args = Part::function("call-1", "python_repl", to_value!({}));
            let msg = call_with_sandbox(&tool, args, make_sandbox().await).await;
            let phase = msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/phase")
                .and_then(|v| v.as_str())
                .unwrap();
            assert_eq!(phase, "validation");
        }

        #[tokio::test]
        async fn test_run_print_returns_stdout() {
            let tool = build_python_repl_tool().await.unwrap();
            let args = Part::function(
                "call-1",
                "python_repl",
                to_value!({ "code": "print('ailoy')" }),
            );
            let msg = call_with_sandbox(&tool, args, make_sandbox().await).await;
            let stdout = msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/stdout")
                .and_then(|v| v.as_str())
                .unwrap();
            assert!(stdout.contains("ailoy"), "stdout: {stdout:?}");
        }

        #[tokio::test]
        async fn test_pip_install_failure_returns_phase_pip_install() {
            let tool = build_python_repl_tool().await.unwrap();
            let args = Part::function(
                "call-1",
                "python_repl",
                to_value!({
                    "code": "import xyzzy_nonexistent",
                    "pip_install": ["xyzzy-nonexistent-pkg-12345"]
                }),
            );
            let msg = call_with_sandbox(&tool, args, make_sandbox().await).await;
            let phase = msg.contents[0]
                .as_value()
                .unwrap()
                .pointer("/phase")
                .and_then(|v| v.as_str())
                .unwrap();
            assert_eq!(phase, "pip_install");
        }

        #[tokio::test]
        async fn test_state_persists_across_calls() {
            let sandbox = make_sandbox().await;
            let tool = build_python_repl_tool().await.unwrap();

            let call1 = Part::function(
                "call-1",
                "python_repl",
                to_value!({ "code": "with open('/workspace/counter.txt', 'w') as f: f.write('42')" }),
            );
            let r1 = call_with_sandbox(&tool, call1, sandbox.clone()).await;
            let exit1 = r1.contents[0]
                .as_value()
                .unwrap()
                .pointer("/exit_code")
                .and_then(|v| v.as_integer())
                .unwrap_or(-1);
            assert_eq!(exit1, 0, "first call should succeed");

            let call2 = Part::function(
                "call-2",
                "python_repl",
                to_value!({ "code": "print(open('/workspace/counter.txt').read())" }),
            );
            let r2 = call_with_sandbox(&tool, call2, sandbox.clone()).await;
            let stdout = r2.contents[0]
                .as_value()
                .unwrap()
                .pointer("/stdout")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            assert!(
                stdout.contains("42"),
                "second call should see file from first call, got: {stdout:?}"
            );
        }

        #[tokio::test]
        async fn test_pip_install_and_plot_image() {
            let sandbox = make_sandbox().await;
            let tool = build_python_repl_tool().await.unwrap();

            let args = Part::function(
                "call-1",
                "python_repl",
                to_value!({
                    "code": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(0, 2 * np.pi, 100)\nplt.plot(x, np.sin(x))\nplt.savefig('/workspace/plot.png')\nprint('saved')",
                    "pip_install": ["numpy", "matplotlib"]
                }),
            );
            let msg = call_with_sandbox(&tool, args, sandbox.clone()).await;
            let result = msg.contents[0].as_value().unwrap();
            let exit_code = result
                .pointer("/exit_code")
                .and_then(|v| v.as_integer())
                .unwrap_or(-1);
            let stdout = result
                .pointer("/stdout")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            assert_eq!(
                exit_code,
                0,
                "plot script failed — stderr: {:?}",
                result.pointer("/stderr")
            );
            assert!(
                stdout.contains("saved"),
                "expected 'saved' in stdout, got: {stdout:?}"
            );

            let bytes = sandbox
                .read_file_bytes("/workspace/plot.png")
                .await
                .unwrap();
            assert!(
                bytes.starts_with(b"\x89PNG"),
                "expected PNG magic bytes in plot file"
            );
        }
    }
}
