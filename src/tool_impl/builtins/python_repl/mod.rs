#![cfg(feature = "sandbox-microvm")]

use std::sync::Arc;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    sandbox::Sandbox,
    tool::{Tool, ToolFunc},
};

pub async fn build_python_repl_tool(sandbox: Arc<Sandbox>) -> anyhow::Result<Tool> {
    let desc = ToolDescBuilder::new("python_repl")
        .description(
            "Execute a Python script in an isolated MicroVM and return stdout/stderr. \
             The VM persists across calls within a session — pip-installed packages \
             and created files are available in subsequent calls. \
             Use `/workspace` as the working directory for any files you create or read. \
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

    let f = ToolFunc::new(move |args: Value| {
        let sandbox = sandbox.clone();
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

            // pip install if requested
            if !pip_packages.is_empty() {
                let pkg_args: Vec<&str> = pip_packages.iter().map(String::as_str).collect();
                let mut install_args = vec!["install"];
                install_args.extend_from_slice(&pkg_args);
                match sandbox.exec("pip", &install_args).await {
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

            // Write code to file and execute
            if let Err(e) = sandbox
                .write_file("/workspace/__ailoy_run.py", code.as_bytes())
                .await
            {
                return crate::to_value!({
                    "stdout": "",
                    "stderr": format!("failed to write code file: {e}").as_str(),
                    "exit_code": -1,
                    "phase": "execution"
                });
            }

            match sandbox.shell("python3 /workspace/__ailoy_run.py").await {
                Ok(result) => crate::to_value!({
                    "stdout": result.stdout.as_str(),
                    "stderr": result.stderr.as_str(),
                    "exit_code": result.exit_code as i64,
                    "timed_out": result.timed_out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{message::Part, sandbox::SandboxConfig, to_value};

    async fn make_sandbox() -> Arc<Sandbox> {
        Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("failed to create sandbox"),
        )
    }

    // ── descriptor tests (sandbox needed to build the tool, but no code runs) ──

    #[tokio::test]
    async fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        assert_eq!(tool.get_desc().name, "python_repl");
    }

    #[tokio::test]
    async fn test_tool_has_description() {
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        assert!(tool.get_desc().description.is_some());
    }

    #[tokio::test]
    async fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
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
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        let item_type = tool
            .get_desc()
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    // ── execution tests (require VM boot + python:3.12-slim image) ────────────

    #[tokio::test]
    async fn test_missing_code_param_returns_validation_error() {
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        let args = Part::function("call-1", "python_repl", to_value!({}));
        let msg = tool.call(&args).await.unwrap();
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
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        let args = Part::function(
            "call-1",
            "python_repl",
            to_value!({ "code": "print('ailoy')" }),
        );
        let msg = tool.call(&args).await.unwrap();
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
        let tool = build_python_repl_tool(make_sandbox().await).await.unwrap();
        let args = Part::function(
            "call-1",
            "python_repl",
            to_value!({
                "code": "import xyzzy_nonexistent",
                "pip_install": ["xyzzy-nonexistent-pkg-12345"]
            }),
        );
        let msg = tool.call(&args).await.unwrap();
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "pip_install");
    }

    /// Verifies that VM state persists across tool calls — a key property of the
    /// sandbox approach over the old per-call subprocess model.
    #[tokio::test]
    async fn test_state_persists_across_calls() {
        let sandbox = make_sandbox().await;
        let tool = build_python_repl_tool(sandbox).await.unwrap();

        // First call: set a variable by writing to a file
        let call1 = Part::function(
            "call-1",
            "python_repl",
            to_value!({ "code": "with open('/workspace/counter.txt', 'w') as f: f.write('42')" }),
        );
        let r1 = tool.call(&call1).await.unwrap();
        let exit1 = r1.contents[0]
            .as_value()
            .unwrap()
            .pointer("/exit_code")
            .and_then(|v| v.as_integer())
            .unwrap_or(-1);
        assert_eq!(exit1, 0, "first call should succeed");

        // Second call: read the file written in the first call
        let call2 = Part::function(
            "call-2",
            "python_repl",
            to_value!({ "code": "print(open('/workspace/counter.txt').read())" }),
        );
        let r2 = tool.call(&call2).await.unwrap();
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
        let tool = build_python_repl_tool(sandbox.clone()).await.unwrap();

        let args = Part::function(
            "call-1",
            "python_repl",
            to_value!({
                "code": "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nimport numpy as np\nx = np.linspace(0, 2 * np.pi, 100)\nplt.plot(x, np.sin(x))\nplt.savefig('/workspace/plot.png')\nprint('saved')",
                "pip_install": ["numpy", "matplotlib"]
            }),
        );
        let msg = tool.call(&args).await.unwrap();
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

        // Verify PNG magic bytes in the written file
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
