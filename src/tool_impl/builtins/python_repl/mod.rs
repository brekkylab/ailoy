mod runner;

use std::sync::Arc;

use crate::{
    agent::AgentSpec,
    datatype::Value,
    message::ToolDescBuilder,
    tool::{ToolContext, ToolFactory, ToolFunc},
};

pub async fn build_python_repl_tool() -> anyhow::Result<ToolFactory> {
    let desc = ToolDescBuilder::new("python_repl")
        .description(
            "Execute a Python script and return stdout/stderr.
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

    let f = |_spec: &AgentSpec| {
        let runner = Arc::new(runner::PythonReplRunner::new());

        ToolFunc::new(move |args: Value, ctx: ToolContext| {
            let runner = runner.clone();
            let runenv = ctx.runenv.clone();
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

                if !pip_packages.is_empty() {
                    let pkg_refs: Vec<&str> = pip_packages.iter().map(String::as_str).collect();
                    match runner.install_packages(&runenv, &pkg_refs).await {
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

                match runner.run(&runenv, &code, &[]).await {
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
        })
    };

    Ok(ToolFactory::with_initializer(desc, f))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{agent::AgentSpec, runenv::Local, tool::ToolContext};

    fn spec() -> AgentSpec {
        AgentSpec::new("test")
    }

    fn ctx() -> ToolContext {
        ToolContext::new("1", Arc::new(Local {}))
    }

    #[tokio::test]
    async fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        assert_eq!(tool.get_desc().name, "python_repl");
    }

    #[tokio::test]
    async fn test_tool_has_description() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        assert!(tool.get_desc().description.is_some());
    }

    #[tokio::test]
    async fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
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
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let item_type = tool
            .get_desc()
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    #[tokio::test]
    async fn test_missing_code_param_returns_validation_error() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let args = crate::to_value!({});
        let msg = tool.call_next(args, ctx()).await;
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
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let args = crate::to_value!({ "code": "print('ailoy')" });
        let msg = tool.call_next(args, ctx()).await;
        let result = msg.contents[0].as_value().unwrap();
        let stdout = result
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(stdout.contains("ailoy"), "stdout: {stdout:?}");
    }

    #[tokio::test]
    async fn test_exit_code_nonzero_on_error() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let args = crate::to_value!({ "code": "raise SystemExit(42)" });
        let msg = tool.call_next(args, ctx()).await;
        let exit_code = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/exit_code")
            .and_then(|v| v.as_integer())
            .unwrap_or(0);
        assert_eq!(exit_code, 42);
    }

    #[tokio::test]
    async fn test_pip_install_failure_returns_phase_pip_install() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let args = crate::to_value!({
            "code": "import xyzzy_nonexistent",
            "pip_install": ["xyzzy-nonexistent-pkg-12345"]
        });
        let msg = tool.call_next(args, ctx()).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "pip_install");
    }

    #[tokio::test]
    async fn test_stderr_captured_on_script_error() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let args = crate::to_value!({ "code": "import sys; print('err', file=sys.stderr); raise SystemExit(1)" });
        let msg = tool.call_next(args, ctx()).await;
        let result = msg.contents[0].as_value().unwrap();
        let stderr = result
            .pointer("/stderr")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(stderr.contains("err"), "stderr: {stderr:?}");
    }

    #[tokio::test]
    async fn test_pip_install_and_plot_image() {
        let tool = build_python_repl_tool().await.unwrap().make(&spec());
        let plot_path = std::env::temp_dir().join("ailoy_test_plot.png");
        let plot_path_str = plot_path.to_string_lossy();
        let args = crate::to_value!({
            "code": format!(
                "import matplotlib\nmatplotlib.use('Agg')\n\
                 import matplotlib.pyplot as plt\nimport numpy as np\n\
                 x = np.linspace(0, 2 * np.pi, 100)\n\
                 plt.plot(x, np.sin(x))\n\
                 plt.savefig('{plot_path_str}')\nprint('saved')"
            ),
            "pip_install": ["numpy", "matplotlib"]
        });
        let msg = tool.call_next(args, ctx()).await;
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
        let bytes = std::fs::read(&plot_path).expect("plot file not found on host");
        assert!(
            bytes.starts_with(b"\x89PNG"),
            "expected PNG magic bytes in plot file"
        );
        let _ = std::fs::remove_file(&plot_path);
    }
}
