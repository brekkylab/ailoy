mod env;
mod uv;

use std::sync::Arc;

use env::{InstallResult, PythonEnv};
use uv::ensure_uv;

use crate::{
    datatype::Value,
    message::ToolDescBuilder,
    tool::{Tool, ToolFunc},
};

/// ```text
/// +----------------------+                   call with (`pip_install`, `code`)
/// | PythonReplConfig     |                                 |
/// | - python_version     |                                 v
/// | - venv_path          |   +-----------------------------+--------------------+
/// | - packages           |-->| "python_repl" Tool                        |
/// +----------------------+   |                                                  |
///                            | +-----------+     +----------------------------+ |
///                            | | Acquire   | --> | Create PythonEnv           | |
///                            | | `uv`      |     | - prepare venv             | |
///                            | +-----------+     | - install initial packages | |
///                            |                   +----------------------------+ |
///                            +-----------------------------+--------------------+
///                                                          |
///                                                          v
///                                       Install additional packages `pip_install`
///                                                          |
///                                                          v
///                                                   Run python `code`
/// ```

/// Config supplied by the user when declaring a `PythonRepl` tool provider.
pub struct PythonReplConfig {
    /// Python version to provision (e.g. `"3.12"`). `None` → latest stable.
    pub python_version: Option<String>,
    /// Persistent venv path. `None` → temp dir, cleaned up when the tool is dropped.
    pub venv_path: Option<String>,
    /// Packages to pre-install before the first tool call.
    pub packages: Vec<String>,
}

/// Build the `python_repl` [`Tool`].
///
/// Resolves the `uv` binary, creates (or reuses) the virtual environment,
/// and pre-installs any packages listed in `config.packages`.
///
/// Returns an error if:
/// - `uv` cannot be found or downloaded
/// - the virtual environment cannot be created
/// - any package listed in `config.packages` fails to install
pub async fn build_python_repl_tool(config: PythonReplConfig) -> anyhow::Result<Tool> {
    let uv = ensure_uv().await?;

    let env = match config.venv_path {
        Some(ref path) => {
            let expanded = shellexpand::tilde(path).into_owned();
            PythonEnv::new(
                uv,
                config.python_version,
                std::path::PathBuf::from(expanded),
                false,
            )
            .await?
        }
        None => PythonEnv::new_temp(uv, config.python_version).await?,
    };

    // Pre-install user-specified packages.  Failure here is fatal so the
    // user learns about misconfigured environments early.
    if !config.packages.is_empty() {
        match env.install_packages(&config.packages).await {
            InstallResult::Failed { stderr } => {
                anyhow::bail!(
                    "Failed to pre-install packages {:?}: {}",
                    config.packages,
                    stderr
                );
            }
            _ => {}
        }
    }

    let env = Arc::new(env);

    let desc = ToolDescBuilder::new("python_repl")
        .description(
            "Execute a Python script and return its stdout/stderr output. \
             The script runs in a virtual environment shared across tool calls \
             within this session. Use `pip_install` to install required packages \
             before execution. Each execution is stateless — variables from \
             previous calls are not available.",
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
                    "description": "Packages to install before running. Supports version specifiers (e.g. 'numpy>=1.24', 'pandas==2.1.0')."
                }
            },
            "required": ["code"]
        }))
        .build();

    let f = ToolFunc::new(move |args: Value| {
        let env = env.clone();
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
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default();

            // Install packages if requested.
            if !pip_packages.is_empty() {
                match env.install_packages(&pip_packages).await {
                    InstallResult::Failed { stderr } => {
                        return crate::to_value!({
                            "stdout": "",
                            "stderr": format!("pip install error: {stderr}").as_str(),
                            "exit_code": 1,
                            "phase": "pip_install"
                        });
                    }
                    _ => {}
                }
            }

            // Execute the script.
            match env.run_code(&code, 60).await {
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
    use futures::StreamExt as _;

    use super::*;

    fn default_config() -> PythonReplConfig {
        PythonReplConfig {
            python_version: None,
            venv_path: None,
            packages: vec![],
        }
    }

    // Helper: run a tool call and return the result message.
    async fn run(tool: &Tool, call: crate::message::Part) -> crate::message::Message {
        tool.get_func()
            .call(call)
            .unwrap()
            .next()
            .await
            .unwrap()
            .message
    }

    // ── descriptor tests (no uv required) ────────────────────────────────────

    #[tokio::test]
    async fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        assert_eq!(tool.get_desc().name, "python_repl");
    }

    #[tokio::test]
    async fn test_tool_has_description() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        assert!(tool.get_desc().description.is_some());
    }

    #[tokio::test]
    async fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let required = tool
            .get_desc()
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"code"), "code must be required");
    }

    #[tokio::test]
    async fn test_tool_schema_pip_install_is_array_of_strings() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let item_type = tool
            .get_desc()
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    // ── execution tests ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_missing_code_param_returns_validation_error() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let call = crate::message::Part::function("call-1", "python_repl", crate::to_value!({}));
        let msg = run(&tool, call).await;
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
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let call = crate::message::Part::function(
            "call-1",
            "python_repl",
            crate::to_value!({ "code": "print('ailoy')" }),
        );
        let msg = run(&tool, call).await;
        let stdout = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(stdout.contains("ailoy"), "stdout: {:?}", stdout);
    }

    #[tokio::test]
    async fn test_pip_install_failure_returns_phase_pip_install() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let call = crate::message::Part::function(
            "call-1",
            "python_repl",
            crate::to_value!({
                "code": "import xyzzy_nonexistent",
                "pip_install": ["xyzzy-nonexistent-pkg-12345"]
            }),
        );
        let msg = run(&tool, call).await;
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "pip_install");
    }

    /// Installs numpy + matplotlib, generates a sine-wave chart, and validates
    /// the saved PNG — all by calling the tool directly (no LLM involved).
    #[test_with::executable(uv)]
    #[tokio::test]
    async fn test_numpy_matplotlib_chart() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let chart_dir = tempfile::tempdir().expect("failed to create temp dir");
        let chart_path = chart_dir.path().join("sine_chart.png");

        // Install numpy and matplotlib, then generate and save the chart.
        let code = format!(
            "import numpy as np\n\
             import matplotlib\n\
             matplotlib.use('Agg')\n\
             import matplotlib.pyplot as plt\n\
             x = np.linspace(0, 4 * np.pi, 200)\n\
             plt.plot(x, np.sin(x))\n\
             plt.savefig('{}')\n\
             print('done')",
            chart_path.display()
        );

        let call = crate::message::Part::function(
            "call-1",
            "python_repl",
            crate::to_value!({
                "code": code,
                "pip_install": ["numpy", "matplotlib"]
            }),
        );
        let msg = run(&tool, call).await;
        let result = msg.contents[0].as_value().unwrap();

        let exit_code = result
            .pointer("/exit_code")
            .and_then(|v| v.as_integer())
            .unwrap_or(-1);
        let stdout = result
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let stderr = result
            .pointer("/stderr")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(
            exit_code, 0,
            "expected exit_code 0.\nstdout: {stdout}\nstderr: {stderr}"
        );

        assert!(
            chart_path.exists(),
            "chart file was not created at {:?}",
            chart_path
        );

        const PNG_MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
        let header = std::fs::read(&chart_path).unwrap();
        assert!(
            header.starts_with(PNG_MAGIC),
            "file at {:?} is not a valid PNG (got {:?})",
            chart_path,
            &header[..header.len().min(8)]
        );
    }
}
