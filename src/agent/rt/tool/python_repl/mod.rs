mod env;
mod uv;

use std::sync::Arc;

use env::{InstallResult, PythonEnv};
use futures::future::BoxFuture;
use uv::resolve_uv_path;

use crate::{
    agent::rt::tool::{ToolFunc, ToolRuntime},
    datatype::Value,
    message::ToolDescBuilder,
};

/// Config supplied by the user when declaring a `PythonRepl` tool provider.
pub struct PythonReplConfig {
    /// Python version to provision (e.g. `"3.12"`). `None` → latest stable.
    pub python_version: Option<String>,
    /// Persistent venv path. `None` → temp dir, cleaned up when the tool is dropped.
    pub venv_path: Option<String>,
    /// Packages to pre-install before the first tool call.
    pub packages: Vec<String>,
}

/// Build the `python_repl` [`ToolRuntime`].
///
/// Resolves the `uv` binary, creates (or reuses) the virtual environment,
/// and pre-installs any packages listed in `config.packages`.
///
/// Returns an error if:
/// - `uv` cannot be found or downloaded
/// - the virtual environment cannot be created
/// - any package listed in `config.packages` fails to install
pub async fn build_python_repl_tool(config: PythonReplConfig) -> anyhow::Result<ToolRuntime> {
    let uv = resolve_uv_path()?;

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
        match env.install_packages(&config.packages).await? {
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

    let f: Arc<ToolFunc> = Arc::new(move |args: Value| {
        let env = env.clone();
        Box::pin(async move {
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
                    Ok(InstallResult::Failed { stderr }) => {
                        return crate::to_value!({
                            "stdout": "",
                            "stderr": stderr.as_str(),
                            "exit_code": 1,
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
        }) as BoxFuture<'static, Value>
    });

    Ok(ToolRuntime::new(desc, f))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PythonReplConfig {
        PythonReplConfig {
            python_version: None,
            venv_path: None,
            packages: vec![],
        }
    }

    // ── descriptor tests (no uv required) ────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_tool_name_is_python_repl() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        assert_eq!(tool.desc().name, "python_repl");
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_tool_has_description() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        assert!(tool.desc().description.is_some());
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_tool_schema_requires_code() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
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
    #[ignore = "requires uv"]
    async fn test_tool_schema_pip_install_is_array_of_strings() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        let item_type = tool
            .desc()
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    // ── execution tests ───────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_missing_code_param_returns_validation_error() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        use crate::message::Part;
        let call = Part::function_with_id("call-1", "python_repl", crate::to_value!({}));
        let msg = tool.run(call).await.unwrap();
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "validation");
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_print_returns_stdout() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        use crate::message::Part;
        let call = Part::function_with_id(
            "call-1",
            "python_repl",
            crate::to_value!({ "code": "print('ailoy')" }),
        );
        let msg = tool.run(call).await.unwrap();
        let stdout = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(stdout.contains("ailoy"), "stdout: {:?}", stdout);
    }

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_pip_install_failure_returns_phase_pip_install() {
        let tool = build_python_repl_tool(default_config()).await.unwrap();
        use crate::message::Part;
        let call = Part::function_with_id(
            "call-1",
            "python_repl",
            crate::to_value!({
                "code": "import xyzzy_nonexistent",
                "pip_install": ["xyzzy-nonexistent-pkg-12345"]
            }),
        );
        let msg = tool.run(call).await.unwrap();
        let phase = msg.contents[0]
            .as_value()
            .unwrap()
            .pointer("/phase")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(phase, "pip_install");
    }
}
