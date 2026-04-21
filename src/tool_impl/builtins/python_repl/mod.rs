#![cfg(feature = "sandbox-microvm")]

mod source_tool;

use std::sync::Arc;

pub(crate) use source_tool::{PythonSourceToolConfig, build_python_source_tool};

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
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
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
            if let Err(e) = sandbox.write_file("/workspace/__ailoy_run.py", code.as_bytes()).await {
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
