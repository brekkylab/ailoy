mod runner;

use std::sync::Arc;

use crate::{
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
    tool_func,
};

pub fn get_python_repl_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("python_repl")
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
        .build()
}

pub fn get_python_repl_tool_factory() -> impl Fn(&ToolDesc) -> ToolFunc {
    |_| {
        let runner = Arc::new(runner::PythonReplRunner::new());
        tool_func!(async |args: Value, console: &dyn Console| -> Value
            with[runner = runner.clone()]
            {
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
                match runner.install_packages(console, &pkg_refs).await {
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

            match runner.run(console, &code, &[]).await {
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
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        datatype::Value,
        message::Message,
        runenv::{Local, Machine},
        tool::ToolProvider,
    };

    fn provider() -> ToolProvider {
        let mut provider = ToolProvider::new();
        provider.insert_func_factory("python_repl", get_python_repl_tool_factory());
        provider
    }

    async fn call(args: Value) -> Message {
        let provider = provider();
        let funcs = provider.provide(&[get_python_repl_tool_desc()]).unwrap();
        let f = funcs.get("python_repl").unwrap();
        let mut local = Local::new();
        let console = local.start().await.unwrap();
        f.call(args, "1", console).next().await.unwrap().message
    }

    #[test]
    fn test_tool_name_is_python_repl() {
        assert_eq!(get_python_repl_tool_desc().name, "python_repl");
    }

    #[test]
    fn test_tool_has_description() {
        assert!(get_python_repl_tool_desc().description.is_some());
    }

    #[test]
    fn test_tool_schema_requires_code() {
        let desc = get_python_repl_tool_desc();
        let required = desc
            .parameters
            .pointer("/required")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
        assert!(names.contains(&"code"));
    }

    #[test]
    fn test_tool_schema_pip_install_is_array_of_strings() {
        let desc = get_python_repl_tool_desc();
        let item_type = desc
            .parameters
            .pointer("/properties/pip_install/items/type")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(item_type, "string");
    }

    #[tokio::test]
    async fn test_missing_code_param_returns_validation_error() {
        let msg = call(crate::to_value!({})).await;
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
        let msg = call(crate::to_value!({ "code": "print('ailoy')" })).await;
        let result = msg.contents[0].as_value().unwrap();
        let stdout = result
            .pointer("/stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(stdout.contains("ailoy"), "stdout: {stdout:?}");
    }

    #[tokio::test]
    async fn test_exit_code_nonzero_on_error() {
        let msg = call(crate::to_value!({ "code": "raise SystemExit(42)" })).await;
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
        let msg = call(crate::to_value!({
            "code": "import xyzzy_nonexistent",
            "pip_install": ["xyzzy-nonexistent-pkg-12345"]
        }))
        .await;
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
        let msg = call(crate::to_value!({
            "code": "import sys; print('err', file=sys.stderr); raise SystemExit(1)"
        }))
        .await;
        let result = msg.contents[0].as_value().unwrap();
        let stderr = result
            .pointer("/stderr")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(stderr.contains("err"), "stderr: {stderr:?}");
    }

    #[tokio::test]
    async fn test_pip_install_and_plot_image() {
        let plot_path = std::env::temp_dir().join("ailoy_test_plot.png");
        let plot_path_str = plot_path.to_string_lossy();
        let msg = call(crate::to_value!({
            "code": format!(
                "import matplotlib\nmatplotlib.use('Agg')\n\
                 import matplotlib.pyplot as plt\nimport numpy as np\n\
                 x = np.linspace(0, 2 * np.pi, 100)\n\
                 plt.plot(x, np.sin(x))\n\
                 plt.savefig('{plot_path_str}')\nprint('saved')"
            ),
            "pip_install": ["numpy", "matplotlib"]
        }))
        .await;
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

    /// The domain allowlist documented on `SETUP_CMD` is the narrow posture that
    /// actually gets `python_repl` running inside a sandbox. Proving it needs a
    /// real VM: the setup script fetches over both plain HTTP (apt, matched from
    /// the gateway resolver's cache) and TLS (the uv release and PyPI, matched by
    /// SNI), and only the runtime exercises those two paths.
    ///
    /// The control is the second half — a domain nobody listed has to stay out of
    /// reach, or this would pass just as well against an accidentally-open
    /// policy. `test_host_only_blocks_public_egress` covers the other baseline:
    /// that `HostOnly` on its own reaches nothing outward.
    ///
    /// Ignored for the same reason the web_search tests are: it leans on three
    /// services this repo does not control, and one of them is a moving target
    /// — the setup script asks GitHub for the *latest* uv release, so a change
    /// to that release breaks this test without anything here changing. Run it
    /// when the posture or the setup script moves:
    ///
    /// ```text
    /// cargo test --features sandbox --lib -- --ignored sandbox_setup_under_domain_allowlist
    /// ```
    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[ignore = "slow: boots a VM and installs from apt, GitHub, and PyPI"]
    async fn sandbox_setup_under_domain_allowlist() {
        use crate::runenv::{SandboxBuilder, SandboxNetwork};

        let mut sandbox = SandboxBuilder::new()
            .network(SandboxNetwork::HostOnly.with_domain_suffixes([
                "ubuntu.com",
                "github.com",
                "githubusercontent.com",
                "pypi.org",
                "pythonhosted.org",
            ]))
            .build()
            .await
            .expect("build sandbox");
        let console = sandbox.start().await.expect("start sandbox");

        let provider = provider();
        let funcs = provider.provide(&[get_python_repl_tool_desc()]).unwrap();
        let f = funcs.get("python_repl").unwrap();

        // Bootstrap (apt -> uv release -> venv) plus a real wheel download.
        let msg = f
            .call(
                crate::to_value!({
                    "code": "import idna; print('idna', idna.__name__)",
                    "pip_install": ["idna"]
                }),
                "1",
                console,
            )
            .next()
            .await
            .unwrap()
            .message;
        let result = msg.contents[0].as_value().unwrap();
        assert_eq!(
            result.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(0),
            "setup and pip install must succeed under the documented posture: {result:?}"
        );

        // Control: a destination outside the allowlist stays unreachable, over
        // the same TLS path the allowed ones just used.
        let blocked = f
            .call(
                crate::to_value!({
                    "code": "import urllib.request; \
                             urllib.request.urlopen('https://example.com', timeout=15)"
                }),
                "2",
                console,
            )
            .next()
            .await
            .unwrap()
            .message;
        let blocked = blocked.contents[0].as_value().unwrap();
        assert_ne!(
            blocked.pointer("/exit_code").and_then(|v| v.as_integer()),
            Some(0),
            "an unlisted domain must not be reachable: {blocked:?}"
        );
        // Checked by kind, not just by exit code: a syntax error in the probe
        // would also exit non-zero and would look like a block. `urlopen` wraps
        // every connection-level OSError -- refusal, reset, timeout -- in
        // URLError, so this holds regardless of how the runtime drops it.
        let stderr = blocked
            .pointer("/stderr")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        assert!(
            stderr.contains("URLError"),
            "the probe must fail on the connection, not on the script: {stderr}"
        );
    }
}
