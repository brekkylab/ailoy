use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::Mutex;

use crate::sandbox::{ExecResult, Sandbox};

/// Shared helper that runs a Python script either inside a MicroVM sandbox or
/// on the host process.  Constructed once at tool-factory time; `sandbox` is
/// supplied per call so the runner itself stays `Send + Sync + 'static`.
///
/// ## Lifecycle
/// 1. Call [`PythonScriptRunner::ensure_setup`] once (lazy, guarded by a mutex)
///    to run any one-time shell setup (e.g. `pip install docling`).
/// 2. Optionally call [`PythonScriptRunner::install_packages`] per call for
///    dynamic pip installs (used by `python_repl`).
/// 3. Call [`PythonScriptRunner::run`] with the script source and env vars.
pub struct PythonScriptRunner {
    initialized: Arc<Mutex<bool>>,
    setup_script: Option<String>,
    setup_timeout_secs: u64,
}

impl PythonScriptRunner {
    /// `setup_script` — optional shell command run once on first use (e.g. pip install).
    /// `setup_timeout_secs` — timeout for the setup command; ignored when `setup_script` is `None`.
    pub fn new(setup_script: impl Into<Option<String>>, setup_timeout_secs: u64) -> Self {
        Self {
            initialized: Arc::new(Mutex::new(false)),
            setup_script: setup_script.into(),
            setup_timeout_secs,
        }
    }

    /// Run the setup script exactly once.  Subsequent calls are no-ops.
    pub async fn ensure_setup(&self, sandbox: Option<&Arc<Sandbox>>) -> anyhow::Result<()> {
        let Some(script) = &self.setup_script else {
            return Ok(());
        };
        let mut guard = self.initialized.lock().await;
        if *guard {
            return Ok(());
        }
        let result = self
            .run_shell(sandbox, script, Some(self.setup_timeout_secs))
            .await
            .context("setup script failed")?;
        if result.timed_out {
            anyhow::bail!("setup timed out after {}s", self.setup_timeout_secs);
        }
        if result.exit_code != 0 {
            anyhow::bail!(
                "setup failed (exit {}): {}",
                result.exit_code,
                result.stderr.trim()
            );
        }
        *guard = true;
        Ok(())
    }

    /// Install pip packages.  Returns immediately (success) when `packages` is empty.
    pub async fn install_packages(
        &self,
        sandbox: Option<&Arc<Sandbox>>,
        packages: &[&str],
    ) -> anyhow::Result<ExecResult> {
        if packages.is_empty() {
            return Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
            });
        }
        let quoted: Vec<String> = packages.iter().map(|p| format!("'{p}'")).collect();
        let cmd = format!("pip install {}", quoted.join(" "));
        self.run_shell(sandbox, &cmd, None).await
    }

    /// Write `source` to a temporary file, execute it with `env`, then delete the file.
    pub async fn run(
        &self,
        sandbox: Option<&Arc<Sandbox>>,
        source: &str,
        env: &[(&str, &str)],
    ) -> anyhow::Result<ExecResult> {
        self.run_with_timeout(sandbox, source, env, 0).await
    }

    /// Like [`run`] but enforces a per-execution timeout.
    /// `timeout_secs = 0` means no timeout.
    pub async fn run_with_timeout(
        &self,
        sandbox: Option<&Arc<Sandbox>>,
        source: &str,
        env: &[(&str, &str)],
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        if let Some(sb) = sandbox {
            self.run_in_sandbox(sb, source, env, timeout_secs).await
        } else {
            self.run_on_host(source, env, timeout_secs).await
        }
    }

    // -------------------------------------------------------------------------

    async fn run_in_sandbox(
        &self,
        sb: &Arc<Sandbox>,
        source: &str,
        env: &[(&str, &str)],
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        let script_path = format!("/tmp/__ailoy_run_{}.py", uuid::Uuid::new_v4());

        sb.write_file(&script_path, source.as_bytes())
            .await
            .context("failed to write script to sandbox")?;

        let env_prefix = env
            .iter()
            .map(|(k, v)| format!("{k}='{v}'"))
            .collect::<Vec<_>>()
            .join(" ");
        let cmd = if env_prefix.is_empty() {
            format!("python3 {script_path}")
        } else {
            format!("{env_prefix} python3 {script_path}")
        };

        let result = if timeout_secs > 0 {
            sb.shell_with_timeout(&cmd, timeout_secs).await
        } else {
            sb.shell(&cmd).await
        };
        let _ = sb.shell(&format!("rm -f {script_path}")).await;
        result.context("sandbox execution error")
    }

    async fn run_on_host(
        &self,
        source: &str,
        env: &[(&str, &str)],
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        let script_path =
            std::env::temp_dir().join(format!("__ailoy_run_{}.py", uuid::Uuid::new_v4()));

        std::fs::write(&script_path, source.as_bytes()).context("failed to write script")?;

        let mut cmd = tokio::process::Command::new("python3");
        cmd.arg(&script_path);
        for (k, v) in env {
            cmd.env(k, v);
        }

        let result = if timeout_secs > 0 {
            match tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
            {
                Ok(r) => r.context("failed to run python3")?,
                Err(_) => {
                    let _ = std::fs::remove_file(&script_path);
                    return Ok(ExecResult {
                        stdout: String::new(),
                        stderr: String::new(),
                        exit_code: -1,
                        timed_out: true,
                    });
                }
            }
        } else {
            cmd.output().await.context("failed to run python3")?
        };

        let _ = std::fs::remove_file(&script_path);
        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&result.stderr).into_owned(),
            exit_code: result.status.code().unwrap_or(-1),
            timed_out: false,
        })
    }

    async fn run_shell(
        &self,
        sandbox: Option<&Arc<Sandbox>>,
        script: &str,
        timeout_secs: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        if let Some(sb) = sandbox {
            if let Some(t) = timeout_secs {
                sb.shell_with_timeout(script, t).await
            } else {
                sb.shell(script).await
            }
            .context("sandbox shell error")
        } else {
            let out = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(script)
                .output()
                .await
                .context("host shell error")?;
            Ok(ExecResult {
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                exit_code: out.status.code().unwrap_or(-1),
                timed_out: false,
            })
        }
    }
}
