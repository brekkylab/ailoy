use std::{collections::HashSet, path::PathBuf};

use anyhow::{Context as _, Result, bail};
use tokio::{process::Command, sync::Mutex};

/// Result of a `pip install` step.
#[derive(Debug)]
pub enum InstallResult {
    /// Every requested specifier was already tracked in the in-memory cache —
    /// no subprocess was spawned.
    AlreadyInstalled,
    /// All packages were installed (or upgraded) successfully.
    Success,
    /// `uv pip install` exited non-zero.
    Failed { stderr: String },
}

/// Result of executing a Python script.
#[derive(Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// A managed Python virtual environment backed by `uv`.
pub struct PythonEnv {
    /// Path to the `uv` binary.
    uv: PathBuf,
    /// Requested Python version string (e.g. `"3.12"`).  `None` → latest stable.
    python_version: Option<String>,
    /// Root directory of the virtual environment.
    venv_path: PathBuf,
    /// In-memory cache of specifier strings that have been successfully installed,
    /// used to avoid redundant subprocess calls.
    installed: Mutex<HashSet<String>>,
    /// When `true` the venv directory is deleted on [`Drop`].
    is_temp: bool,
}

impl PythonEnv {
    /// Create (or reuse) a virtual environment at `venv_path`.
    ///
    /// If `is_temp` is `true` the directory will be deleted when this
    /// [`PythonEnv`] is dropped.
    pub async fn new(
        uv: PathBuf,
        python_version: Option<String>,
        venv_path: PathBuf,
        is_temp: bool,
    ) -> Result<Self> {
        let env = Self {
            uv,
            python_version,
            venv_path,
            installed: Mutex::new(HashSet::new()),
            is_temp,
        };
        env.ensure_python().await?;
        env.ensure_venv().await?;
        Ok(env)
    }

    /// Create a virtual environment in a temporary directory that is
    /// automatically cleaned up on drop.
    pub async fn new_temp(uv: PathBuf, python_version: Option<String>) -> Result<Self> {
        let tmp = tempfile::tempdir().context("failed to create temp dir for venv")?;
        // `into_path` gives us the PathBuf without deleting the dir on drop —
        // we handle cleanup ourselves in `PythonEnv::drop`.
        let venv_path = tmp.keep().join("venv");
        Self::new(uv, python_version, venv_path, true).await
    }

    /// Path to the Python interpreter inside the venv.
    pub fn python_path(&self) -> PathBuf {
        if cfg!(target_os = "windows") {
            self.venv_path.join("Scripts").join("python.exe")
        } else {
            self.venv_path.join("bin").join("python")
        }
    }

    // ── private helpers ──────────────────────────────────────────────────────

    async fn ensure_python(&self) -> Result<()> {
        let version_arg = self.python_version.as_deref().unwrap_or("3");
        let status = Command::new(&self.uv)
            .args(["python", "install", version_arg])
            .status()
            .await
            .context("failed to spawn `uv python install`")?;
        if !status.success() {
            bail!("uv python install {} failed", version_arg);
        }
        Ok(())
    }

    async fn ensure_venv(&self) -> Result<()> {
        if self.python_path().exists() {
            return Ok(());
        }
        let mut cmd = Command::new(&self.uv);
        cmd.args(["venv", self.venv_path.to_str().unwrap()]);
        if let Some(ver) = &self.python_version {
            cmd.args(["--python", ver.as_str()]);
        }
        let status = cmd.status().await.context("failed to spawn `uv venv`")?;
        if !status.success() {
            bail!("uv venv failed for {:?}", self.venv_path);
        }
        Ok(())
    }

    /// Install pip packages into this venv.
    ///
    /// Specifiers that are already present in the in-memory cache are skipped;
    /// for everything else `uv pip install` is called.
    pub async fn install_packages(&self, packages: &[String]) -> Result<InstallResult> {
        if packages.is_empty() {
            return Ok(InstallResult::AlreadyInstalled);
        }

        let mut cache = self.installed.lock().await;
        let new: Vec<&String> = packages
            .iter()
            .filter(|p| !cache.contains(p.as_str()))
            .collect();

        if new.is_empty() {
            return Ok(InstallResult::AlreadyInstalled);
        }

        let output = Command::new(&self.uv)
            .args([
                "pip",
                "install",
                "--python",
                self.python_path().to_str().unwrap(),
            ])
            .args(new.iter().copied())
            .output()
            .await
            .context("failed to spawn `uv pip install`")?;

        if output.status.success() {
            for spec in &new {
                cache.insert(spec.to_string());
            }
            Ok(InstallResult::Success)
        } else {
            Ok(InstallResult::Failed {
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            })
        }
    }

    /// Execute a Python script string, returning captured stdout/stderr and
    /// the exit code.
    ///
    /// The code is written to a temporary file and passed to the venv's Python
    /// interpreter.  The temp file is deleted after the child exits.
    pub async fn run_code(&self, code: &str, timeout_secs: u64) -> Result<ExecResult> {
        // Write code to a temp file.
        let script = tempfile::NamedTempFile::with_suffix(".py")
            .context("failed to create temp script file")?;
        std::fs::write(script.path(), code).context("failed to write script to temp file")?;

        let child = Command::new(self.python_path())
            .arg(script.path())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .context("failed to spawn Python interpreter")?;

        let deadline = tokio::time::Duration::from_secs(timeout_secs);
        let output = tokio::time::timeout(deadline, child.wait_with_output())
            .await
            .context("Python script timed out")?
            .context("failed to wait for Python process")?;

        Ok(ExecResult {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        })
    }
}

impl Drop for PythonEnv {
    fn drop(&mut self) {
        if self.is_temp {
            // Best-effort: ignore errors (the process may already be exiting).
            if let Some(parent) = self.venv_path.parent() {
                let _ = std::fs::remove_dir_all(parent);
            }
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    impl PythonEnv {
        /// Expose the venv directory path (useful in tests).
        pub fn venv_path(&self) -> &Path {
            &self.venv_path
        }
    }

    /// Skip the test if `uv` is not available on PATH.
    fn uv_or_skip() -> PathBuf {
        match which::which("uv") {
            Ok(p) => p,
            Err(_) => {
                eprintln!("SKIP: `uv` not found on PATH");
                // Returning an obviously wrong path causes the test to be
                // skipped via the ignore mechanism — we use `#[ignore]` below.
                PathBuf::from("/uv-not-found")
            }
        }
    }

    // ── venv creation ────────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_create_temp_env_venv_dir_exists() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        assert!(env.venv_path().exists(), "venv dir should exist");
        assert!(env.python_path().exists(), "python binary should exist");
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_temp_env_is_cleaned_up_on_drop() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        // Capture venv parent (the temp dir) before dropping.
        let parent = env.venv_path().parent().unwrap().to_path_buf();
        assert!(parent.exists());
        drop(env);
        assert!(!parent.exists(), "temp dir should be deleted on drop");
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_persistent_env_not_cleaned_up_on_drop() {
        let uv = uv_or_skip();
        let dir = tempfile::tempdir().unwrap();
        let venv_path = dir.path().join("venv");
        let env = PythonEnv::new(uv, None, venv_path.clone(), false)
            .await
            .unwrap();
        drop(env);
        assert!(venv_path.exists(), "persistent venv should survive drop");
    }

    // ── package installation ─────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_install_package_success() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env
            .install_packages(&["requests".to_string()])
            .await
            .unwrap();
        assert!(matches!(result, InstallResult::Success));
    }

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_install_same_package_twice_uses_cache() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        env.install_packages(&["requests".to_string()])
            .await
            .unwrap();
        // Second call with identical specifier must not spawn subprocess.
        let result = env
            .install_packages(&["requests".to_string()])
            .await
            .unwrap();
        assert!(matches!(result, InstallResult::AlreadyInstalled));
    }

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_install_version_specifier_change_reinstalls() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        env.install_packages(&["requests==2.31.0".to_string()])
            .await
            .unwrap();
        // Different specifier string → subprocess must be called.
        let result = env
            .install_packages(&["requests==2.32.0".to_string()])
            .await
            .unwrap();
        assert!(matches!(result, InstallResult::Success));
    }

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_install_nonexistent_package_returns_failed() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env
            .install_packages(&["this-package-definitely-does-not-exist-xyzzy".to_string()])
            .await
            .unwrap();
        assert!(
            matches!(result, InstallResult::Failed { .. }),
            "expected Failed, got {:?}",
            result
        );
    }

    #[tokio::test]
    #[ignore = "requires uv + network"]
    async fn test_install_empty_list_returns_already_installed() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env.install_packages(&[]).await.unwrap();
        assert!(matches!(result, InstallResult::AlreadyInstalled));
    }

    // ── code execution ───────────────────────────────────────────────────────

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_hello_world() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env.run_code("print('hello world')", 30).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(
            result.stdout.contains("hello world"),
            "stdout: {:?}",
            result.stdout
        );
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_code_captures_stderr() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env
            .run_code("import sys; sys.stderr.write('err output\\n')", 30)
            .await
            .unwrap();
        assert!(
            result.stderr.contains("err output"),
            "stderr: {:?}",
            result.stderr
        );
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_code_runtime_error_nonzero_exit() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        let result = env.run_code("raise ValueError('oops')", 30).await.unwrap();
        assert_ne!(result.exit_code, 0, "expected non-zero exit for exception");
        assert!(
            result.stderr.contains("ValueError"),
            "stderr: {:?}",
            result.stderr
        );
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_code_each_execution_is_stateless() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        // First run: define a variable.
        let first = env.run_code("x = 42", 30).await.unwrap();
        assert_eq!(first.exit_code, 0);
        // Second run: variable must NOT be visible.
        let second = env.run_code("print(x)", 30).await.unwrap();
        assert_ne!(second.exit_code, 0, "x should not exist in a fresh run");
    }

    #[tokio::test]
    #[ignore = "requires uv"]
    async fn test_run_code_timeout() {
        let uv = uv_or_skip();
        let env = PythonEnv::new_temp(uv, None).await.unwrap();
        // 1-second timeout against an infinite loop.
        let result = env.run_code("while True: pass", 1).await;
        assert!(result.is_err(), "expected timeout error");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("timed out"), "error: {}", msg);
    }
}
