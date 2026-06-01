use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::{Console, ExecResult, Machine};

/// `Machine` that runs commands directly on the host. Start/stop are
/// no-ops because there is no underlying VM to spin up.
#[derive(Debug, Clone, Default)]
pub struct Local {}

impl Local {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Machine for Local {
    type Handle = LocalConsole;

    async fn start(&mut self) -> anyhow::Result<LocalConsole> {
        Ok(LocalConsole {})
    }

    async fn stop(&mut self) {}
}

pub struct LocalConsole {}

#[async_trait]
impl Console for LocalConsole {
    fn get_os(&self) -> &str {
        std::env::consts::OS
    }

    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        let mut command = tokio::process::Command::new(program);
        command.args(args).kill_on_drop(true);
        let result = if let Some(secs) = timeout {
            tokio::time::timeout(std::time::Duration::from_secs(secs), command.output()).await
        } else {
            Ok(command.output().await)
        };
        match result {
            Ok(Ok(out)) => Ok(ExecResult {
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                exit_code: out.status.code().unwrap_or(-1),
                timed_out: false,
            }),
            Ok(Err(e)) => Ok(ExecResult {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
                timed_out: false,
            }),
            Err(_) => Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
        }
    }

    async fn get_cwd(&self) -> anyhow::Result<PathBuf> {
        std::env::current_dir().map_err(|e| anyhow::anyhow!("get_cwd: {e}"))
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        tokio::fs::read(path)
            .await
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| anyhow::anyhow!("write {}: mkdir parent: {e}", path.display()))?;
        }
        tokio::fs::write(path, content)
            .await
            .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runenv::RunEnv;

    fn sh(cmd: &str) -> (String, Vec<String>) {
        ("sh".to_string(), vec!["-c".to_string(), cmd.to_string()])
    }

    #[tokio::test]
    async fn test_exec_stdout() {
        let env = RunEnv::new(Local::new());
        let handle = env.get().await.unwrap();
        let (prog, args) = sh("echo hello");
        let result = handle.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_exec_exit_code() {
        let env = RunEnv::new(Local::new());
        let handle = env.get().await.unwrap();
        let (prog, args) = sh("exit 42");
        let result = handle.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_exec_stderr() {
        let env = RunEnv::new(Local::new());
        let handle = env.get().await.unwrap();
        let (prog, args) = sh("echo err >&2");
        let result = handle.exec(prog, args, None).await.unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let env = RunEnv::new(Local::new());
        let handle = env.get().await.unwrap();
        let (prog, args) = sh("sleep 10");
        let result = handle.exec(prog, args, Some(1)).await.unwrap();
        assert!(result.timed_out);
    }

    #[tokio::test]
    async fn test_handle_shared_across_get_calls() {
        let env = RunEnv::new(Local::new());
        let h1 = env.get().await.unwrap();
        let h2 = env.get().await.unwrap();
        // While at least one handle is alive, `get()` must return the same
        // booted instance rather than reboot.
        assert!(std::sync::Arc::ptr_eq(&h1, &h2));
    }
}
