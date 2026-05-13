use std::path::{Path, PathBuf};

use crate::runenv::{ExecResult, RunEnv};

#[derive(Debug, Clone, Default)]
pub struct Local {}

impl Local {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl RunEnv for Local {
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
        command.args(args);
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

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        tokio::fs::read(path).await.map_err(Into::into)
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, content).await.map_err(Into::into)
    }

    async fn get_cwd(&self) -> anyhow::Result<PathBuf> {
        Ok(std::env::current_dir()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local() -> Local {
        Local {}
    }

    fn sh(cmd: &str) -> (String, Vec<String>) {
        ("sh".to_string(), vec!["-c".to_string(), cmd.to_string()])
    }

    #[tokio::test]
    async fn test_exec_stdout() {
        let (prog, args) = sh("echo hello");
        let result = local().exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_exec_exit_code() {
        let (prog, args) = sh("exit 42");
        let result = local().exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_exec_stderr() {
        let (prog, args) = sh("echo err >&2");
        let result = local().exec(prog, args, None).await.unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let (prog, args) = sh("sleep 10");
        let result = local().exec(prog, args, Some(1)).await.unwrap();
        assert!(result.timed_out);
    }

    #[tokio::test]
    async fn test_write_read_roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();
        local().write(path, b"ailoy").await.unwrap();
        let bytes = local().read(path).await.unwrap();
        assert_eq!(bytes, b"ailoy");
    }

    #[tokio::test]
    async fn test_write_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a/b/c/file.txt");
        local().write(&path, b"nested").await.unwrap();
        let bytes = local().read(&path).await.unwrap();
        assert_eq!(bytes, b"nested");
    }
}
