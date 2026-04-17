pub mod host;
#[cfg(feature = "sandbox-krun")]
pub mod krun;

use std::sync::Arc;

use async_trait::async_trait;

#[derive(Clone, Debug)]
pub struct ExecRequest {
    pub command: String,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

#[async_trait]
pub trait Sandbox: Send + Sync {
    async fn exec(&self, request: ExecRequest) -> anyhow::Result<ExecResult>;
    async fn shutdown(&self) -> anyhow::Result<()>;
}

#[derive(Clone)]
pub struct ToolContext {
    pub sandbox: Option<Arc<dyn Sandbox>>,
}

impl ToolContext {
    pub fn empty() -> Self {
        Self { sandbox: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::host::{HostSandbox, HostSandboxConfig};

    #[tokio::test]
    async fn test_host_sandbox_echo() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "echo hello".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim() == "hello");
    }

    #[tokio::test]
    async fn test_host_sandbox_stderr() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "echo err >&2".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stderr.trim() == "err");
    }

    #[tokio::test]
    async fn test_host_sandbox_exit_code() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "exit 42".to_string(),
                timeout_secs: 10,
            })
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_host_sandbox_timeout() {
        let sandbox = HostSandbox::new(HostSandboxConfig::default());
        let result = sandbox
            .exec(ExecRequest {
                command: "sleep 100".to_string(),
                timeout_secs: 1,
            })
            .await
            .unwrap();
        assert!(result.timed_out);
        assert_eq!(result.exit_code, -1);
    }
}
