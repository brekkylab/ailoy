pub mod host;
#[cfg(feature = "sandbox-krun")]
pub mod krun;

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
