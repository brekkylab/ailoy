use std::path::Path;

mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

/// Execution result from a shell command.
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

#[async_trait::async_trait]
pub trait RunEnv: Send + Sync {
    /// timeout: secs
    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult>;

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>>;

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()>;
}
