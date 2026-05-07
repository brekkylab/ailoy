use std::path::Path;

mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

#[derive(Debug, Clone)]
pub enum Dirent {
    Dir {
        name: String,
        permission: u8,
        children: Vec<Dirent>,
    },
    File {
        name: String,
        permission: u8,
        sz: usize,
    },
}

/// Execution result from a shell command.
#[derive(Debug, Clone)]
pub struct ExecResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

/// Execution environment for tools that touch the filesystem or run subprocesses.
///
/// An [`Agent`](crate::agent::Agent) holds an `Arc<dyn RunEnv>` in [`AgentState::runenv`](crate::agent::AgentState::runenv)
/// and passes it to every tool call via [`ToolContext`](crate::tool::ToolContext).  Sub-agents
/// declared in [`AgentSpec::subagents`](crate::agent::AgentSpec) inherit the parent's
/// `RunEnv`, so they share the same filesystem and process namespace.
///
/// Built-in implementations:
/// * [`Local`] — runs commands directly on the host (the default).
/// * [`Sandbox`] (with the `sandbox` feature) — runs commands inside a microVM.
#[async_trait::async_trait]
pub trait RunEnv: Send + Sync + 'static {
    /// Run `program` with `args`. `timeout` is in seconds; when elapsed the
    /// resulting [`ExecResult`] has `timed_out = true`.
    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult>;

    async fn ls(&self, path: &Path) -> anyhow::Result<Vec<Dirent>>;

    async fn mkdir(&self, path: &Path) -> anyhow::Result<()>;

    async fn rmdir(&self, path: &Path) -> anyhow::Result<()>;

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>>;

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()>;
}
