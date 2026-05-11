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

impl Dirent {
    pub fn name(&self) -> &str {
        match self {
            Dirent::Dir { name, .. } | Dirent::File { name, .. } => name,
        }
    }

    pub fn permission(&self) -> u8 {
        match self {
            Dirent::Dir { permission, .. } | Dirent::File { permission, .. } => *permission,
        }
    }

    pub fn is_dir(&self) -> bool {
        matches!(self, Dirent::Dir { .. })
    }

    pub fn is_file(&self) -> bool {
        matches!(self, Dirent::File { .. })
    }

    pub fn children(&self) -> Option<&[Dirent]> {
        match self {
            Dirent::Dir { children, .. } => Some(children),
            Dirent::File { .. } => None,
        }
    }

    pub fn size(&self) -> Option<usize> {
        match self {
            Dirent::File { sz, .. } => Some(*sz),
            Dirent::Dir { .. } => None,
        }
    }
}

impl std::fmt::Display for Dirent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn perm_chars(p: u8) -> [char; 3] {
            [
                if p & 0b100 != 0 { 'r' } else { '-' },
                if p & 0b010 != 0 { 'w' } else { '-' },
                if p & 0b001 != 0 { 'x' } else { '-' },
            ]
        }
        fn write_entry(
            d: &Dirent,
            f: &mut std::fmt::Formatter<'_>,
            depth: usize,
            first: &mut bool,
        ) -> std::fmt::Result {
            if !*first {
                writeln!(f)?;
            }
            *first = false;
            for _ in 0..depth {
                f.write_str("  ")?;
            }
            let [r, w, x] = perm_chars(d.permission());
            match d {
                Dirent::Dir { name, children, .. } => {
                    write!(f, "{r}{w}{x} {name}/")?;
                    for child in children {
                        write_entry(child, f, depth + 1, first)?;
                    }
                    Ok(())
                }
                Dirent::File { name, sz, .. } => {
                    write!(f, "{r}{w}{x} {name} ({sz} bytes)")
                }
            }
        }
        let mut first = true;
        write_entry(self, f, 0, &mut first)
    }
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
