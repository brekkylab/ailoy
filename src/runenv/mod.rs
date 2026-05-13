use std::{
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

#[derive(Debug, Clone)]
pub enum FSEntry {
    Dir {
        children: Vec<String>,
        created_at: SystemTime,
        updated_at: SystemTime,
    },
    File {
        permission: u8,
        sz: usize,
        created_at: SystemTime,
        updated_at: SystemTime,
    },
}

impl std::fmt::Display for FSEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FSEntry::Dir { children, .. } => write!(f, "dir ({} entries)", children.len()),
            FSEntry::File { permission, sz, .. } => {
                let r = if permission & 0b100 != 0 { 'r' } else { '-' };
                let w = if permission & 0b010 != 0 { 'w' } else { '-' };
                let x = if permission & 0b001 != 0 { 'x' } else { '-' };
                write!(f, "{r}{w}{x} ({sz} bytes)")
            }
        }
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
    /// `linux`, `macos`, `windows`...
    fn get_os(&self) -> &str;

    /// Run `program` with `args`. `timeout` is in seconds; when elapsed the
    /// resulting [`ExecResult`] has `timed_out = true`.
    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult>;

    /// Inspect the entry in path, and returns metadata.
    async fn stat(&self, path: &Path) -> anyhow::Result<FSEntry> {
        let path_str = path.to_string_lossy().into_owned();
        // Single-quote the path for POSIX `sh -c`, escaping any inner single quotes.
        let quoted_path = format!("'{}'", path_str.replace('\'', "'\\''"));

        // First line: kind \t mode \t size \t mtime \t btime
        // Then (if directory): one child basename per line via `ls -1A`.
        let script = match self.get_os() {
            "linux" => format!(
                "stat -c '%F\\t%a\\t%s\\t%Y\\t%W' {quoted_path} && \
                 if [ -d {quoted_path} ]; then ls -1A {quoted_path}; fi",
            ),
            "macos" => format!(
                "stat -f '%HT\\t%Lp\\t%z\\t%m\\t%B' {quoted_path} && \
                 if [ -d {quoted_path} ]; then ls -1A {quoted_path}; fi",
            ),
            other => anyhow::bail!("inspect: blanket impl does not support OS '{other}'"),
        };

        let result = self
            .exec("sh".into(), vec!["-c".into(), script], None)
            .await?;
        if result.exit_code != 0 {
            anyhow::bail!(
                "inspect {} failed (exit {}): {}",
                path.display(),
                result.exit_code,
                result.stderr,
            );
        }

        let mut lines = result.stdout.lines();
        let meta_line = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("inspect {}: empty output", path.display()))?;
        let parts: Vec<&str> = meta_line.split('\t').collect();
        if parts.len() < 5 {
            anyhow::bail!(
                "inspect {}: malformed metadata line {meta_line:?}",
                path.display(),
            );
        }
        let (kind, mode, size, mtime, btime) = (parts[0], parts[1], parts[2], parts[3], parts[4]);

        // Owner permission bits: take the top of the user nibble from the
        // trailing octal triple of `mode` (e.g. "0644"/"755" → 6/7).
        let permission: u8 = {
            let trailing3 = mode.chars().rev().take(3).collect::<Vec<_>>();
            let first = trailing3.last().and_then(|c| c.to_digit(8)).unwrap_or(0);
            (first & 0o7) as u8
        };

        // `stat -c '%Y'` / `stat -f '%m'` emit Unix epoch seconds; reject NaN/negative.
        // Linux `%W` returns 0 when birth time is unknown (e.g. ext4 without statx).
        let parse_epoch = |s: &str| -> SystemTime {
            let secs: f64 = s.parse().unwrap_or(0.0);
            if secs.is_finite() && secs >= 0.0 {
                let whole = secs.trunc() as u64;
                let nanos = (secs.fract() * 1_000_000_000.0).round().abs() as u32;
                UNIX_EPOCH + Duration::new(whole, nanos.min(999_999_999))
            } else {
                UNIX_EPOCH
            }
        };
        let updated_at = parse_epoch(mtime);
        let created_at = parse_epoch(btime);

        let is_dir = match self.get_os() {
            "linux" => kind == "directory",
            "macos" => kind == "Directory",
            _ => false,
        };

        if is_dir {
            let children = lines.map(|s| s.to_string()).collect();
            Ok(FSEntry::Dir {
                children,
                created_at,
                updated_at,
            })
        } else {
            let sz: usize = size.parse().unwrap_or(0);
            Ok(FSEntry::File {
                permission,
                sz,
                created_at,
                updated_at,
            })
        }
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>>;

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()>;
}
