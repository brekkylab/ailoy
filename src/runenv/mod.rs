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
        readable: bool,
        writable: bool,
        created_at: SystemTime,
        updated_at: SystemTime,
    },
    File {
        readable: bool,
        writable: bool,
        executable: bool,
        sz: usize,
        created_at: SystemTime,
        updated_at: SystemTime,
    },
}

impl std::fmt::Display for FSEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FSEntry::Dir {
                children,
                readable,
                writable,
                ..
            } => {
                let r = if *readable { 'r' } else { '-' };
                let w = if *writable { 'w' } else { '-' };
                write!(f, "dir [{r}{w}-, {} entries]", children.len())
            }
            FSEntry::File {
                readable,
                writable,
                executable,
                sz,
                ..
            } => {
                let r = if *readable { 'r' } else { '-' };
                let w = if *writable { 'w' } else { '-' };
                let x = if *executable { 'x' } else { '-' };
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

    /// Run `script` through the system shell
    ///
    /// `bash -c` on Linux/macOS, `powershell -Command` on Windows.
    async fn exec_shell(&self, script: String, timeout: Option<u64>) -> anyhow::Result<ExecResult> {
        let (program, args) = match self.get_os() {
            "linux" | "macos" => ("bash".to_string(), vec!["-c".to_string(), script]),
            "windows" => (
                "powershell".to_string(),
                vec!["-Command".to_string(), script],
            ),
            other => anyhow::bail!("exec_shell: unsupported OS '{other}'"),
        };
        self.exec(program, args, timeout).await
    }

    /// Inspect the entry in path, and returns metadata.
    async fn stat(&self, path: &Path) -> anyhow::Result<FSEntry> {
        let path_str = path.to_string_lossy().into_owned();

        // First line: kind \t mode \t size \t mtime \t btime
        // Then (if directory): one child basename per line.
        let script = match self.get_os() {
            "linux" => {
                // Single-quote the path for POSIX shells, escaping any inner single quotes.
                let q = format!("'{}'", path_str.replace('\'', "'\\''"));
                format!(
                    "stat -c '%F\\t%a\\t%s\\t%Y\\t%W' {q} && \
                     if [ -d {q} ]; then ls -1A {q}; fi",
                )
            }
            "macos" => {
                let q = format!("'{}'", path_str.replace('\'', "'\\''"));
                format!(
                    "stat -f '%HT\\t%Lp\\t%z\\t%m\\t%B' {q} && \
                     if [ -d {q} ]; then ls -1A {q}; fi",
                )
            }
            // Windows has no POSIX mode, so emit just the owner octal digit
            // (`4` = r--, `6` = rw-). `kind` is lowercase to share the
            // `is_dir` check with linux.
            "windows" => {
                // PowerShell single-quoted strings escape `'` as `''`.
                let q = format!("'{}'", path_str.replace('\'', "''"));
                format!(
                    "$ErrorActionPreference='Stop'; \
                     $i=Get-Item -LiteralPath {q} -Force; \
                     $d=$i.PSIsContainer; \
                     $k=if($d){{'directory'}}else{{'file'}}; \
                     $s=if($d){{0}}else{{$i.Length}}; \
                     $m=if($i.IsReadOnly){{'4'}}else{{'6'}}; \
                     $u=([DateTimeOffset]$i.LastWriteTimeUtc).ToUnixTimeSeconds(); \
                     $b=([DateTimeOffset]$i.CreationTimeUtc).ToUnixTimeSeconds(); \
                     \"$k`t$m`t$s`t$u`t$b\"; \
                     if($d){{ Get-ChildItem -LiteralPath {q} -Force -Name }}",
                )
            }
            other => anyhow::bail!("inspect: blanket impl does not support OS '{other}'"),
        };

        let result = self.exec_shell(script, None).await?;
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

        // Owner permission bits: take the user nibble from the trailing octal
        // triple of `mode` (e.g. "0644"/"755" → 6/7), then split into the
        // readable (0b100), writable (0b010), and executable (0b001) bits.
        let (readable, writable, executable) = {
            let trailing3 = mode.chars().rev().take(3).collect::<Vec<_>>();
            let first = trailing3.last().and_then(|c| c.to_digit(8)).unwrap_or(0);
            let bits = (first & 0o7) as u8;
            (bits & 0b100 != 0, bits & 0b010 != 0, bits & 0b001 != 0)
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
            "linux" | "windows" => kind == "directory",
            "macos" => kind == "Directory",
            _ => false,
        };

        if is_dir {
            let children = lines.map(|s| s.to_string()).collect();
            Ok(FSEntry::Dir {
                children,
                readable,
                writable,
                created_at,
                updated_at,
            })
        } else {
            let sz: usize = size.parse().unwrap_or(0);
            Ok(FSEntry::File {
                readable,
                writable,
                executable,
                sz,
                created_at,
                updated_at,
            })
        }
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>>;

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()>;
}
