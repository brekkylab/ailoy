use std::path::Path;

use base64::{Engine as _, engine::general_purpose::STANDARD};

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

    /// Read a file's bytes.
    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let path_s = path.to_string_lossy();
        // shell out via base64 so any runenv with only `exec` can serve reads. Native impls override to skip the round-trip.
        let script = match self.get_os() {
            "linux" | "macos" => {
                format!("base64 < '{}'", path_s.replace('\'', "'\\''"))
            }
            "windows" => format!(
                "[Convert]::ToBase64String([IO.File]::ReadAllBytes('{}'))",
                path_s.replace('\'', "''")
            ),
            other => anyhow::bail!("read: unsupported OS '{other}'"),
        };

        let result = self.exec_shell(script, None).await?;
        if result.exit_code != 0 {
            anyhow::bail!(
                "read {} failed (exit {}): {}",
                path.display(),
                result.exit_code,
                result.stderr.trim(),
            );
        }

        // base64(1) wraps at ~76 cols; PowerShell emits a single line plus a
        // trailing newline. Strip all ASCII whitespace before decoding so both
        // shapes work with the strict STANDARD engine.
        let cleaned: String = result
            .stdout
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        STANDARD
            .decode(cleaned.as_bytes())
            .map_err(|e| anyhow::anyhow!("read {}: base64 decode failed: {e}", path.display()))
    }

    /// Write `content` to `path`, creating parent directories.
    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        // base64 in-process and pipe through the shell so any runenv with only `exec` can serve writes.
        let b64 = STANDARD.encode(content);
        let path_s = path.to_string_lossy();
        let parent_s = path.parent().map(|p| p.to_string_lossy().into_owned());

        let script = match self.get_os() {
            "linux" | "macos" => {
                // Here-doc keeps the (potentially large) base64 payload off the
                // command line, avoiding ARG_MAX limits. The delimiter is
                // single-quoted so the body is literal — no parameter expansion.
                let path_q = path_s.replace('\'', "'\\''");
                let mkdir = match &parent_s {
                    Some(p) if !p.is_empty() => {
                        format!("mkdir -p '{}' && ", p.replace('\'', "'\\''"))
                    }
                    _ => String::new(),
                };
                format!("{mkdir}base64 -d > '{path_q}' <<'AILOY_B64_EOF'\n{b64}\nAILOY_B64_EOF\n")
            }
            "windows" => {
                // PowerShell here-string `@'…'@` plays the same role as the
                // shell here-doc: literal, no interpolation.
                let path_q = path_s.replace('\'', "''");
                let mkdir = match &parent_s {
                    Some(p) if !p.is_empty() => format!(
                        "New-Item -ItemType Directory -Force -Path '{}' | Out-Null; ",
                        p.replace('\'', "''")
                    ),
                    _ => String::new(),
                };
                format!(
                    "{mkdir}$b64 = @'\n{b64}\n'@\n[IO.File]::WriteAllBytes('{path_q}', [Convert]::FromBase64String($b64))",
                )
            }
            other => anyhow::bail!("write: unsupported OS '{other}'"),
        };

        let result = self.exec_shell(script, None).await?;
        if result.exit_code != 0 {
            anyhow::bail!(
                "write {} failed (exit {}): {}",
                path.display(),
                result.exit_code,
                result.stderr.trim(),
            );
        }
        Ok(())
    }
}
