use std::{
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use tokio::sync::Mutex;

mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

#[async_trait]
pub trait Container: Send + Sync + 'static {
    type Handle: Console;

    async fn boot(&mut self) -> anyhow::Result<Self::Handle>;

    async fn shutdown(&mut self);
}

/// Object-safe erased view of `Container`. The blanket impl below adapts any
/// concrete `Container` by boxing the handle into `Arc<dyn Console>`.
#[async_trait]
trait ContainerDyn: Send + Sync + 'static {
    async fn boot(&mut self) -> anyhow::Result<Arc<dyn Console>>;

    async fn shutdown(&mut self);
}

#[async_trait]
impl<B: Container> ContainerDyn for B {
    async fn boot(&mut self) -> anyhow::Result<Arc<dyn Console>> {
        // UFCS so this doesn't recurse into the ContainerDyn impl we're in.
        Ok(Arc::new(Container::boot(self).await?))
    }

    async fn shutdown(&mut self) {
        Container::shutdown(self).await;
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

#[async_trait]
pub trait Console: Send + Sync {
    fn get_os(&self) -> &str;

    /// Run `program` with `args`. `timeout` is in seconds; when elapsed the
    /// resulting [`ExecResult`] has `timed_out = true`.
    ///
    /// Takes `&self` so a single booted handle can be shared by multiple
    /// `RunEnvHandle` clones. Implementations are responsible for their own
    /// internal synchronization (e.g. a Mutex around the stdin/stdout pipe).
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

    /// Return the current working directory.
    async fn get_cwd(&self) -> anyhow::Result<PathBuf> {
        let script = match self.get_os() {
            "linux" | "macos" => "pwd",
            "windows" => "(Get-Location).Path",
            other => anyhow::bail!("get_cwd: unsupported OS '{other}'"),
        };

        let result = self.exec_shell(script.to_string(), None).await?;
        if result.exit_code != 0 {
            anyhow::bail!(
                "get_cwd failed (exit {}): {}",
                result.exit_code,
                result.stderr.trim(),
            );
        }
        Ok(PathBuf::from(result.stdout.trim_end_matches(['\r', '\n'])))
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

enum RunEnvState {
    Idle,
    /// Holds a `Weak` so that when the last user-visible `RunEnvHandle` drops,
    /// the inner `Arc` count reaches zero and `RunenvInner::drop` fires.
    Running(Weak<RunEnvHandle>),
}

#[derive(Clone)]
pub struct RunEnv {
    machine: Arc<Mutex<dyn ContainerDyn>>,
    state: Arc<Mutex<RunEnvState>>,
}

impl RunEnv {
    pub fn new<B: Container>(machine: B) -> Self {
        Self {
            machine: Arc::new(Mutex::new(machine)),
            state: Arc::new(Mutex::new(RunEnvState::Idle)),
        }
    }

    pub fn local() -> Self {
        Self::new(Local::new())
    }

    #[cfg(feature = "sandbox")]
    pub async fn sandbox(config: SandboxConfig) -> anyhow::Result<Self> {
        Ok(Self::new(Sandbox::new(config).await?))
    }

    pub async fn get(&self) -> anyhow::Result<Arc<RunEnvHandle>> {
        loop {
            let mut s = self.state.lock().await;
            match &*s {
                RunEnvState::Idle => {
                    // Hold the state lock through boot; other callers block on
                    // `state.lock().await` and observe `Running` once we finish.
                    let console = self.machine.lock().await.boot().await?;
                    let handle = Arc::new(RunEnvHandle {
                        machine: self.machine.clone(),
                        console,
                        state: self.state.clone(),
                    });
                    *s = RunEnvState::Running(Arc::downgrade(&handle));
                    return Ok(handle);
                }
                RunEnvState::Running(weak) => {
                    if let Some(inner) = weak.upgrade() {
                        return Ok(inner);
                    }
                    // Last user handle just dropped but the `Drop`-spawned
                    // shutdown hasn't acquired the state lock yet. Release the
                    // state lock so it can make progress, then retry.
                    drop(s);
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
        }
    }
}

pub struct RunEnvHandle {
    machine: Arc<Mutex<dyn ContainerDyn>>,

    console: Arc<dyn Console>,

    state: Arc<Mutex<RunEnvState>>,
}

impl std::ops::Deref for RunEnvHandle {
    type Target = dyn Console;

    fn deref(&self) -> &Self::Target {
        &*self.console
    }
}

impl Drop for RunEnvHandle {
    fn drop(&mut self) {
        let machine = self.machine.clone();
        let state = self.state.clone();
        tokio::spawn(async move {
            // Hold the state lock through shutdown so concurrent `get()` calls
            // block until we transition back to `Idle`.
            let mut s = state.lock().await;
            machine.lock().await.shutdown().await;
            *s = RunEnvState::Idle;
        });
    }
}
