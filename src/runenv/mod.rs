//! Running environment abstractions.
//!
//! A `RunEnv` represents a computing environment that an LLM can interact
//! with — a uniform abstraction over the local machine, a sandboxed
//! container, or any other backend that implements the traits below.
//!
//! This module is designed primarily as the execution substrate for agents.
//!
//! # Usage
//!
//! Construct a `RunEnv` for the desired backend, then call [`RunEnv::get`]
//! to obtain a handle and drive the environment through it:
//!
//! ```ignore
//! # async fn run() -> anyhow::Result<()> {
//! use ailoy::runenv::RunEnv;
//!
//! let env = RunEnv::local();
//! let handle = env.get().await?;
//!
//! let result = handle.exec_shell("echo hello".to_string(), None).await?;
//! println!("{}", result.stdout);
//! # Ok(()) }
//! ```
//!
//! For an isolated environment, use the `sandbox` feature and pass a
//! [`SandboxConfig`]:
//!
//! ```ignore
//! # async fn run() -> anyhow::Result<()> {
//! use ailoy::runenv::{RunEnv, SandboxConfig};
//!
//! let env = RunEnv::sandbox(SandboxConfig::default()).await?;
//! let handle = env.get().await?;
//!
//! let result = handle.exec_shell("uname -a".to_string(), None).await?;
//! println!("{}", result.stdout);
//! # Ok(()) }
//! ```
//!
//! # Lifecycle
//!
//! A `RunEnv` is cheap to construct and starts out idle. The underlying
//! machine is only booted on demand: callers obtain a [`RunEnvHandle`] via
//! [`RunEnv::get`], which lazily boots the container on the first call and
//! reuses the same booted handle for subsequent callers. When the last
//! outstanding `RunEnvHandle` is dropped, the container is shut down
//! automatically and the `RunEnv` returns to the idle state, ready to be
//! booted again on the next `get()`.
//!
//! All operations against the environment go through the handle, which
//! derefs to [`Console`].
//!
//! # Extending with a new runenv
//!
//! To add a new kind of running environment, implement two traits:
//!
//! - [`Machine`]: describes how to boot and shut down the underlying
//!   machine. `boot` returns a handle type that implements [`Console`].
//! - [`Console`]: the booted handle. At minimum, implement
//!   [`Console::get_os`] and [`Console::exec`]; the other methods
//!   (`exec_shell`, `get_cwd`, `read`, `write`) have default implementations
//!   built on top of `exec`, but may be overridden for efficiency when a
//!   backend can serve them natively without shelling out.
//!
//! A minimal skeleton looks like this:
//!
//! ```ignore
//! use async_trait::async_trait;
//! use ailoy::runenv::{Console, Machine, ExecResult, RunEnv};
//!
//! struct MyMachine { /* connection state, config, ... */ }
//! struct MyConsole { /* booted session, e.g. an SSH channel */ }
//!
//! #[async_trait]
//! impl Machine for MyMachine {
//!     type Handle = MyConsole;
//!
//!     async fn boot(&mut self) -> anyhow::Result<Self::Handle> {
//!         // open connection, spawn shell, etc.
//!         Ok(MyConsole { /* ... */ })
//!     }
//!
//!     async fn shutdown(&mut self) {
//!         // close connection, kill process, etc.
//!     }
//! }
//!
//! #[async_trait]
//! impl Console for MyConsole {
//!     fn get_os(&self) -> &str { "linux" }
//!
//!     async fn exec(
//!         &self,
//!         program: String,
//!         args: Vec<String>,
//!         timeout: Option<u64>,
//!     ) -> anyhow::Result<ExecResult> {
//!         // run the command on the backend and collect stdout/stderr/exit
//!         todo!()
//!     }
//! }
//!
//! # async fn run() -> anyhow::Result<()> {
//! let env = RunEnv::new(MyMachine { /* ... */ });
//! let handle = env.get().await?;
//! let result = handle.exec_shell("echo hi".to_string(), None).await?;
//! println!("{}", result.stdout);
//! # Ok(()) }
//! ```

use std::{
    path::{Path, PathBuf},
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use tokio::sync::Mutex;

mod file_entry;
mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use file_entry::*;
pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

/// Backend that knows how to boot and tear down a running environment,
/// yielding a [`Console`] handle for the booted instance.
///
/// See the [module-level documentation](self) for the overall design.
#[async_trait]
pub trait Machine: Send + Sync + 'static {
    type Handle: Console;

    async fn boot(&mut self) -> anyhow::Result<Self::Handle>;

    async fn shutdown(&mut self);
}

/// Object-safe erased view of `Machine`. The blanket impl below adapts any
/// concrete `Machine` by boxing the handle into `Arc<dyn Console>`.
#[async_trait]
trait MachineDyn: Send + Sync + 'static {
    async fn boot(&mut self) -> anyhow::Result<Arc<dyn Console>>;

    async fn shutdown(&mut self);
}

#[async_trait]
impl<B: Machine> MachineDyn for B {
    async fn boot(&mut self) -> anyhow::Result<Arc<dyn Console>> {
        // UFCS so this doesn't recurse into the MachineDyn impl we're in.
        Ok(Arc::new(Machine::boot(self).await?))
    }

    async fn shutdown(&mut self) {
        Machine::shutdown(self).await;
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

/// Handle to a booted running environment, exposing command execution and
/// file I/O against it.
///
/// See the [module-level documentation](self) for the overall design.
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
    /// `sh -c` on Linux/macOS, `powershell -Command` on Windows.
    async fn exec_shell(&self, script: String, timeout: Option<u64>) -> anyhow::Result<ExecResult> {
        let (program, args) = match self.get_os() {
            "linux" | "macos" => ("sh".to_string(), vec!["-c".to_string(), script]),
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

/// A running environment. Wraps a [`Machine`] backend and hands out shared
/// [`RunEnvHandle`]s through [`RunEnv::get`]; the backend is booted on the
/// first `get` and shut down when the last handle is dropped.
///
/// See the [module-level documentation](self) for the overall design.
#[derive(Clone)]
pub struct RunEnv {
    machine: Arc<Mutex<dyn MachineDyn>>,
    state: Arc<Mutex<RunEnvState>>,
}

impl RunEnv {
    pub fn new<B: Machine>(machine: B) -> Self {
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

/// Shared handle to a booted [`RunEnv`]. Derefs to [`Console`] so callers can
/// invoke `exec`, `read`, `write`, etc. directly on the handle.
///
/// See the [module-level documentation](self) for the overall design.
pub struct RunEnvHandle {
    machine: Arc<Mutex<dyn MachineDyn>>,

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
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                rt.block_on(async move {
                    // Hold the state lock through shutdown so concurrent `get()` calls
                    // block until we transition back to `Idle`.
                    let mut s = state.lock().await;
                    machine.lock().await.shutdown().await;
                    *s = RunEnvState::Idle;
                });
            }
            let _ = tx.send(());
        });
        let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
    }
}
