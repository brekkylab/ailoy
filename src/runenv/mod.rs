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
//! machine is only started on demand: callers obtain a [`RunEnvHandle`] via
//! [`RunEnv::get`], which lazily starts the container on the first call and
//! reuses the same started handle for subsequent callers. When the last
//! outstanding `RunEnvHandle` is dropped, the container is stopped
//! automatically and the `RunEnv` returns to the idle state, ready to be
//! started again on the next `get()`.
//!
//! All operations against the environment go through the handle, which
//! derefs to [`Console`].
//!
//! # Extending with a new runenv
//!
//! To add a new kind of running environment, implement two traits:
//!
//! - [`Machine`]: describes how to start and stop the underlying
//!   machine. `start` returns a handle type that implements [`Console`].
//! - [`Console`]: the started handle. At minimum, implement
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
//! struct MyConsole { /* started session, e.g. an SSH channel */ }
//!
//! #[async_trait]
//! impl Machine for MyMachine {
//!     type Handle = MyConsole;
//!
//!     async fn start(&mut self) -> anyhow::Result<Self::Handle> {
//!         // open connection, spawn shell, etc.
//!         Ok(MyConsole { /* ... */ })
//!     }
//!
//!     async fn stop(&mut self) {
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

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};

mod file_entry;
mod local;
#[cfg(feature = "sandbox")]
mod sandbox;

pub use file_entry::*;
pub use local::*;
#[cfg(feature = "sandbox")]
pub use sandbox::*;

/// Backend that knows how to start and stop a running environment,
/// yielding a [`Console`] handle for the started instance.
///
/// See the [module-level documentation](self) for the overall design.
#[async_trait]
pub trait Machine: Send + Sync + 'static {
    type Console: Console;

    /// Whether the machine is currently running.
    fn is_running(&self) -> bool;

    /// Start the machine and return its console.
    /// Returns the existing console if already running.
    async fn start<'a>(&'a mut self) -> anyhow::Result<&'a Self::Console>;

    /// Stop the machine and release its resources.
    /// No-op if already stopped.
    ///
    /// Note that this is not essential op.
    /// Using [`start`] alone is enough to use the machine.
    /// However, calling `stop` when idle helps keep resource use low.
    /// When (or whether) to call it is up to the caller.
    async fn stop(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

/// Object-safe erased view of `Machine`. The blanket impl below adapts any
/// concrete `Machine` by erasing the handle to `&dyn Console`.
///
/// Use this when you need to store a heterogeneous machine behind a single
/// type, e.g. `Arc<Mutex<dyn MachineDyn>>` shared across agents.
#[async_trait]
pub trait MachineDyn: Send + Sync + 'static {
    fn is_running(&self) -> bool;

    async fn start<'a>(&'a mut self) -> anyhow::Result<&'a dyn Console>;

    async fn stop(&mut self) -> anyhow::Result<()>;
}

#[async_trait]
impl<B: Machine> MachineDyn for B {
    fn is_running(&self) -> bool {
        Machine::is_running(self)
    }

    async fn start<'a>(&'a mut self) -> anyhow::Result<&'a dyn Console> {
        // UFCS so this doesn't recurse into the MachineDyn impl we're in.
        // `&B::Console` unsizes to `&dyn Console` via coercion.
        Ok(Machine::start(self).await?)
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        Machine::stop(self).await
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

/// Handle to a started running environment, exposing command execution and
/// file I/O against it.
///
/// See the [module-level documentation](self) for the overall design.
#[async_trait]
pub trait Console: Send + Sync {
    fn get_os(&self) -> &str;

    /// Run `program` with `args`. `timeout` is in seconds; when elapsed the
    /// resulting [`ExecResult`] has `timed_out = true`.
    ///
    /// Takes `&self` so a single started handle can be shared by multiple
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
    /// `sh -c` on Linux/macOS, `powershell -EncodedCommand` on Windows.
    async fn exec_shell(&self, script: String, timeout: Option<u64>) -> anyhow::Result<ExecResult> {
        let (program, args) = match self.get_os() {
            "linux" | "macos" => ("sh".to_string(), vec!["-c".to_string(), script]),
            "windows" => {
                // On Windows the script is transported as UTF-16LE base64 to bypass
                // CMD/PowerShell argv quoting; quotes, newlines, and metacharacters in
                // `script` pass through verbatim. Output encoding (BOM, code page) is
                // unaffected and must still be handled by callers if relevant.
                let utf16le: Vec<u8> = script
                    .encode_utf16()
                    .flat_map(|u| u.to_le_bytes())
                    .collect();
                let b64 = STANDARD.encode(&utf16le);
                (
                    "powershell".to_string(),
                    vec!["-NoLogo".to_string(), "-EncodedCommand".to_string(), b64],
                )
            }
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
