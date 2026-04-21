#![cfg(feature = "sandbox-microvm")]
//! Thin wrapper around the `microsandbox` crate, exposing an ailoy-internal
//! `Sandbox` type so the public API is not coupled to the underlying library.

use std::{collections::HashMap, path::Path, time::Duration};

use microsandbox::{
    Sandbox as MsbSandbox,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxStatus},
};
use uuid::Uuid;

//--------------------------------------------------------------------------------------------------
// Types
//--------------------------------------------------------------------------------------------------

/// Configuration for creating a new sandbox.
pub struct SandboxConfig {
    /// Unique sandbox name (e.g. "ailoy-{uuid}").
    pub name: String,

    /// OCI container image. Default: `"python:3.12-slim"`.
    pub image: String,

    /// Number of virtual CPUs. Default: `2`.
    pub cpus: u8,

    /// Guest memory in MiB. Default: `512`.
    pub memory_mib: u32,

    /// Default working directory inside the sandbox. Default: `"/workspace"`.
    /// Created automatically after boot if it does not already exist.
    pub workdir: String,

    /// Environment variables passed to every command.
    pub env: HashMap<String, String>,

    /// When `true`, disable all network access. Default: `false`.
    pub disable_network: bool,

    /// Idle shutdown timeout in seconds. Default: `300`.
    pub idle_timeout_secs: u64,

    /// Per-exec timeout in seconds. Default: `60`.
    pub default_timeout_secs: u64,

    /// Maximum characters to keep from stdout/stderr. Default: `8000`.
    pub max_output_chars: usize,

    /// When `true`, the sandbox is not removed on drop and can be reused by
    /// name in a future session. Default: `false`.
    pub persist: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            name: format!("ailoy-{}", Uuid::new_v4()),
            image: "python:3.12-slim".to_string(),
            cpus: 2,
            memory_mib: 512,
            workdir: "/workspace".to_string(),
            env: HashMap::new(),
            disable_network: false,
            idle_timeout_secs: 300,
            default_timeout_secs: 60,
            max_output_chars: 8000,
            persist: false,
        }
    }
}

/// The result of running a command inside a sandbox.
pub struct ExecResult {
    /// Captured stdout (possibly truncated to `max_output_chars`).
    pub stdout: String,
    /// Captured stderr (possibly truncated to `max_output_chars`).
    pub stderr: String,
    /// Process exit code.
    pub exit_code: i32,
    /// Whether the command was killed due to a timeout.
    pub timed_out: bool,
}

/// A running sandbox wrapping a `microsandbox::Sandbox`.
pub struct Sandbox {
    /// The underlying microsandbox handle, wrapped in `Option` so it can be
    /// moved out in `Drop` for async cleanup.
    inner: Option<MsbSandbox>,
    name: String,
    persist: bool,
    default_timeout_secs: u64,
    max_output_chars: usize,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("name", &self.name)
            .field("persist", &self.persist)
            .field("default_timeout_secs", &self.default_timeout_secs)
            .field("max_output_chars", &self.max_output_chars)
            .finish_non_exhaustive()
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        if self.persist {
            return;
        }
        let Some(inner) = self.inner.take() else {
            return;
        };
        let name = self.name.clone();
        // Spawn a dedicated OS thread with its own tokio runtime for cleanup,
        // then block until it completes.
        //
        // Key design decisions:
        //
        // 1. Blocking wait (`rx.recv_timeout`) is required: fire-and-forget threads
        //    are killed when the test binary exits, leaving the sandbox in the DB as
        //    "Running" — which microsandbox later reconciles as "crashed".
        //
        // 2. `remove_persisted()` is used instead of `handle.remove()` because
        //    `handle.remove()` requires status == Stopped.  `remove_persisted()`
        //    directly deletes the DB record and files with no status check, and
        //    has no dependency on the AgentClient (unlike `stop_and_wait()`).
        //
        // 3. `handle.kill()` first (signal-based, no AgentClient) ensures the VM
        //    process is dead before we delete the record.  ProcessHandle::drop()
        //    on `inner` sends SIGTERM as a fallback if kill() was skipped/failed.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                rt.block_on(async move {
                    // Kill via signal (no AgentClient needed)
                    if let Ok(mut handle) = MsbSandbox::get(&name).await {
                        let _ = handle.kill().await;
                    }
                    // remove_persisted: deletes DB record + files unconditionally;
                    // ProcessHandle::drop() sends SIGTERM as a fallback.
                    let _ = inner.remove_persisted().await;
                });
            }
            let _ = tx.send(());
        });
        let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
    }
}

//--------------------------------------------------------------------------------------------------
// Methods
//--------------------------------------------------------------------------------------------------

impl Sandbox {
    /// Boot a new sandbox, or reuse an existing one when `config.persist` is `true`.
    ///
    /// - `persist: false` (default): always creates a fresh sandbox; on drop it is
    ///   stopped and its data removed automatically.
    /// - `persist: true`: if a sandbox with the same `name` already exists it is
    ///   reused (connected if running, restarted if stopped); otherwise a new one
    ///   is created. The sandbox is **not** removed on drop, so it survives across
    ///   process restarts.
    pub async fn new(config: SandboxConfig) -> anyhow::Result<Self> {
        let persist = config.persist;
        let default_timeout_secs = config.default_timeout_secs;
        let max_output_chars = config.max_output_chars;
        let workdir = config.workdir.clone();

        let inner = if persist {
            create_or_reuse(config).await?
        } else {
            create_fresh(config).await?
        };

        let name = inner.name().to_string();
        let sandbox = Self {
            inner: Some(inner),
            name,
            persist,
            default_timeout_secs,
            max_output_chars,
        };

        // Create workdir after sandbox has been booted
        sandbox.msb().shell(&format!("mkdir -p {workdir}")).await?;

        Ok(sandbox)
    }

    /// Run a command and wait for completion.
    pub async fn exec(&self, cmd: &str, args: &[&str]) -> anyhow::Result<ExecResult> {
        let timeout = Duration::from_secs(self.default_timeout_secs);
        let owned_args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let result = self
            .msb()
            .exec_with(cmd, |b: ExecOptionsBuilder| {
                b.args(owned_args.iter().map(|s| s.as_str()))
                    .timeout(timeout)
            })
            .await;

        self.handle_exec_result(result)
    }

    /// Run a shell command via the sandbox's default shell.
    pub async fn shell(&self, script: &str) -> anyhow::Result<ExecResult> {
        let result = self.msb().shell(script).await;
        self.handle_exec_result(result)
    }

    /// Run a shell command with a custom per-call timeout.
    pub async fn shell_with_timeout(
        &self,
        script: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        let result = self
            .msb()
            .exec_with("sh", |b: ExecOptionsBuilder| {
                b.args(["-c", script])
                    .timeout(Duration::from_secs(timeout_secs))
            })
            .await;
        self.handle_exec_result(result)
    }

    /// Write raw bytes to a file inside the sandbox.
    pub async fn write_file(&self, guest_path: &str, data: &[u8]) -> anyhow::Result<()> {
        self.msb().fs().write(guest_path, data).await?;
        Ok(())
    }

    /// Read a file from the sandbox as a UTF-8 string.
    pub async fn read_file(&self, guest_path: &str) -> anyhow::Result<String> {
        let s = self.msb().fs().read_to_string(guest_path).await?;
        Ok(s)
    }

    /// Read a file from the sandbox as raw bytes.
    pub async fn read_file_bytes(&self, guest_path: &str) -> anyhow::Result<Vec<u8>> {
        let bytes = self.msb().fs().read(guest_path).await?;
        Ok(bytes.to_vec())
    }

    /// Copy a file from the host filesystem into the sandbox.
    pub async fn copy_from_host(&self, host: &Path, guest: &str) -> anyhow::Result<()> {
        self.msb().fs().copy_from_host(host, guest).await?;
        Ok(())
    }

    /// Copy a file from the sandbox to the host filesystem.
    pub async fn copy_to_host(&self, guest: &str, host: &Path) -> anyhow::Result<()> {
        self.msb().fs().copy_to_host(guest, host).await?;
        Ok(())
    }

    /// Explicitly stop the sandbox.
    ///
    /// For non-persistent sandboxes this also removes all persisted data — the
    /// same cleanup that would happen on drop.  For persistent sandboxes only
    /// the VM is stopped; data is kept for the next `Sandbox::new` call.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let Some(inner) = self.inner.take() else {
            return Ok(());
        };
        // stop_and_wait() uses the AgentClient and works here because shutdown()
        // is called on the same runtime that created the sandbox.
        inner.stop_and_wait().await?;
        if !self.persist {
            // remove_persisted: delete DB record + files without status check.
            inner.remove_persisted().await?;
        }
        Ok(())
    }

    //----------------------------------------------------------------------------------------------
    // Helpers
    //----------------------------------------------------------------------------------------------

    fn msb(&self) -> &MsbSandbox {
        self.inner
            .as_ref()
            .expect("sandbox has already been shut down")
    }

    fn handle_exec_result(
        &self,
        result: Result<microsandbox::ExecOutput, microsandbox::MicrosandboxError>,
    ) -> anyhow::Result<ExecResult> {
        use microsandbox::MicrosandboxError;

        match result {
            Ok(output) => {
                let stdout =
                    truncate_output(output.stdout().unwrap_or_default(), self.max_output_chars);
                let stderr =
                    truncate_output(output.stderr().unwrap_or_default(), self.max_output_chars);
                Ok(ExecResult {
                    stdout,
                    stderr,
                    exit_code: output.status().code,
                    timed_out: false,
                })
            }
            Err(MicrosandboxError::ExecTimeout(_)) => Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
            Err(e) => Err(e.into()),
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Free functions
//--------------------------------------------------------------------------------------------------

/// Create a brand-new ephemeral sandbox (persist = false).
async fn create_fresh(config: SandboxConfig) -> anyhow::Result<MsbSandbox> {
    let mut builder = MsbSandbox::builder(&config.name)
        .image(config.image.as_str())
        .cpus(config.cpus)
        .memory(config.memory_mib)
        .idle_timeout(config.idle_timeout_secs)
        .pull_policy(PullPolicy::IfMissing);

    for (k, v) in &config.env {
        builder = builder.env(k.as_str(), v.as_str());
    }
    if config.disable_network {
        builder = builder.disable_network();
    }

    Ok(builder.create().await?)
}

/// Reuse an existing sandbox by name or create it if not found (persist = true).
///
/// Running sandbox → connect (no lifecycle takeover).
/// Stopped sandbox → restart in detached mode.
/// Unknown name    → create new in detached mode.
async fn create_or_reuse(config: SandboxConfig) -> anyhow::Result<MsbSandbox> {
    match MsbSandbox::get(&config.name).await {
        Ok(handle) => {
            let sb = match handle.status() {
                SandboxStatus::Running | SandboxStatus::Draining => handle.connect().await?,
                _ => MsbSandbox::start_detached(&config.name).await?,
            };
            Ok(sb)
        }
        Err(_) => {
            // Sandbox not found — create a new one in detached mode so it
            // outlives the current Sandbox wrapper when dropped.
            let mut builder = MsbSandbox::builder(&config.name)
                .image(config.image.as_str())
                .cpus(config.cpus)
                .memory(config.memory_mib)
                .idle_timeout(config.idle_timeout_secs)
                .pull_policy(PullPolicy::IfMissing);

            for (k, v) in &config.env {
                builder = builder.env(k.as_str(), v.as_str());
            }
            if config.disable_network {
                builder = builder.disable_network();
            }

            Ok(builder.create_detached().await?)
        }
    }
}

/// Truncate a `String` to at most `max_chars` Unicode scalar values.
fn truncate_output(s: String, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s
    } else {
        s.chars().take(max_chars).collect()
    }
}
