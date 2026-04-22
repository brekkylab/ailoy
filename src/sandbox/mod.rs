//! Thin wrapper around the `microsandbox` crate, exposing an ailoy-internal
//! `Sandbox` type so the public API is not coupled to the underlying library.

#[cfg(feature = "sandbox")]
use std::time::Duration;
use std::{collections::HashMap, path::Path};

#[cfg(feature = "sandbox")]
use microsandbox::{
    Sandbox as MsbSandbox,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxStatus},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

//--------------------------------------------------------------------------------------------------
// Types shared between sandbox and no-sandbox builds
//--------------------------------------------------------------------------------------------------

fn fresh_sandbox_name() -> String {
    format!("ailoy-{}", Uuid::new_v4())
}

/// Configuration for creating a new sandbox.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SandboxConfig {
    /// Unique sandbox name — auto-generated at runtime, not serialized.
    #[serde(default = "fresh_sandbox_name", skip_serializing)]
    #[schemars(skip)]
    pub name: String,

    /// OCI container image. Default: `"python:3.12-slim"`.
    pub image: String,

    /// Number of virtual CPUs. Default: `2`.
    pub cpus: u8,

    /// Guest memory in MiB. Default: `2048`.
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
            memory_mib: 2048,
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
#[derive(Debug)]
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

//--------------------------------------------------------------------------------------------------
// Real Sandbox implementation (requires "sandbox" feature)
//--------------------------------------------------------------------------------------------------

/// `inner` is behind a `RwLock` so that:
/// - `exec`/`shell`/file ops hold a **read lock** → multiple tool calls run concurrently.
/// - `start`/`stop`/`shutdown` hold a **write lock** → exclusive, wait for all readers.
#[cfg(feature = "sandbox")]
pub struct Sandbox {
    inner: tokio::sync::RwLock<Option<MsbSandbox>>,
    name: String,
    persist: bool,
    default_timeout_secs: u64,
    max_output_chars: usize,
}

#[cfg(feature = "sandbox")]
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

#[cfg(feature = "sandbox")]
impl Drop for Sandbox {
    fn drop(&mut self) {
        if self.persist {
            return;
        }
        // `get_mut` bypasses the lock — safe here because Drop has `&mut self`,
        // guaranteeing no other owner exists.
        let inner = self.inner.get_mut().take();
        let name = self.name.clone();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                rt.block_on(async move {
                    match inner {
                        Some(inner) => {
                            // VM was running: kill then remove persisted state.
                            if let Ok(mut handle) = MsbSandbox::get(&name).await {
                                let _ = handle.kill().await;
                            }
                            let _ = inner.remove_persisted().await;
                        }
                        None => {
                            // VM was stopped: SandboxHandle::remove() deletes
                            // persisted state without needing to start the VM.
                            if let Ok(handle) = MsbSandbox::get(&name).await {
                                let _ = handle.remove().await;
                            }
                        }
                    }
                });
            }
            let _ = tx.send(());
        });
        let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
    }
}

#[cfg(feature = "sandbox")]
impl Sandbox {
    /// Create and start a new sandbox.  The VM is running on return.
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
            inner: tokio::sync::RwLock::new(Some(inner)),
            name,
            persist,
            default_timeout_secs,
            max_output_chars,
        };

        sandbox.shell(&format!("mkdir -p {workdir}")).await?;

        Ok(sandbox)
    }

    /// Return `true` if the VM is currently running.
    pub fn is_running(&self) -> bool {
        self.inner.try_read().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Start a stopped sandbox.  No-op if already running.
    pub async fn start(&self) -> anyhow::Result<()> {
        let mut guard = self.inner.write().await;
        if guard.is_some() {
            return Ok(());
        }
        let inner = MsbSandbox::start_detached(&self.name).await?;
        *guard = Some(inner);
        Ok(())
    }

    /// Stop the running sandbox without removing its persisted state.
    /// No-op if already stopped.  Waits for all ongoing exec/shell calls to
    /// finish before stopping (write lock blocks until all read locks release).
    pub async fn stop(&self) -> anyhow::Result<()> {
        let mut guard = self.inner.write().await;
        let Some(inner) = guard.take() else {
            return Ok(());
        };
        inner.stop_and_wait().await?;
        Ok(())
    }

    /// Stop the sandbox and, if not persisted, remove its on-disk state.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let mut guard = self.inner.write().await;
        let Some(inner) = guard.take() else {
            return Ok(());
        };
        inner.stop_and_wait().await?;
        if !self.persist {
            inner.remove_persisted().await?;
        }
        Ok(())
    }

    // ---- execution methods (read lock — concurrent) -------------------------

    pub async fn exec(&self, cmd: &str, args: &[&str]) -> anyhow::Result<ExecResult> {
        let timeout = Duration::from_secs(self.default_timeout_secs);
        let owned_args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        let result = inner
            .exec_with(cmd, |b: ExecOptionsBuilder| {
                b.args(owned_args.iter().map(|s| s.as_str()))
                    .timeout(timeout)
            })
            .await;
        self.handle_exec_result(result)
    }

    pub async fn shell(&self, script: &str) -> anyhow::Result<ExecResult> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        let result = inner.shell(script).await;
        self.handle_exec_result(result)
    }

    pub async fn shell_with_timeout(
        &self,
        script: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        let result = inner
            .exec_with("sh", |b: ExecOptionsBuilder| {
                b.args(["-c", script])
                    .timeout(Duration::from_secs(timeout_secs))
            })
            .await;
        self.handle_exec_result(result)
    }

    pub async fn write_file(&self, guest_path: &str, data: &[u8]) -> anyhow::Result<()> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        inner.fs().write(guest_path, data).await?;
        Ok(())
    }

    pub async fn read_file(&self, guest_path: &str) -> anyhow::Result<String> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        let s = inner.fs().read_to_string(guest_path).await?;
        Ok(s)
    }

    pub async fn read_file_bytes(&self, guest_path: &str) -> anyhow::Result<Vec<u8>> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        let bytes = inner.fs().read(guest_path).await?;
        Ok(bytes.to_vec())
    }

    pub async fn copy_from_host(&self, host: &Path, guest: &str) -> anyhow::Result<()> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        inner.fs().copy_from_host(host, guest).await?;
        Ok(())
    }

    pub async fn copy_to_host(&self, guest: &str, host: &Path) -> anyhow::Result<()> {
        let guard = self.inner.read().await;
        let inner = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("sandbox is not running"))?;
        inner.fs().copy_to_host(guest, host).await?;
        Ok(())
    }

    // ---- internal -----------------------------------------------------------

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
// Stub Sandbox (no "sandbox" feature — methods always return Err at runtime)
//--------------------------------------------------------------------------------------------------

/// Stub type that exists so `Option<Arc<Sandbox>>` compiles in all builds.
/// All methods return `Err` immediately; no `Sandbox` instance is ever
/// constructed without the `sandbox` feature.
#[cfg(not(feature = "sandbox"))]
pub struct Sandbox;

#[cfg(not(feature = "sandbox"))]
impl Sandbox {
    pub async fn start(&self) -> anyhow::Result<()> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn stop(&self) -> anyhow::Result<()> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub fn is_running(&self) -> bool {
        false
    }

    pub async fn exec(&self, _cmd: &str, _args: &[&str]) -> anyhow::Result<ExecResult> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn shell(&self, _script: &str) -> anyhow::Result<ExecResult> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn shell_with_timeout(
        &self,
        _script: &str,
        _timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn write_file(&self, _guest_path: &str, _data: &[u8]) -> anyhow::Result<()> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn read_file(&self, _guest_path: &str) -> anyhow::Result<String> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn read_file_bytes(&self, _guest_path: &str) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn copy_from_host(&self, _host: &Path, _guest: &str) -> anyhow::Result<()> {
        anyhow::bail!("sandbox feature not enabled")
    }

    pub async fn copy_to_host(&self, _guest: &str, _host: &Path) -> anyhow::Result<()> {
        anyhow::bail!("sandbox feature not enabled")
    }
}

//--------------------------------------------------------------------------------------------------
// Free functions (sandbox feature only)
//--------------------------------------------------------------------------------------------------

#[cfg(feature = "sandbox")]
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

#[cfg(feature = "sandbox")]
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

#[cfg(feature = "sandbox")]
fn truncate_output(s: String, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s
    } else {
        s.chars().take(max_chars).collect()
    }
}

//--------------------------------------------------------------------------------------------------
// Tests
//--------------------------------------------------------------------------------------------------

#[cfg(all(test, feature = "sandbox"))]
mod tests {
    use std::sync::Arc;

    use super::*;

    async fn make_sandbox() -> Arc<Sandbox> {
        Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("failed to create sandbox"),
        )
    }

    /// new() starts the VM and it is immediately running.
    #[tokio::test]
    async fn test_new_sandbox_is_running() {
        let sb = make_sandbox().await;
        assert!(sb.is_running(), "sandbox should be running after new()");
    }

    /// stop() halts the VM; start() brings it back up.
    #[tokio::test]
    async fn test_stop_and_start() {
        let sb = make_sandbox().await;

        sb.stop().await.expect("stop failed");
        assert!(!sb.is_running(), "sandbox should be stopped after stop()");

        sb.start().await.expect("start failed");
        assert!(sb.is_running(), "sandbox should be running after start()");
    }

    /// start() and stop() are idempotent.
    #[tokio::test]
    async fn test_start_stop_idempotent() {
        let sb = make_sandbox().await;

        // Double stop
        sb.stop().await.expect("first stop failed");
        sb.stop().await.expect("second stop should be a no-op");
        assert!(!sb.is_running());

        // Double start
        sb.start().await.expect("first start failed");
        sb.start().await.expect("second start should be a no-op");
        assert!(sb.is_running());
    }

    /// VM state (files written before stop) survives a stop/start cycle.
    #[tokio::test]
    async fn test_filesystem_persists_across_stop_start() {
        let sb = make_sandbox().await;

        sb.shell("echo hello > /workspace/test.txt")
            .await
            .expect("write failed");

        sb.stop().await.expect("stop failed");
        sb.start().await.expect("start failed");

        let content = sb
            .read_file("/workspace/test.txt")
            .await
            .expect("read failed");
        assert!(
            content.contains("hello"),
            "file should survive stop/start cycle, got: {content:?}"
        );
    }

    /// exec() fails with a clear error when the VM is stopped.
    #[tokio::test]
    async fn test_exec_while_stopped_returns_error() {
        let sb = make_sandbox().await;
        sb.stop().await.expect("stop failed");

        let result = sb.shell("echo test").await;
        assert!(
            result.is_err(),
            "shell() should fail when sandbox is stopped"
        );
        assert!(
            result.unwrap_err().to_string().contains("not running"),
            "error should mention 'not running'"
        );
    }
}
