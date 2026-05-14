//! Thin wrapper around the `microsandbox` crate, exposing an ailoy-internal
//! `Sandbox` type so the public API is not coupled to the underlying library.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use microsandbox::{
    ExecOutput, MicrosandboxError, Sandbox as MsbSandbox,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxBuilder, SandboxStatus},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{ExecResult, RunEnv};
use crate::util::truncate::middle_truncate;

fn fresh_sandbox_name() -> String {
    format!("ailoy-{}", Uuid::new_v4())
}

/// A volume mount attached to a sandbox at creation time.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VolumeMount {
    /// Bind-mount a host directory into the guest.
    Bind {
        /// Absolute or relative host path.
        host: PathBuf,
        /// Absolute guest path (e.g. `/data`).
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
    },
    /// Mount a microsandbox named volume (`~/.microsandbox/volumes/<name>/`).
    /// The volume persists across sandbox restarts and can be shared between sandboxes.
    Named {
        /// Name of the pre-existing microsandbox volume.
        name: String,
        /// Absolute guest path.
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
    },
    /// Memory-backed temporary filesystem. Disappears when the sandbox stops.
    Tmpfs {
        /// Absolute guest path.
        guest: String,
        /// Size limit in MiB. `None` means no limit.
        size_mib: Option<u32>,
    },
}

/// Configuration for creating a new sandbox.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SandboxConfig {
    /// Unique sandbox name.
    ///
    /// - `Some("my-sandbox")` — use this exact name.  Serialized/deserialized
    ///   correctly, so a named config round-trips and reconnects to the same VM.
    /// - `None` (default) — auto-generate a UUID-based name at [`Sandbox::new`]
    ///   time.  Omitted from serialized output; deserialized back as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// OCI container image. Default: `"ubuntu:latest"`.
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

    /// Per-exec timeout in seconds. Default: `60`.
    pub default_timeout_secs: u64,

    /// Maximum characters to keep from stdout/stderr. Default: `30_000`.
    pub max_output_chars: usize,

    /// When `true`, the sandbox is not removed on drop and can be reused by
    /// name in a future session. Default: `false`.
    pub persist: bool,

    /// Volume mounts attached at sandbox creation time.
    #[serde(default)]
    pub volumes: Vec<VolumeMount>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            name: None,
            image: "ubuntu:latest".to_string(),
            cpus: 2,
            memory_mib: 2048,
            workdir: "/workspace".to_string(),
            env: HashMap::new(),
            disable_network: false,
            default_timeout_secs: 60,
            max_output_chars: 30_000,
            persist: false,
            volumes: Vec::new(),
        }
    }
}

/// `inner` is behind a `Mutex` so that all sandbox operations (exec, shell, file I/O,
/// start/stop) are serialized. Tool calls may be spawned in parallel but queue up here,
/// preventing concurrent commands from racing on the same VM filesystem.
pub struct Sandbox {
    inner: tokio::sync::Mutex<MsbSandbox>,
    name: String,
    workdir: String,
    persist: bool,
    default_timeout_secs: u64,
    max_output_chars: usize,
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("name", &self.name)
            .field("workdir", &self.workdir)
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
        let name = self.name.clone();
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        std::thread::spawn(move || {
            if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                rt.block_on(async move {
                    if let Ok(handle) = MsbSandbox::get(&name).await {
                        match handle.status() {
                            SandboxStatus::Running | SandboxStatus::Draining => {
                                if let Ok(connected) = handle.connect().await {
                                    let _ = connected.stop_and_wait().await;
                                    let _ = connected.remove_persisted().await;
                                }
                            }
                            _ => {
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

#[async_trait::async_trait]
impl RunEnv for Sandbox {
    fn get_os(&self) -> &str {
        "linux"
    }

    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        let timeout_secs = timeout.unwrap_or(self.default_timeout_secs);
        let max_output_chars = self.max_output_chars;
        let guard = self.ensure_running().await?;
        let result = {
            let raw = guard
                .exec_with(&program, |b: ExecOptionsBuilder| {
                    b.args(args.iter().map(|s| s.as_str()))
                        .timeout(Duration::from_secs(timeout_secs))
                })
                .await;
            Self::handle_exec_result_static(raw, max_output_chars)
        };
        let _ = guard.stop_and_wait().await;
        result
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let guest_path = path.to_string_lossy().into_owned();
        let guard = self.ensure_running().await?;
        let result = guard.fs().read(&guest_path).await.map(|b| b.to_vec());
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        let guest_path = path.to_string_lossy().into_owned();
        let guard = self.ensure_running().await?;
        let result = guard.fs().write(&guest_path, content).await;
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }
}

impl Sandbox {
    /// Create a sandbox and register it.  The VM is started once to set up the workdir,
    /// then immediately stopped.  Subsequent operations lazy-start the VM as needed.
    pub async fn new(config: SandboxConfig) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        if !microsandbox::setup::is_installed() {
            log::warn!(
                "microsandbox runtime not found — downloading to ~/.microsandbox, \
                 this may take a moment"
            );
            microsandbox::setup::install()
                .await
                .context("microsandbox install")?;
            log::info!("microsandbox runtime installed");
        }

        let name = config.name.clone().unwrap_or_else(fresh_sandbox_name);
        validate_sandbox_name(&name)?;
        let workdir = config.workdir.clone();
        let persist = config.persist;
        let default_timeout_secs = config.default_timeout_secs;
        let max_output_chars = config.max_output_chars;

        let inner = if persist {
            if MsbSandbox::get(&name).await.is_ok() {
                MsbSandbox::start_detached(&name)
                    .await
                    .context("sandbox start")?
            } else {
                create_registered(&name, &config)
                    .await
                    .context("sandbox create-registered")?
            }
        } else {
            create_registered(&name, &config)
                .await
                .context("sandbox create-registered")?
        };
        // Ensure workdir exists, then stop — VM will be started again on first op.
        let _ = inner.shell(&format!("mkdir -p {workdir}")).await;
        inner.stop_and_wait().await?;

        Ok(Self {
            inner: tokio::sync::Mutex::new(inner),
            name,
            workdir,
            persist,
            default_timeout_secs,
            max_output_chars,
        })
    }

    /// Return `true` if the VM is currently running according to microsandbox.
    pub async fn is_running(&self) -> bool {
        vm_is_running(&self.name).await
    }

    /// Start a stopped sandbox.  No-op if already running.
    pub async fn start(&self) -> anyhow::Result<()> {
        let _guard = self.ensure_running().await?;
        Ok(())
    }

    /// Stop the running sandbox without removing its persisted state.
    /// No-op if already stopped.
    pub async fn stop(&self) -> anyhow::Result<()> {
        let guard = self.inner.lock().await;
        if vm_is_running(&self.name).await {
            guard.stop_and_wait().await?;
        }
        Ok(())
    }

    /// Stop the sandbox and, if not persisted, remove its on-disk state.
    pub async fn shutdown(&self) -> anyhow::Result<()> {
        let guard = self.inner.lock().await;
        if vm_is_running(&self.name).await {
            guard.stop_and_wait().await?;
        }
        if !self.persist {
            if let Ok(handle) = MsbSandbox::get(&self.name).await {
                handle.remove().await?;
            }
        }
        Ok(())
    }

    pub async fn shell(&self, script: &str) -> anyhow::Result<ExecResult> {
        let script = script.to_string();
        let max_output_chars = self.max_output_chars;
        let guard = self.ensure_running().await?;
        let result = {
            let raw = guard.shell(&script).await;
            Self::handle_exec_result_static(raw, max_output_chars)
        };
        let _ = guard.stop_and_wait().await;
        result
    }

    pub async fn shell_with_timeout(
        &self,
        script: &str,
        timeout_secs: u64,
    ) -> anyhow::Result<ExecResult> {
        let script = script.to_string();
        let max_output_chars = self.max_output_chars;
        let guard = self.ensure_running().await?;
        let result = {
            let raw = guard
                .exec_with("sh", |b: ExecOptionsBuilder| {
                    b.args(["-c", &script])
                        .timeout(Duration::from_secs(timeout_secs))
                })
                .await;
            Self::handle_exec_result_static(raw, max_output_chars)
        };
        let _ = guard.stop_and_wait().await;
        result
    }

    pub async fn write_file(&self, guest_path: &str, data: &[u8]) -> anyhow::Result<()> {
        let guest_path = guest_path.to_string();
        let data = data.to_vec();
        let guard = self.ensure_running().await?;
        let result = guard.fs().write(&guest_path, &data).await;
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    pub async fn read_file(&self, guest_path: &str) -> anyhow::Result<String> {
        let guest_path = guest_path.to_string();
        let guard = self.ensure_running().await?;
        let result = guard.fs().read_to_string(&guest_path).await;
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    pub async fn read_file_bytes(&self, guest_path: &str) -> anyhow::Result<Vec<u8>> {
        let guest_path = guest_path.to_string();
        let guard = self.ensure_running().await?;
        let result = guard.fs().read(&guest_path).await.map(|b| b.to_vec());
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    pub async fn copy_from_host(&self, host: &Path, guest: &str) -> anyhow::Result<()> {
        let host = host.to_path_buf();
        let guest = guest.to_string();
        let guard = self.ensure_running().await?;
        let result = guard.fs().copy_from_host(&host, &guest).await;
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    pub async fn copy_to_host(&self, guest: &str, host: &Path) -> anyhow::Result<()> {
        let guest = guest.to_string();
        let host = host.to_path_buf();
        let guard = self.ensure_running().await?;
        let result = guard.fs().copy_to_host(&guest, &host).await;
        let _ = guard.stop_and_wait().await;
        Ok(result?)
    }

    /// Acquire the mutex and start the VM for the next operation.
    ///
    /// We always stop the VM after each operation, so `start_detached()` should
    /// succeed unconditionally here. If the previous `stop_and_wait()` failed
    /// silently and the VM is still alive, microsandbox returns
    /// `SandboxStillRunning` — we force-stop and retry once.
    ///
    /// Avoids relying on `vm_is_running()` (a DB + PID liveness check) which can
    /// return a false positive when a dead VM's PID is quickly reused by the OS,
    /// causing the stale inner handle to be used and all writes/execs to fail.
    async fn ensure_running(&self) -> anyhow::Result<tokio::sync::MutexGuard<'_, MsbSandbox>> {
        let mut guard = self.inner.lock().await;

        let started = match MsbSandbox::start_detached(&self.name).await {
            Ok(s) => s,
            Err(MicrosandboxError::SandboxStillRunning(_)) => {
                // Previous stop_and_wait() failed silently — force-stop and retry.
                let _ = guard.stop_and_wait().await;
                MsbSandbox::start_detached(&self.name)
                    .await
                    .map_err(|e| anyhow::anyhow!("sandbox start after force-stop: {e}"))?
            }
            Err(e) => return Err(anyhow::anyhow!("sandbox start: {e}")),
        };

        let _ = started.shell(&format!("mkdir -p {}", self.workdir)).await;
        *guard = started;
        Ok(guard)
    }

    fn handle_exec_result_static(
        result: Result<ExecOutput, MicrosandboxError>,
        max_output_chars: usize,
    ) -> anyhow::Result<ExecResult> {
        match result {
            Ok(output) => {
                let stdout = middle_truncate(output.stdout().unwrap_or_default(), max_output_chars);
                let stderr = middle_truncate(output.stderr().unwrap_or_default(), max_output_chars);
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

/// Remove a persist=true sandbox by name without holding a [`Sandbox`] instance.
///
/// Idempotent: if the named sandbox does not exist, returns `Ok(())`.
pub async fn remove_persisted(name: &str) -> anyhow::Result<()> {
    match MsbSandbox::get(name).await {
        Err(_) => Ok(()),
        Ok(handle) => {
            match handle.status() {
                SandboxStatus::Running | SandboxStatus::Draining => {
                    let connected = handle.connect().await?;
                    connected.stop_and_wait().await?;
                    connected.remove_persisted().await?;
                }
                _ => {
                    handle.remove().await?;
                }
            }
            Ok(())
        }
    }
}

/// Validate that `name` won't produce a socket path that exceeds the OS limit.
fn validate_sandbox_name(name: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    const SUN_PATH_MAX: usize = 104;
    #[cfg(not(target_os = "macos"))]
    const SUN_PATH_MAX: usize = 108;

    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory"))?;
    let home_str = home
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("home directory path is not valid UTF-8"))?;

    let socket_path_len = home_str.len()
        + "/.microsandbox/sandboxes/".len()
        + name.len()
        + "/runtime/agent.sock".len()
        + 1; // null terminator

    if socket_path_len > SUN_PATH_MAX {
        let max_name = SUN_PATH_MAX.saturating_sub(home_str.len() + 25 + 19 + 1);
        anyhow::bail!(
            "sandbox name '{}' is too long ({} chars); the resulting socket path \
             would be {} bytes but the OS limit (SUN_PATH_MAX) is {}. \
             With home directory '{}', the maximum sandbox name length is {} chars.",
            name,
            name.len(),
            socket_path_len,
            SUN_PATH_MAX,
            home_str,
            max_name,
        );
    }
    Ok(())
}

async fn vm_is_running(name: &str) -> bool {
    MsbSandbox::get(name)
        .await
        .map(|h| matches!(h.status(), SandboxStatus::Running | SandboxStatus::Draining))
        .unwrap_or(false)
}

/// Create and start a new sandbox, returning the running handle.
async fn create_registered(name: &str, config: &SandboxConfig) -> anyhow::Result<MsbSandbox> {
    let mut builder = MsbSandbox::builder(name)
        .image(config.image.as_str())
        .cpus(config.cpus)
        .memory(config.memory_mib)
        .pull_policy(PullPolicy::IfMissing);

    for (k, v) in &config.env {
        builder = builder.env(k.as_str(), v.as_str());
    }
    if config.disable_network {
        builder = builder.disable_network();
    }
    for mount in &config.volumes {
        builder = apply_volume_mount(builder, mount);
    }
    let sb = builder.create_detached().await?;
    Ok(sb)
}

fn apply_volume_mount(builder: SandboxBuilder, mount: &VolumeMount) -> SandboxBuilder {
    match mount {
        VolumeMount::Bind {
            host,
            guest,
            readonly,
        } => {
            let host = host.clone();
            let ro = *readonly;
            builder.volume(guest, move |m| {
                let m = m.bind(host);
                if ro { m.readonly() } else { m }
            })
        }
        VolumeMount::Named {
            name,
            guest,
            readonly,
        } => {
            let name = name.clone();
            let ro = *readonly;
            builder.volume(guest, move |m| {
                let m = m.named(name);
                if ro { m.readonly() } else { m }
            })
        }
        VolumeMount::Tmpfs { guest, size_mib } => {
            let size = *size_mib;
            builder.volume(guest, move |m| {
                let m = m.tmpfs();
                if let Some(s) = size { m.size(s) } else { m }
            })
        }
    }
}

#[cfg(test)]
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

    /// `SandboxConfig::name` round-trips through JSON: Some is preserved, None is omitted.
    #[test]
    fn test_sandbox_config_name_serde_roundtrip() {
        // Some("my-sandbox") must survive serialize → deserialize.
        let cfg_named = SandboxConfig {
            name: Some("my-sandbox".to_string()),
            persist: true,
            ..SandboxConfig::default()
        };
        let json = serde_json::to_string(&cfg_named).expect("serialize failed");
        assert!(
            json.contains("\"name\""),
            "expected 'name' key in JSON when name is Some, got: {json}"
        );
        let back: SandboxConfig = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(
            back.name,
            Some("my-sandbox".to_string()),
            "name must survive round-trip when Some"
        );

        // None must be omitted from serialized output and stay None on deserialize.
        let cfg_anon = SandboxConfig::default();
        let json_anon = serde_json::to_string(&cfg_anon).expect("serialize failed");
        assert!(
            !json_anon.contains("\"name\""),
            "expected 'name' key absent from JSON when name is None, got: {json_anon}"
        );
        let back_anon: SandboxConfig =
            serde_json::from_str(&json_anon).expect("deserialize failed");
        assert!(
            back_anon.name.is_none(),
            "name must stay None after round-trip when originally None"
        );
    }

    /// Names that would exceed the socket path limit are rejected synchronously.
    #[test]
    fn test_name_too_long_is_rejected() {
        let long_name = "a".repeat(100);
        let result = validate_sandbox_name(&long_name);
        assert!(result.is_err(), "expected Err for a 100-char name");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("too long") && msg.contains("SUN_PATH_MAX"),
            "error message should explain the limit, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_stop_and_start() {
        let sb = make_sandbox().await;

        sb.stop().await.expect("stop failed");
        assert!(
            !sb.is_running().await,
            "sandbox should be stopped after stop()"
        );

        sb.start().await.expect("start failed");
        assert!(
            sb.is_running().await,
            "sandbox should be running after start()"
        );
    }

    #[tokio::test]
    async fn test_start_stop_idempotent() {
        let sb = make_sandbox().await;

        sb.stop().await.expect("first stop failed");
        sb.stop().await.expect("second stop should be a no-op");
        assert!(!sb.is_running().await);

        sb.start().await.expect("first start failed");
        sb.start().await.expect("second start should be a no-op");
        assert!(sb.is_running().await);
    }

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

    #[tokio::test]
    async fn test_restart_latency() {
        let sb = make_sandbox().await;

        sb.stop().await.expect("stop failed");

        let t = std::time::Instant::now();
        sb.start().await.expect("start failed");
        let restart_ms = t.elapsed().as_millis();

        assert!(
            restart_ms < 3000,
            "restart should be well under 3s, got {restart_ms}ms"
        );
    }

    #[tokio::test]
    async fn test_shared_arc_serial_ops() {
        let sb = make_sandbox().await;

        let sb1 = sb.clone();
        let sb2 = sb.clone();
        let t1 = tokio::spawn(async move { sb1.shell("echo a").await });
        let t2 = tokio::spawn(async move { sb2.shell("echo b").await });

        let (r1, r2) = tokio::join!(t1, t2);
        let r1 = r1.expect("task1 panic").expect("task1 shell error");
        let r2 = r2.expect("task2 panic").expect("task2 shell error");

        assert_eq!(r1.exit_code, 0, "task1 exit_code: {:?}", r1.stderr);
        assert_eq!(r2.exit_code, 0, "task2 exit_code: {:?}", r2.stderr);
    }

    #[tokio::test]
    async fn test_sequential_ops_on_shared_arc() {
        let sb = make_sandbox().await;

        sb.write_file("/workspace/marker.txt", b"sequential")
            .await
            .expect("write_file failed");

        for i in 0..10 {
            let content = sb
                .read_file("/workspace/marker.txt")
                .await
                .unwrap_or_else(|e| panic!("read_file failed on iteration {i}: {e}"));
            assert!(
                content.contains("sequential"),
                "iteration {i}: expected 'sequential' in content, got: {content:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_remove_persisted_idempotent() {
        let result = remove_persisted("ailoy-nonexistent-sandbox-xyz-12345").await;
        assert!(
            result.is_ok(),
            "remove_persisted on unknown name should return Ok, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_remove_persisted_cleans_up() {
        let name = format!("at-rp-{}", &uuid::Uuid::new_v4().to_string()[..8]);

        {
            let sb = Sandbox::new(SandboxConfig {
                persist: true,
                name: Some(name.clone()),
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox");
            sb.write_file("/workspace/marker.txt", b"persist_marker")
                .await
                .expect("write_file failed");
        }

        remove_persisted(&name)
            .await
            .expect("remove_persisted failed");

        let sb2 = Sandbox::new(SandboxConfig {
            persist: true,
            name: Some(name.clone()),
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to recreate sandbox");

        let result = sb2.read_file("/workspace/marker.txt").await;
        assert!(
            result.is_err(),
            "fresh VM should not contain the marker file from the previous run"
        );

        remove_persisted(&name).await.ok();
    }

    #[tokio::test]
    async fn test_bind_mount_host_to_guest() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");
        let host_file = host_dir.path().join("hello.txt");
        std::fs::write(&host_file, "bind_mount_works").expect("failed to write host file");

        let sb = Arc::new(
            Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Bind {
                    host: host_dir.path().to_path_buf(),
                    guest: "/mnt/host".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );

        let result = sb
            .shell("cat /mnt/host/hello.txt")
            .await
            .expect("shell failed");
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(
            result.stdout.contains("bind_mount_works"),
            "stdout: {:?}",
            result.stdout
        );
    }

    #[tokio::test]
    async fn test_bind_mount_readonly_rejects_guest_write() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");

        let sb = Arc::new(
            Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Bind {
                    host: host_dir.path().to_path_buf(),
                    guest: "/mnt/ro".to_string(),
                    readonly: true,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );

        let result = sb
            .shell("echo should_fail > /mnt/ro/file.txt")
            .await
            .expect("shell failed");
        assert_ne!(result.exit_code, 0, "write to read-only mount should fail");
    }

    #[tokio::test]
    async fn test_bind_mount_guest_write_visible_on_host() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");

        let sb = Arc::new(
            Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Bind {
                    host: host_dir.path().to_path_buf(),
                    guest: "/mnt/shared".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );

        let result = sb
            .shell("echo guest_wrote > /mnt/shared/out.txt")
            .await
            .expect("shell failed");
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);

        let host_content =
            std::fs::read_to_string(host_dir.path().join("out.txt")).expect("host file not found");
        assert!(
            host_content.contains("guest_wrote"),
            "got: {host_content:?}"
        );
    }

    #[tokio::test]
    async fn test_tmpfs_mount_is_writable() {
        let sb = Arc::new(
            Sandbox::new(SandboxConfig {
                volumes: vec![VolumeMount::Tmpfs {
                    guest: "/mnt/tmp".to_string(),
                    size_mib: Some(64),
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox"),
        );

        let result = sb
            .shell("echo tmpfs_ok > /mnt/tmp/test.txt && cat /mnt/tmp/test.txt")
            .await
            .expect("shell failed");
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(
            result.stdout.contains("tmpfs_ok"),
            "stdout: {:?}",
            result.stdout
        );
    }

    #[tokio::test]
    async fn test_named_volume_persists_across_sandboxes() {
        use microsandbox::Volume;

        let vol_name = format!("ailoy-test-vol-{}", uuid::Uuid::new_v4());

        Volume::builder(&vol_name)
            .create()
            .await
            .expect("failed to create named volume");

        let write_result = async {
            let sb = Arc::new(
                Sandbox::new(SandboxConfig {
                    volumes: vec![VolumeMount::Named {
                        name: vol_name.clone(),
                        guest: "/mnt/vol".to_string(),
                        readonly: false,
                    }],
                    ..SandboxConfig::default()
                })
                .await
                .expect("failed to create first sandbox"),
            );
            sb.shell("echo named_vol_works > /mnt/vol/data.txt")
                .await
                .expect("write failed")
        }
        .await;

        assert_eq!(
            write_result.exit_code, 0,
            "write failed, stderr: {}",
            write_result.stderr
        );

        let read_result = async {
            let sb = Arc::new(
                Sandbox::new(SandboxConfig {
                    volumes: vec![VolumeMount::Named {
                        name: vol_name.clone(),
                        guest: "/mnt/vol".to_string(),
                        readonly: false,
                    }],
                    ..SandboxConfig::default()
                })
                .await
                .expect("failed to create second sandbox"),
            );
            sb.shell("cat /mnt/vol/data.txt")
                .await
                .expect("read failed")
        }
        .await;

        if let Ok(handle) = Volume::get(&vol_name).await {
            let _ = handle.remove().await;
        }

        assert_eq!(
            read_result.exit_code, 0,
            "read failed, stderr: {}",
            read_result.stderr
        );
        assert!(
            read_result.stdout.contains("named_vol_works"),
            "stdout: {:?}",
            read_result.stdout
        );
    }
}
