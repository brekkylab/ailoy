//! microsandbox-backed `Container` for runenv.
//!
//! Adapted from [`crate::runenv::sandbox`] to the v2 Container/Console split.
//! Heavyweight, fallible setup happens in `Sandbox::new`; `Container::boot`
//! restarts the VM, and `Container::shutdown` stops/removes it.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use async_trait::async_trait;
use microsandbox::{
    ExecOutput, MicrosandboxError, Sandbox as MsbSandbox,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxBuilder, SandboxStatus},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Console, Container, ExecResult};
use crate::util::truncate::middle_truncate;

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

fn fresh_sandbox_name() -> String {
    format!("ailoy-{}", Uuid::new_v4())
}

/// `Container` implementation backed by a microsandbox VM.
///
/// The VM is created at construction time and left stopped; `boot()` starts
/// it for the duration of the returned [`SandboxConsole`], and `shutdown()`
/// stops (and, unless `config.persist`, removes) it.
pub struct Sandbox {
    config: SandboxConfig,
    name: String,
}

impl Sandbox {
    /// Install the microsandbox runtime if needed, create or attach to the
    /// named VM, ensure `workdir` exists, then stop the VM.
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

        let inner = if config.persist && MsbSandbox::get(&name).await.is_ok() {
            MsbSandbox::start_detached(&name)
                .await
                .context("sandbox start")?
        } else {
            create_registered(&name, &config)
                .await
                .context("sandbox create-registered")?
        };
        let _ = inner.shell(&format!("mkdir -p {}", config.workdir)).await;
        inner.stop_and_wait().await?;

        Ok(Self { config, name })
    }
}

impl std::fmt::Debug for Sandbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sandbox")
            .field("name", &self.name)
            .field("workdir", &self.config.workdir)
            .field("persist", &self.config.persist)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Container for Sandbox {
    type Handle = SandboxConsole;

    async fn boot(&mut self) -> anyhow::Result<SandboxConsole> {
        let inner = start_with_retry(&self.name).await?;
        // Bind-mounted workdirs reset on each boot — re-create.
        let _ = inner
            .shell(&format!("mkdir -p {}", self.config.workdir))
            .await;
        Ok(SandboxConsole {
            inner,
            default_timeout_secs: self.config.default_timeout_secs,
            max_output_chars: self.config.max_output_chars,
        })
    }

    async fn shutdown(&mut self) {
        // Stop the VM if it is currently running.
        if let Ok(handle) = MsbSandbox::get(&self.name).await {
            if matches!(
                handle.status(),
                SandboxStatus::Running | SandboxStatus::Draining
            ) {
                if let Ok(connected) = handle.connect().await {
                    let _ = connected.stop_and_wait().await;
                }
            }
        }
        // Remove the registration. We call the static Sandbox::remove() so it
        // re-fetches a fresh handle whose status reflects the stop above.
        // SandboxHandle::remove() rejects Running handles, so calling
        // handle.remove() on the original (stale) handle would silently fail.
        if !self.config.persist {
            let _ = MsbSandbox::remove(&self.name).await;
        }
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Catches VMs that were created in `new()` but never reached
        // `Container::shutdown` (e.g. the `Runenv` was dropped without ever
        // calling `get()`). Persisted VMs survive.
        if self.config.persist {
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
                        if matches!(
                            handle.status(),
                            SandboxStatus::Running | SandboxStatus::Draining
                        ) {
                            if let Ok(connected) = handle.connect().await {
                                let _ = connected.stop_and_wait().await;
                            }
                        }
                    }
                    // Re-fetch via the static method so the fresh handle has the
                    // updated Stopped status; the old handle's cached status is stale.
                    let _ = MsbSandbox::remove(&name).await;
                });
            }
            let _ = tx.send(());
        });
        let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
    }
}

/// `Console` wrapping a running microsandbox VM.
pub struct SandboxConsole {
    inner: MsbSandbox,
    default_timeout_secs: u64,
    max_output_chars: usize,
}

#[async_trait]
impl Console for SandboxConsole {
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
        let raw = self
            .inner
            .exec_with(&program, |b: ExecOptionsBuilder| {
                b.args(args.iter().map(|s| s.as_str()))
                    .timeout(Duration::from_secs(timeout_secs))
            })
            .await;
        handle_exec_result(raw, self.max_output_chars)
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let path_s = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("read: path {} is not valid UTF-8", path.display()))?;
        let bytes = self
            .inner
            .fs()
            .read(path_s)
            .await
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        Ok(bytes.to_vec())
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        let path_s = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("write: path {} is not valid UTF-8", path.display()))?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let parent_s = parent.to_str().ok_or_else(|| {
                anyhow::anyhow!("write: parent {} is not valid UTF-8", parent.display())
            })?;
            // microsandbox's fs().write() does not create parents; mkdir -p first.
            let _ = self
                .inner
                .shell(&format!("mkdir -p '{}'", parent_s.replace('\'', "'\\''")))
                .await;
        }
        self.inner
            .fs()
            .write(path_s, content)
            .await
            .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))
    }
}

/// `start_detached` with one force-stop retry when microsandbox reports the
/// VM as already running — covers the case where a previous `stop_and_wait`
/// failed silently and left a stale handle alive.
async fn start_with_retry(name: &str) -> anyhow::Result<MsbSandbox> {
    match MsbSandbox::start_detached(name).await {
        Ok(s) => Ok(s),
        Err(MicrosandboxError::SandboxStillRunning(_)) => {
            if let Ok(h) = MsbSandbox::get(name).await
                && let Ok(connected) = h.connect().await
            {
                let _ = connected.stop_and_wait().await;
            }
            MsbSandbox::start_detached(name)
                .await
                .map_err(|e| anyhow::anyhow!("sandbox start after force-stop: {e}"))
        }
        Err(e) => Err(anyhow::anyhow!("sandbox start: {e}")),
    }
}

fn handle_exec_result(
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
        + 1;

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
    use super::*;
    use crate::runenv::RunEnv;

    async fn make_env() -> RunEnv {
        RunEnv::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("failed to create sandbox"),
        )
    }

    #[tokio::test]
    async fn test_exec_stdout() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        let result = handle
            .exec("echo".to_string(), vec!["hello".to_string()], None)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        let result = handle
            .exec("sleep".to_string(), vec!["10".to_string()], Some(1))
            .await
            .unwrap();
        assert!(result.timed_out);
    }

    #[tokio::test]
    async fn test_vm_stays_up_across_exec_calls() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        {
            let r = handle
                .exec(
                    "sh".to_string(),
                    vec![
                        "-c".to_string(),
                        "echo first > /workspace/marker".to_string(),
                    ],
                    None,
                )
                .await
                .unwrap();
            assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        }
        {
            let r = handle
                .exec(
                    "cat".to_string(),
                    vec!["/workspace/marker".to_string()],
                    None,
                )
                .await
                .unwrap();
            assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
            assert!(r.stdout.contains("first"));
        }
    }
}
