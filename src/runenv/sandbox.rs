//! microsandbox-backed `Machine` for runenv.
//!
//! Adapted from [`crate::runenv::sandbox`] to the v2 Machine/Console split.
//! Heavyweight, fallible setup happens in `Sandbox::new`; `Machine::boot`
//! restarts the VM, and `Machine::shutdown` stops/removes it.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Duration,
};

use async_trait::async_trait;
use microsandbox::{
    ExecOutput, MicrosandboxError, Sandbox as MsbSandbox, Snapshot,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxBuilder, SandboxStatus},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{Console, ExecResult, Machine};
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
    /// Mount a microsandbox named volume, stored under the resolved
    /// microsandbox home directory (e.g. `~/.microsandbox/volumes/<name>/`
    /// or `$MSB_HOME/volumes/<name>/`).
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

    /// Override microsandbox home (binaries, db, snapshots, per-sandbox state).
    /// Equivalent to setting `MSB_HOME`. Takes effect on the first
    /// `Sandbox::new` only; later calls with a different value return an
    /// error. Defaults to `$MSB_HOME` or `~/.microsandbox`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home: Option<PathBuf>,
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
            home: None,
        }
    }
}

fn fresh_sandbox_name() -> String {
    format!("ailoy-{}", Uuid::new_v4())
}

/// `Machine` implementation backed by a microsandbox VM.
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

        if let Some(home) = &config.home {
            check_home_conflict(home, std::env::var_os("MSB_HOME").as_deref())?;
            unsafe {
                std::env::set_var("MSB_HOME", home);
            }
        }

        if !microsandbox::setup::is_installed() {
            log::warn!(
                "microsandbox runtime not found — downloading to {}, \
                 this may take a moment",
                microsandbox::config::config().home().display(),
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

    /// Fork this sandbox into a new one initialized from a filesystem snapshot.
    ///
    /// The source sandbox must be stopped; running sandboxes are rejected
    /// with an error. The returned `Sandbox` is registered but stopped, ready
    /// to be booted.
    pub async fn fork(&self, new_cfg: SandboxConfig) -> anyhow::Result<Sandbox> {
        if vm_is_running(&self.name).await {
            anyhow::bail!("cannot fork running sandbox '{}'; stop it first", self.name);
        }

        let new_name = new_cfg.name.clone().unwrap_or_else(fresh_sandbox_name);
        validate_sandbox_name(&new_name)?;
        let snap_name = format!("fork-{new_name}");

        let handle = MsbSandbox::get(&self.name)
            .await
            .map_err(|e| anyhow::anyhow!("fork: source sandbox not found in DB: {e}"))?;

        handle
            .snapshot(&snap_name)
            .await
            .map_err(|e| anyhow::anyhow!("fork: snapshot failed: {e}"))?;

        let result = create_from_snapshot(&new_name, &snap_name, &new_cfg).await;

        // Always clean up the snapshot artifact, regardless of outcome.
        if let Err(e) = Snapshot::remove(&snap_name, true).await {
            log::warn!("fork: failed to clean up snapshot '{snap_name}': {e}");
        }

        result.map_err(|e| {
            // Best-effort removal of any partially-created sandbox.
            let nn = new_name.clone();
            tokio::spawn(async move {
                if let Ok(h) = MsbSandbox::get(&nn).await {
                    let _ = h.remove().await;
                }
            });
            e
        })?;

        Ok(Sandbox {
            config: SandboxConfig {
                name: Some(new_name.clone()),
                ..new_cfg
            },
            name: new_name,
        })
    }

    async fn stop(name: &str, persist: bool) {
        // Stop the VM if it is currently running.
        if let Ok(handle) = MsbSandbox::get(name).await {
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
        if !persist {
            let _ = MsbSandbox::remove(name).await;
        }
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
impl Machine for Sandbox {
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
        Self::stop(&self.name, self.config.persist).await;
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Catches VMs that were created in `new()` but never reached
        // `Machine::shutdown` (e.g. the `Runenv` was dropped without ever
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
                rt.block_on(Self::stop(&name, false));
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

/// Return `Err` if `requested` disagrees with an already-set `MSB_HOME`.
/// Microsandbox caches the resolved home in a `OnceLock`, so changing it
/// mid-process is silently ignored. Surface the mismatch instead.
fn check_home_conflict(requested: &Path, current: Option<&std::ffi::OsStr>) -> anyhow::Result<()> {
    if let Some(prev) = current
        && prev != requested.as_os_str()
    {
        anyhow::bail!(
            "MSB_HOME already set to {:?}; cannot change to {:?} in the same process",
            prev,
            requested.display(),
        );
    }
    Ok(())
}

/// Validate that `name` won't produce a socket path that exceeds the OS limit.
fn validate_sandbox_name(name: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    const SUN_PATH_MAX: usize = 104;
    #[cfg(not(target_os = "macos"))]
    const SUN_PATH_MAX: usize = 108;

    let home = microsandbox::config::config().home();
    let home_str = home
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("home directory path is not valid UTF-8"))?;

    const SANDBOXES_PREFIX: &str = "/sandboxes/";
    const RUNTIME_SUFFIX: &str = "/runtime/agent.sock";

    let socket_path_len =
        home_str.len() + SANDBOXES_PREFIX.len() + name.len() + RUNTIME_SUFFIX.len() + 1; // null terminator

    if socket_path_len > SUN_PATH_MAX {
        let max_name = SUN_PATH_MAX
            .saturating_sub(home_str.len() + SANDBOXES_PREFIX.len() + RUNTIME_SUFFIX.len() + 1);
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

async fn create_from_snapshot(
    new_name: &str,
    snap_name: &str,
    config: &SandboxConfig,
) -> anyhow::Result<MsbSandbox> {
    let mut builder = MsbSandbox::builder(new_name)
        .from_snapshot(snap_name)
        .cpus(config.cpus)
        .memory(config.memory_mib);

    for (k, v) in &config.env {
        builder = builder.env(k.as_str(), v.as_str());
    }
    if config.disable_network {
        builder = builder.disable_network();
    }
    for mount in &config.volumes {
        builder = apply_volume_mount(builder, mount);
    }

    let sb = builder
        .create_detached()
        .await
        .map_err(|e| anyhow::anyhow!("fork: create from snapshot: {e}"))?;

    // Workdir already exists from the snapshot, but ensure it is present.
    let _ = sb.shell(&format!("mkdir -p {}", config.workdir)).await;
    let _ = sb.shell("sync").await;
    sb.stop_and_wait()
        .await
        .map_err(|e| anyhow::anyhow!("fork: stop new sandbox after creation: {e}"))?;

    Ok(sb)
}

async fn vm_is_running(name: &str) -> bool {
    MsbSandbox::get(name)
        .await
        .map(|h| matches!(h.status(), SandboxStatus::Running | SandboxStatus::Draining))
        .unwrap_or(false)
}

/// Remove a `persist = true` sandbox by name without holding a [`Sandbox`] instance.
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
    use std::time::Instant;

    use super::*;
    use crate::runenv::RunEnv;

    fn short_id() -> String {
        Uuid::new_v4().to_string()[..8].to_string()
    }

    async fn make_env() -> RunEnv {
        RunEnv::sandbox(SandboxConfig::default())
            .await
            .expect("failed to create sandbox")
    }

    /// `SandboxConfig::name` round-trips through JSON: Some is preserved, None is omitted.
    #[test]
    fn test_sandbox_config_name_serde_roundtrip() {
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

    /// `check_home_conflict` accepts unset env, matching env, and rejects a mismatch.
    #[test]
    fn test_check_home_conflict() {
        let requested = Path::new("/tmp/msb-a");
        assert!(check_home_conflict(requested, None).is_ok());
        assert!(check_home_conflict(requested, Some(requested.as_os_str())).is_ok());

        let other = std::ffi::OsString::from("/tmp/msb-b");
        let err = check_home_conflict(requested, Some(&other))
            .expect_err("mismatched MSB_HOME must error");
        let msg = err.to_string();
        assert!(
            msg.contains("MSB_HOME already set"),
            "error should mention MSB_HOME conflict, got: {msg}"
        );
    }

    /// `home` field round-trips through serde when set, and is omitted when `None`.
    #[test]
    fn test_sandbox_config_home_serde_roundtrip() {
        let cfg_with = SandboxConfig {
            home: Some(PathBuf::from("/tmp/example")),
            ..SandboxConfig::default()
        };
        let json = serde_json::to_string(&cfg_with).expect("serialize");
        assert!(
            json.contains("\"home\":\"/tmp/example\""),
            "expected home field in JSON, got: {json}"
        );
        let back: SandboxConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.home.as_deref(), Some(Path::new("/tmp/example")));

        let cfg_default = SandboxConfig::default();
        let json_default = serde_json::to_string(&cfg_default).expect("serialize default");
        assert!(
            !json_default.contains("\"home\""),
            "home must be omitted when None, got: {json_default}"
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

    /// A single boot serves multiple `exec` calls without the VM going down
    /// between them.
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

    /// Dropping the last handle shuts the VM down; `get()` boots it again.
    #[tokio::test]
    async fn test_stop_and_start() {
        let name = format!("at-ss-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        let handle = env.get().await.unwrap();
        assert!(
            vm_is_running(&name).await,
            "VM should be running after get()"
        );
        drop(handle);
        assert!(
            !vm_is_running(&name).await,
            "VM should be stopped after handle drop"
        );

        let handle = env.get().await.unwrap();
        assert!(vm_is_running(&name).await, "VM should be running again");
        drop(handle);
        remove_persisted(&name).await.ok();
    }

    /// Repeated boot/drop cycles are safe — no resource leak or double-stop error.
    #[tokio::test]
    async fn test_start_stop_idempotent() {
        let name = format!("at-si-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        for _ in 0..3 {
            let handle = env.get().await.unwrap();
            assert!(vm_is_running(&name).await);
            drop(handle);
            assert!(!vm_is_running(&name).await);
        }
        remove_persisted(&name).await.ok();
    }

    /// With `persist = true`, files written during one boot survive the next.
    #[tokio::test]
    async fn test_filesystem_persists_across_stop_start() {
        let name = format!("at-fp-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        {
            let handle = env.get().await.unwrap();
            handle
                .exec_shell("echo hello > /workspace/test.txt".to_string(), None)
                .await
                .expect("write failed");
        }

        let handle = env.get().await.unwrap();
        let bytes = handle
            .read(Path::new("/workspace/test.txt"))
            .await
            .expect("read failed");
        let content = String::from_utf8_lossy(&bytes);
        assert!(
            content.contains("hello"),
            "file should survive stop/start cycle, got: {content:?}"
        );
        drop(handle);
        remove_persisted(&name).await.ok();
    }

    #[tokio::test]
    async fn test_restart_latency() {
        let name = format!("at-rl-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        drop(env.get().await.unwrap());

        let t = Instant::now();
        let _h = env.get().await.unwrap();
        let restart_ms = t.elapsed().as_millis();

        assert!(
            restart_ms < 3000,
            "restart should be well under 3s, got {restart_ms}ms"
        );
        drop(_h);
        remove_persisted(&name).await.ok();
    }

    /// Two cloned `Arc<RunEnvHandle>`s can issue `exec` concurrently from
    /// independent tasks; the underlying console serializes them.
    #[tokio::test]
    async fn test_shared_arc_serial_ops() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();

        let h1 = handle.clone();
        let h2 = handle.clone();
        let t1 = tokio::spawn(async move { h1.exec_shell("echo a".to_string(), None).await });
        let t2 = tokio::spawn(async move { h2.exec_shell("echo b".to_string(), None).await });

        let (r1, r2) = tokio::join!(t1, t2);
        let r1 = r1.expect("task1 panic").expect("task1 shell error");
        let r2 = r2.expect("task2 panic").expect("task2 shell error");

        assert_eq!(r1.exit_code, 0, "task1 exit_code: {:?}", r1.stderr);
        assert_eq!(r2.exit_code, 0, "task2 exit_code: {:?}", r2.stderr);
    }

    /// Sequential reads on a shared handle never lose state — confirms there
    /// is no per-call console teardown.
    #[tokio::test]
    async fn test_sequential_ops_on_shared_arc() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();

        handle
            .write(Path::new("/workspace/marker.txt"), b"sequential")
            .await
            .expect("write failed");

        for i in 0..10 {
            let bytes = handle
                .read(Path::new("/workspace/marker.txt"))
                .await
                .unwrap_or_else(|e| panic!("read failed on iteration {i}: {e}"));
            let content = String::from_utf8_lossy(&bytes);
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
        let name = format!("at-rp-{}", short_id());

        {
            let env = RunEnv::sandbox(SandboxConfig {
                persist: true,
                name: Some(name.clone()),
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox");
            let handle = env.get().await.unwrap();
            handle
                .write(Path::new("/workspace/marker.txt"), b"persist_marker")
                .await
                .expect("write failed");
        }

        remove_persisted(&name)
            .await
            .expect("remove_persisted failed");

        let env2 = RunEnv::sandbox(SandboxConfig {
            persist: true,
            name: Some(name.clone()),
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to recreate sandbox");
        let handle = env2.get().await.unwrap();
        let result = handle.read(Path::new("/workspace/marker.txt")).await;
        assert!(
            result.is_err(),
            "fresh VM should not contain the marker file from the previous run"
        );
        drop(handle);
        remove_persisted(&name).await.ok();
    }

    #[tokio::test]
    async fn test_bind_mount_host_to_guest() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");
        let host_file = host_dir.path().join("hello.txt");
        std::fs::write(&host_file, "bind_mount_works").expect("failed to write host file");

        let env = RunEnv::sandbox(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/host".to_string(),
                readonly: false,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");
        let handle = env.get().await.unwrap();

        let result = handle
            .exec_shell("cat /mnt/host/hello.txt".to_string(), None)
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

        let env = RunEnv::sandbox(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/ro".to_string(),
                readonly: true,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");
        let handle = env.get().await.unwrap();

        let result = handle
            .exec_shell("echo should_fail > /mnt/ro/file.txt".to_string(), None)
            .await
            .expect("shell failed");
        assert_ne!(result.exit_code, 0, "write to read-only mount should fail");
    }

    #[tokio::test]
    async fn test_bind_mount_guest_write_visible_on_host() {
        let host_dir = tempfile::tempdir().expect("failed to create temp dir");

        let env = RunEnv::sandbox(SandboxConfig {
            volumes: vec![VolumeMount::Bind {
                host: host_dir.path().to_path_buf(),
                guest: "/mnt/shared".to_string(),
                readonly: false,
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");
        let handle = env.get().await.unwrap();

        let result = handle
            .exec_shell("echo guest_wrote > /mnt/shared/out.txt".to_string(), None)
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
        let env = RunEnv::sandbox(SandboxConfig {
            volumes: vec![VolumeMount::Tmpfs {
                guest: "/mnt/tmp".to_string(),
                size_mib: Some(64),
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");
        let handle = env.get().await.unwrap();

        let result = handle
            .exec_shell(
                "echo tmpfs_ok > /mnt/tmp/test.txt && cat /mnt/tmp/test.txt".to_string(),
                None,
            )
            .await
            .expect("shell failed");
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(
            result.stdout.contains("tmpfs_ok"),
            "stdout: {:?}",
            result.stdout
        );
    }

    // ── fork tests ──────────────────────────────────────────────────────────
    //
    // Fork operates on the underlying `Sandbox`, not on a booted handle, so
    // these tests bypass `RunEnv` and drive `Machine::boot` / `Machine::shutdown`
    // directly.

    #[tokio::test]
    async fn test_fork_running_returns_error() {
        let src_name = format!("sf-run-{}", short_id());
        let mut src = Sandbox::new(SandboxConfig {
            name: Some(src_name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .expect("create source sandbox");

        // Boot the VM directly so we can observe the "running" state when
        // calling `fork`.
        let _console = src.boot().await.expect("boot");
        assert!(
            vm_is_running(&src_name).await,
            "should be running before fork"
        );

        let result = src
            .fork(SandboxConfig {
                name: Some(format!("sf-ch-{}", short_id())),
                persist: false,
                ..SandboxConfig::default()
            })
            .await;

        assert!(
            result.is_err(),
            "fork() on a running sandbox must return Err, got Ok"
        );

        drop(_console);
        src.shutdown().await;
        remove_persisted(&src_name).await.ok();
    }

    #[tokio::test]
    async fn test_fork_copies_workspace_file() {
        let src_name = format!("sf-ws-{}", short_id());
        let child_name = format!("sf-ch-{}", short_id());

        let mut src = Sandbox::new(SandboxConfig {
            name: Some(src_name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .expect("create source");

        {
            let console = src.boot().await.expect("boot src");
            console
                .exec_shell("echo fork_content > /workspace/note.txt".to_string(), None)
                .await
                .expect("write note");
        }
        src.shutdown().await;

        let mut child = src
            .fork(SandboxConfig {
                name: Some(child_name.clone()),
                persist: true,
                ..SandboxConfig::default()
            })
            .await
            .expect("fork");

        let child_console = child.boot().await.expect("boot child");
        let bytes = child_console
            .read(Path::new("/workspace/note.txt"))
            .await
            .expect("read from child");
        let content = String::from_utf8_lossy(&bytes);

        assert!(
            content.contains("fork_content"),
            "child should have note from source, got: {content:?}"
        );

        drop(child_console);
        child.shutdown().await;
        remove_persisted(&src_name).await.ok();
        remove_persisted(&child_name).await.ok();
    }

    #[tokio::test]
    async fn test_fork_is_isolated() {
        let src_name = format!("sf-iso-a-{}", short_id());
        let child_name = format!("sf-iso-b-{}", short_id());

        let mut src = Sandbox::new(SandboxConfig {
            name: Some(src_name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .expect("create source");

        {
            let console = src.boot().await.expect("boot src");
            console
                .exec_shell("echo original > /workspace/data.txt".to_string(), None)
                .await
                .expect("write original");
        }
        src.shutdown().await;

        let mut child = src
            .fork(SandboxConfig {
                name: Some(child_name.clone()),
                persist: true,
                ..SandboxConfig::default()
            })
            .await
            .expect("fork");

        {
            let console = child.boot().await.expect("boot child");
            console
                .exec_shell("echo mutated > /workspace/data.txt".to_string(), None)
                .await
                .expect("mutate child");
        }
        child.shutdown().await;

        let src_console = src.boot().await.expect("boot src after fork");
        let bytes = src_console
            .read(Path::new("/workspace/data.txt"))
            .await
            .expect("read source after child mutation");
        let src_content = String::from_utf8_lossy(&bytes);

        assert!(
            src_content.contains("original") && !src_content.contains("mutated"),
            "source must not be affected by child write, got: {src_content:?}"
        );

        drop(src_console);
        src.shutdown().await;
        remove_persisted(&src_name).await.ok();
        remove_persisted(&child_name).await.ok();
    }

    #[tokio::test]
    async fn test_fork_snapshot_cleaned_up() {
        let src_name = format!("sf-snap-{}", short_id());
        let child_name = format!("sf-ch-{}", short_id());
        let snap_name = format!("fork-{child_name}");

        let src = Sandbox::new(SandboxConfig {
            name: Some(src_name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .expect("create source");

        let mut child = src
            .fork(SandboxConfig {
                name: Some(child_name.clone()),
                persist: true,
                ..SandboxConfig::default()
            })
            .await
            .expect("fork");

        let snap_dir = microsandbox::config::config()
            .home()
            .join("snapshots")
            .join(&snap_name);

        assert!(
            !snap_dir.exists(),
            "snapshot should be deleted after fork, still found at {snap_dir:?}"
        );

        child.shutdown().await;
        remove_persisted(&src_name).await.ok();
        remove_persisted(&child_name).await.ok();
    }

    #[tokio::test]
    async fn test_named_volume_persists_across_sandboxes() {
        use microsandbox::Volume;

        let vol_name = format!("ailoy-test-vol-{}", Uuid::new_v4());

        Volume::builder(&vol_name)
            .create()
            .await
            .expect("failed to create named volume");

        let write_result = async {
            let env = RunEnv::sandbox(SandboxConfig {
                volumes: vec![VolumeMount::Named {
                    name: vol_name.clone(),
                    guest: "/mnt/vol".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create first sandbox");
            let h = env.get().await.unwrap();
            h.exec_shell("echo named_vol_works > /mnt/vol/data.txt".to_string(), None)
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
            let env = RunEnv::sandbox(SandboxConfig {
                volumes: vec![VolumeMount::Named {
                    name: vol_name.clone(),
                    guest: "/mnt/vol".to_string(),
                    readonly: false,
                }],
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create second sandbox");
            let h = env.get().await.unwrap();
            h.exec_shell("cat /mnt/vol/data.txt".to_string(), None)
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
