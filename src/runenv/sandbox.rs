//! microsandbox-backed `Machine` for runenv.
//!
//! Heavyweight, fallible setup happens in `Sandbox::new`; `Machine::start`
//! starts the VM, `Machine::stop` stops it, and dropping the `Sandbox`
//! removes the VM definition (unless `config.persist` is `true`).

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    time::Duration,
};

use async_trait::async_trait;
use microsandbox::{
    ExecOutput, MicrosandboxError, NetworkPolicy, Sandbox as MsbSandbox, Snapshot,
    sandbox::{ExecOptionsBuilder, PullPolicy, SandboxBuilder, SandboxStatus},
    validate_sandbox_name,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::OnceCell;
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
        /// Name of the microsandbox volume.
        name: String,
        /// Absolute guest path.
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
        /// When `true`, create the named volume atomically with the sandbox
        /// if it does not yet exist, or reuse a compatible existing one. The
        /// volume row is inserted in the same DB transaction as the sandbox
        /// row, so a parallel scan never observes an orphan-shaped row.
        ///
        /// When `false` (default), the volume must already exist.
        #[serde(default)]
        create_if_missing: bool,
        /// Labels attached to the volume when it is created. When reusing an
        /// existing volume, microsandbox requires these to match the
        /// persisted labels.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        labels: Vec<(String, String)>,
    },
    /// Memory-backed temporary filesystem. Disappears when the sandbox stops.
    Tmpfs {
        /// Absolute guest path.
        guest: String,
        /// Size limit in MiB. `None` means no limit.
        size_mib: Option<u32>,
    },
}

impl VolumeMount {
    /// Guest mount path. Matches the `guest` field across all variants.
    pub fn guest_path(&self) -> &str {
        match self {
            VolumeMount::Bind { guest, .. }
            | VolumeMount::Named { guest, .. }
            | VolumeMount::Tmpfs { guest, .. } => guest,
        }
    }
}

/// Label key/value applied to volumes ailoy provisions. The startup sweep
/// removes only labelled volumes so volumes from other tools are never touched.
const AILOY_VOLUME_LABEL_KEY: &str = "ailoy.managed";
const AILOY_VOLUME_LABEL_VALUE: &str = "true";

/// Prefix for the per-sandbox `/tmp` volume injected by `Sandbox::new`.
const AILOY_TMP_VOLUME_PREFIX: &str = "ailoy-tmp-";

fn tmp_volume_name(sandbox_name: &str) -> String {
    format!("{AILOY_TMP_VOLUME_PREFIX}{sandbox_name}")
}

static STARTUP_SWEEP_DONE: OnceCell<()> = OnceCell::const_new();

/// Remove `ailoy-tmp-*` volumes left over by previous process invocations
/// (panic, SIGKILL, or a `RunEnvHandle::Drop` background thread racing
/// process exit). Targets only volumes that carry the `ailoy.managed` label
/// and whose owner sandbox is no longer present, so volumes from other
/// tools and from an in-flight `Sandbox::new` in a parallel process are
/// left alone.
async fn sweep_orphan_tmp_volumes() {
    let active: HashSet<String> = match MsbSandbox::list().await {
        Ok(handles) => handles.into_iter().map(|h| h.name().to_string()).collect(),
        Err(e) => {
            log::warn!("startup sweep: list sandboxes failed: {e}");
            return;
        }
    };

    let volumes = match microsandbox::Volume::list().await {
        Ok(v) => v,
        Err(e) => {
            log::warn!("startup sweep: list volumes failed: {e}");
            return;
        }
    };

    for v in volumes {
        let vname = v.name().to_string();
        let Some(owner) = vname.strip_prefix(AILOY_TMP_VOLUME_PREFIX) else {
            continue;
        };
        let labelled = v
            .labels()
            .iter()
            .any(|(k, val)| k == AILOY_VOLUME_LABEL_KEY && val == AILOY_VOLUME_LABEL_VALUE);
        if !labelled {
            continue;
        }
        if active.contains(owner) {
            continue;
        }
        if let Err(e) = microsandbox::Volume::remove(&vname).await {
            log::warn!("startup sweep: remove '{vname}' failed: {e}");
        }
    }
}

/// Guest network reachability for a sandbox — a single egress policy.
///
/// The sandbox host (`host.microsandbox.internal`, e.g. the ailoy VFS forward
/// server) is reachable in **every** variant, so VFS-in-sandbox always works and
/// no build-time validation is needed. The variants differ only in how much of
/// the *outside* the guest can reach. Each maps to a microsandbox
/// [`NetworkPolicy`] with `default_ingress: Allow` (published-port behavior).
///
/// There is intentionally no fully-offline (host-denied) variant — that isolation
/// posture is rarely needed and is omitted for simplicity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SandboxNetwork {
    /// Host only — the guest reaches the sandbox host but NOT the public
    /// internet, LAN, loopback, or metadata. Least-privilege for VFS-in-sandbox
    /// (the forwarder works; the LLM-driven shell gets no internet).
    HostOnly,
    /// Host + public internet, but NOT private LAN (RFC1918), loopback,
    /// link-local, cloud-metadata (169.254.169.254), or multicast. The default.
    #[default]
    Public,
    /// Unrestricted egress and ingress: everything `Public` reaches PLUS private
    /// LAN, loopback, link-local, cloud-metadata, and multicast. Grant
    /// deliberately — this reopens SSRF-to-metadata and local-network access.
    Full,
}

#[cfg(feature = "sandbox")]
impl SandboxNetwork {
    /// Apply this policy to a sandbox builder. Every variant sets an explicit
    /// policy that allows host egress (so the VFS forwarder can always reach the
    /// host forward server).
    fn apply(self, builder: SandboxBuilder) -> SandboxBuilder {
        builder.network(move |n| n.policy(self.policy()))
    }

    /// The microsandbox policy for this variant.
    fn policy(self) -> NetworkPolicy {
        use microsandbox_network::policy::{Action, Destination, DestinationGroup, Rule};
        // `allow_egress(Host)` permits any port to the host — including :53, so the
        // guest's `host.microsandbox.internal` DNS lookup (handled by the host
        // resolver) works without a separate DNS allow.
        let host = || Rule::allow_egress(Destination::Group(DestinationGroup::Host));
        let public = || Rule::allow_egress(Destination::Group(DestinationGroup::Public));
        match self {
            SandboxNetwork::HostOnly => NetworkPolicy {
                default_egress: Action::Deny,
                default_ingress: Action::Allow,
                rules: vec![host()],
            },
            SandboxNetwork::Public => NetworkPolicy {
                default_egress: Action::Deny,
                default_ingress: Action::Allow,
                rules: vec![public(), host()],
            },
            SandboxNetwork::Full => NetworkPolicy::allow_all(),
        }
    }
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
    /// Created automatically after start if it does not already exist.
    pub workdir: String,

    /// Environment variables passed to every command.
    pub env: HashMap<String, String>,

    /// Guest network reachability; see [`SandboxNetwork`]. Default:
    /// [`SandboxNetwork::Public`] (host + public internet). Every variant lets the
    /// guest reach the sandbox host, so VFS-in-sandbox always works; use
    /// [`SandboxNetwork::HostOnly`] to additionally keep the guest off the
    /// internet, or [`SandboxNetwork::Full`] to also allow LAN/loopback/metadata.
    #[serde(default)]
    pub network: SandboxNetwork,

    /// Per-exec timeout in seconds. Default: `60`.
    pub default_timeout_secs: u64,

    /// Maximum characters to keep from stdout/stderr. Default: `30_000`.
    pub max_output_chars: usize,

    /// When `true`, the VM definition survives drop of the [`Sandbox`] struct
    /// and can be reattached by name in a future session. Use
    /// [`Sandbox::remove_persisted`] to delete it explicitly.
    ///
    /// Independent of `Machine::start`/`Machine::stop` cycles — those only
    /// start and stop the VM regardless of this flag. Default: `false`.
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
            network: SandboxNetwork::default(),
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
/// The VM is created at construction time and left stopped; `start()` boots
/// the VM for the duration of the returned [`SandboxConsole`], and `stop()`
/// halts it. The VM definition is removed when the `Sandbox` value is
/// dropped, unless `config.persist` is `true`.
pub struct Sandbox {
    config: SandboxConfig,
    name: String,
}

impl Sandbox {
    /// Install the microsandbox runtime if needed, create or attach to the
    /// named VM, ensure `workdir` exists, then stop the VM.
    pub async fn new(mut config: SandboxConfig) -> anyhow::Result<Self> {
        use anyhow::Context as _;

        if let Some(home) = &config.home {
            check_home_conflict(home, std::env::var_os("MSB_HOME").as_deref())?;
            unsafe {
                std::env::set_var("MSB_HOME", home);
            }
        }

        if !microsandbox::setup::is_installed() {
            let home = microsandbox::config::load_persisted_config_or_default()
                .map(|c| c.home())
                .unwrap_or_default();
            log::warn!(
                "microsandbox runtime not found — downloading to {}, \
                 this may take a moment",
                home.display(),
            );
            microsandbox::setup::install()
                .await
                .context("microsandbox install")?;
            log::info!("microsandbox runtime installed");
        }

        STARTUP_SWEEP_DONE
            .get_or_init(|| async { sweep_orphan_tmp_volumes().await })
            .await;

        let name = config.name.clone().unwrap_or_else(fresh_sandbox_name);
        validate_sandbox_name(&name).map_err(|e| anyhow::anyhow!("sandbox name '{name}': {e}"))?;

        // Pre-populate `/tmp` with a per-sandbox named volume so it
        // survives the start/stop cycle that `RunEnvHandle::Drop` drives.
        // A user-supplied `/tmp` mount takes precedence.
        if !config.volumes.iter().any(|v| v.guest_path() == "/tmp") {
            config.volumes.push(VolumeMount::Named {
                name: tmp_volume_name(&name),
                guest: "/tmp".to_string(),
                readonly: false,
                create_if_missing: true,
                labels: vec![(
                    AILOY_VOLUME_LABEL_KEY.to_string(),
                    AILOY_VOLUME_LABEL_VALUE.to_string(),
                )],
            });
        }

        // Reconnecting to a persisted sandbox that may still be running (or whose
        // previous owner's async stop hasn't finished): `start_detached_resilient`
        // handles `SandboxStillRunning` (force-stop then restart) and bounds the
        // intermittent microsandbox SQLite hang so the reconnect can't stall the
        // agent indefinitely. The fresh-create path is one-shot, not the churn hot
        // path.
        let inner = if config.persist && MsbSandbox::get(&name).await.is_ok() {
            start_detached_resilient(&name).await?
        } else {
            create_registered(&name, &config)
                .await
                .context("sandbox create-registered")?
        };
        let _ = inner.shell(&format!("mkdir -p {}", config.workdir)).await;
        inner.stop().await?;

        Ok(Self { config, name })
    }

    /// Fork this sandbox into a new one initialized from a filesystem snapshot.
    ///
    /// The source sandbox must be stopped; running sandboxes are rejected
    /// with an error. The returned `Sandbox` is registered but stopped, ready
    /// to be started.
    pub async fn fork(&self, new_cfg: SandboxConfig) -> anyhow::Result<Sandbox> {
        if vm_is_running(&self.name).await {
            anyhow::bail!("cannot fork running sandbox '{}'; stop it first", self.name);
        }

        let new_name = new_cfg.name.clone().unwrap_or_else(fresh_sandbox_name);
        validate_sandbox_name(&new_name)
            .map_err(|e| anyhow::anyhow!("sandbox name '{new_name}': {e}"))?;
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

        result.inspect_err(|_| {
            // Best-effort removal of any partially-created sandbox.
            let nn = new_name.clone();
            tokio::spawn(async move {
                if let Ok(h) = MsbSandbox::get(&nn).await {
                    let _ = h.remove().await;
                }
            });
        })?;

        Ok(Sandbox {
            config: SandboxConfig {
                name: Some(new_name.clone()),
                ..new_cfg
            },
            name: new_name,
        })
    }

    /// Stop the named VM if it is running. Does not remove the VM definition.
    async fn stop_vm(name: &str) {
        match MsbSandbox::get(name).await {
            Err(e) => log::warn!("sandbox stop: get '{name}' failed: {e}"),
            Ok(handle) => {
                let _ = handle.stop().await;
            }
        }
    }

    /// Stop the VM and remove its definition. Called only from `Drop`
    /// for non-persist sandboxes; never from `Machine::stop`.
    async fn remove(name: &str) {
        Self::stop_vm(name).await;
        let _ = MsbSandbox::remove(name).await;
        // Drop the per-sandbox /tmp volume provisioned in `Sandbox::new`.
        // No-op if the user supplied their own /tmp mount.
        let _ = microsandbox::Volume::remove(&tmp_volume_name(name)).await;
    }

    /// Remove a `persist = true` sandbox by name without holding a [`Sandbox`] instance.
    ///
    /// Intended for explicit cleanup when the `Sandbox` object is no longer available
    /// (e.g. after a process restart). For `persist = false` sandboxes, removal happens
    /// automatically on drop.
    ///
    /// Idempotent: if the named sandbox does not exist, returns `Ok(())`.
    pub async fn remove_persisted(name: &str) -> anyhow::Result<()> {
        match MsbSandbox::get(name).await {
            Err(_) => Ok(()),
            Ok(handle) => {
                if matches!(
                    handle.status_snapshot(),
                    SandboxStatus::Running | SandboxStatus::Draining
                ) {
                    handle.stop().await?;
                }
                handle.remove().await?;
                let _ = microsandbox::Volume::remove(&tmp_volume_name(name)).await;
                Ok(())
            }
        }
    }

    /// Returns `true` if a persisted sandbox with the given name already exists,
    /// without creating or starting it.
    ///
    /// This is a lightweight existence probe — it never modifies sandbox state.
    /// Returns `false` on any error (e.g. microsandbox runtime not installed).
    pub async fn exists(name: &str) -> bool {
        MsbSandbox::get(name).await.is_ok()
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

    async fn start(&mut self) -> anyhow::Result<SandboxConsole> {
        let inner = start_and_connect(&self.name).await?;
        // Bind-mounted workdirs reset on each start — re-create.
        let _ = inner
            .shell(&format!("mkdir -p {}", self.config.workdir))
            .await;
        Ok(SandboxConsole {
            inner,
            workdir: self.config.workdir.clone(),
            default_timeout_secs: self.config.default_timeout_secs,
            max_output_chars: self.config.max_output_chars,
        })
    }

    /// Stops the VM only; the VM definition is removed when the `Sandbox` is dropped.
    async fn stop(&mut self) {
        Self::stop_vm(&self.name).await;
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Drop is the sole owner of VM definition removal for non-persist
        // sandboxes. `Machine::stop` only stops the VM; removal happens
        // here so the VM definition survives multiple start/stop cycles
        // within the same `Sandbox` lifetime.
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
                rt.block_on(Self::remove(&name));
            }
            let _ = tx.send(());
        });
        // let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
        let _ = rx;
    }
}

/// `Console` wrapping a running microsandbox VM.
pub struct SandboxConsole {
    inner: MsbSandbox,
    workdir: String,
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
                    .cwd(self.workdir.clone())
                    .timeout(Duration::from_secs(timeout_secs))
            })
            .await;
        handle_exec_result(raw, self.max_output_chars)
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        // agentd resolves relative paths against its own cwd (/), not the
        // sandbox workdir. Resolve here so behaviour matches exec_shell.
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            PathBuf::from(&self.workdir).join(path)
        };
        let path_s = abs
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("read: path {} is not valid UTF-8", abs.display()))?;
        let bytes = self
            .inner
            .fs()
            .read(path_s)
            .await
            .map_err(|e| anyhow::anyhow!("read {}: {e}", abs.display()))?;
        Ok(bytes.to_vec())
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        // Same cwd resolution as read — agentd uses /, exec_shell uses workdir.
        let abs = if path.is_absolute() {
            path.to_path_buf()
        } else {
            PathBuf::from(&self.workdir).join(path)
        };
        let path_s = abs
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("write: path {} is not valid UTF-8", abs.display()))?;
        if let Some(parent) = abs.parent()
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
            .map_err(|e| anyhow::anyhow!("write {}: {e}", abs.display()))
    }
}

/// Whether a microsandbox error message looks like a *transient* startup race
/// worth retrying (vs. a genuine config error that should surface promptly).
/// Under rapid sandbox churn a previous owner's async stop can SIGKILL the VM
/// just as a new start/create brings it up, so the agent relay never comes up.
fn is_transient_msb_error(msg: &str) -> bool {
    msg.contains("agent relay")
        || msg.contains("process exited")
        || msg.contains("SIGKILL")
        || msg.contains("signal: 9")
}

/// `MsbSandbox::start_detached` with the embedded-SQLite hang bounded out.
///
/// Under rapid sandbox churn — the agent-k pattern of dropping and recreating the
/// runtime against a persisted sandbox — microsandbox's SQLite state layer can
/// intermittently block while acquiring a DB connection
/// (`sqlx ConnectionWorker::establish`), hanging the start *indefinitely*. That
/// stalls the agent's reconnect and, with it, all provider-filesystem access via
/// the VM. Observed in soak tests as a ~10% multi-minute hang on reconnect.
///
/// Bound each attempt with a timeout and retry a few times: a transient lock is
/// ridden over (a fresh connection attempt after abandoning the stuck one
/// usually succeeds), and a genuinely wedged DB surfaces as a fast error instead
/// of an unbounded hang. `SandboxStillRunning` keeps its prior handling
/// (force-stop then retry); the force-stop is bounded too so it can't hang either.
async fn start_detached_resilient(name: &str) -> anyhow::Result<MsbSandbox> {
    const ATTEMPTS: usize = 4;
    const PER_TRY: Duration = Duration::from_secs(25);
    const BACKOFF: Duration = Duration::from_millis(750);

    // Force-stop the named VM, bounded, to clear partial/transitional state
    // before a retry (also drains a still-running previous owner).
    async fn force_stop(name: &str) {
        if let Ok(h) = MsbSandbox::get(name).await {
            let _ = tokio::time::timeout(PER_TRY, h.stop()).await;
        }
    }

    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        match tokio::time::timeout(PER_TRY, MsbSandbox::start_detached(name)).await {
            Ok(Ok(s)) => return Ok(s),
            Ok(Err(MicrosandboxError::SandboxStillRunning(_))) => {
                force_stop(name).await;
            }
            Ok(Err(e)) if is_transient_msb_error(&e.to_string()) => {
                last = e.to_string();
                log::warn!(
                    "sandbox '{name}': transient start failure (attempt {attempt}/{ATTEMPTS}): {e}; \
                     forcing stop and retrying"
                );
                force_stop(name).await;
            }
            Ok(Err(e)) => return Err(anyhow::anyhow!("sandbox start: {e}")),
            Err(_elapsed) => {
                last = format!("start_detached timed out after {}s", PER_TRY.as_secs());
                log::warn!(
                    "sandbox '{name}': start_detached timed out after {}s \
                     (attempt {attempt}/{ATTEMPTS}); likely microsandbox SQLite \
                     contention — forcing stop and retrying",
                    PER_TRY.as_secs()
                );
                force_stop(name).await;
            }
        }
        if attempt < ATTEMPTS {
            tokio::time::sleep(BACKOFF).await;
        }
    }
    anyhow::bail!(
        "sandbox '{name}': failed to start after {ATTEMPTS} bounded attempts \
         (last: {last}; microsandbox appears wedged under rapid churn)"
    )
}

/// Start the sandbox (bounded + retrying on the SQLite hang and on
/// `SandboxStillRunning`), then detach lifecycle ownership and reconnect without
/// it, so dropping the returned handle will not SIGTERM the VM. SIGTERM bypasses
/// agentd's stop path and loses in-flight dirty pages; all stops must go through
/// `Sandbox::stop`.
async fn start_and_connect(name: &str) -> anyhow::Result<MsbSandbox> {
    let inner = start_detached_resilient(name).await?;
    inner.detach().await;
    MsbSandbox::get(name)
        .await
        .map_err(|e| anyhow::anyhow!("start: sandbox not found: {e}"))?
        .connect()
        .await
        .map_err(|e| anyhow::anyhow!("start: connect failed: {e}"))
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

async fn create_registered(name: &str, config: &SandboxConfig) -> anyhow::Result<MsbSandbox> {
    const ATTEMPTS: usize = 3;
    // `.create()` registers AND boots the VM (waiting for the agent relay), so it
    // is subject to the same transient startup hang/race as a reconnect. This is
    // only the fresh-create path (the sandbox does not exist yet), so cleaning up
    // and retrying from scratch is safe — there is no prior state to preserve.
    const CREATE_TIMEOUT: Duration = Duration::from_secs(45);
    const BACKOFF: Duration = Duration::from_millis(750);

    let mut last = String::new();
    for attempt in 1..=ATTEMPTS {
        let mut builder = MsbSandbox::builder(name)
            .image(config.image.as_str())
            .cpus(config.cpus)
            .memory(config.memory_mib)
            .pull_policy(PullPolicy::IfMissing)
            .detached(true);
        for (k, v) in &config.env {
            builder = builder.env(k.as_str(), v.as_str());
        }
        builder = config.network.apply(builder);
        for mount in &config.volumes {
            builder = apply_volume_mount(builder, mount);
        }

        match tokio::time::timeout(CREATE_TIMEOUT, builder.create()).await {
            Ok(Ok(sb)) => return Ok(sb),
            // A genuine error (bad image, invalid config) — surface immediately.
            Ok(Err(e)) if !is_transient_msb_error(&e.to_string()) => {
                return Err(anyhow::anyhow!("sandbox create: {e}"));
            }
            Ok(Err(e)) => {
                last = e.to_string();
                log::warn!(
                    "sandbox '{name}': transient create failure (attempt {attempt}/{ATTEMPTS}): {e}; \
                     cleaning up and retrying"
                );
            }
            Err(_elapsed) => {
                last = format!("create timed out after {}s", CREATE_TIMEOUT.as_secs());
                log::warn!(
                    "sandbox '{name}': create timed out after {}s (attempt {attempt}/{ATTEMPTS}); \
                     likely microsandbox SQLite contention / relay-wait — cleaning up and retrying",
                    CREATE_TIMEOUT.as_secs()
                );
            }
        }
        // Tear down any partial registration before retrying from scratch.
        Sandbox::remove(name).await;
        if attempt < ATTEMPTS {
            tokio::time::sleep(BACKOFF).await;
        }
    }
    anyhow::bail!(
        "sandbox '{name}': create did not complete after {ATTEMPTS} bounded attempts \
         (last: {last}; microsandbox appears wedged under rapid churn)"
    )
}

async fn create_from_snapshot(
    new_name: &str,
    snap_name: &str,
    config: &SandboxConfig,
) -> anyhow::Result<MsbSandbox> {
    let mut builder = MsbSandbox::builder(new_name)
        .from_snapshot(snap_name)
        .cpus(config.cpus)
        .memory(config.memory_mib)
        .detached(true);

    for (k, v) in &config.env {
        builder = builder.env(k.as_str(), v.as_str());
    }
    builder = config.network.apply(builder);
    for mount in &config.volumes {
        builder = apply_volume_mount(builder, mount);
    }

    let sb = builder
        .create()
        .await
        .map_err(|e| anyhow::anyhow!("fork: create from snapshot: {e}"))?;

    // Ensure workdir exists. A no-op when the source already had it, but
    // required when forking a sandbox whose workdir wasn't in the source.
    let _ = sb.shell(&format!("mkdir -p {}", config.workdir)).await;

    sb.stop()
        .await
        .map_err(|e| anyhow::anyhow!("fork: stop new sandbox: {e}"))?;

    Ok(sb)
}

async fn vm_is_running(name: &str) -> bool {
    MsbSandbox::get(name)
        .await
        .map(|h| {
            matches!(
                h.status_snapshot(),
                SandboxStatus::Running | SandboxStatus::Draining
            )
        })
        .unwrap_or(false)
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
            create_if_missing,
            labels,
        } => {
            let name = name.clone();
            let ro = *readonly;
            let ensure = *create_if_missing;
            let labels = labels.clone();
            builder.volume(guest, move |m| {
                let m = if ensure {
                    m.named_with(name, move |mut n| {
                        n = n.ensure_exists();
                        for (k, v) in labels {
                            n = n.label(k, v);
                        }
                        n
                    })
                } else {
                    m.named(name)
                };
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

    /// Wait until `vm_is_running(name)` returns false, polling every 50ms.
    /// Returns true if the VM stopped before `deadline`, false on timeout.
    /// Used after handle drop, where the background stop thread stops
    /// the VM asynchronously and `drop` itself returns immediately.
    async fn wait_until_stopped(name: &str, deadline: std::time::Duration) -> bool {
        let start = Instant::now();
        loop {
            if !vm_is_running(name).await {
                return true;
            }
            if start.elapsed() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    // ── config & validation ──────────────────────────────────────────────────

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

    /// Names beyond the upstream 128 UTF-8 byte limit are rejected
    /// synchronously by upstream `microsandbox::validate_sandbox_name`.
    #[test]
    fn test_name_too_long_is_rejected() {
        let long_name = "a".repeat(129);
        let result = validate_sandbox_name(&long_name);
        assert!(result.is_err(), "expected Err for a 129-byte name");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("too long") || msg.contains("128"),
            "error message should explain the byte limit, got: {msg}"
        );
    }

    /// Names up to 128 UTF-8 bytes pass syntactic validation; the runtime
    /// may still reject longer-derived paths beyond that, but validation
    /// itself accepts the boundary.
    #[test]
    fn test_name_at_128_bytes_passes_validation() {
        let name = "a".repeat(128);
        assert!(validate_sandbox_name(&name).is_ok());
    }

    // ── exec ─────────────────────────────────────────────────────────────────

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

    /// A failing command propagates its non-zero exit code; `timed_out` stays false.
    #[tokio::test]
    async fn test_exec_nonzero_exit_code() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        let result = handle
            .exec(
                "sh".to_string(),
                vec!["-c".to_string(), "exit 42".to_string()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 42, "expected exit code 42");
        assert!(!result.timed_out, "should not be marked as timed out");
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

    /// Output exceeding `max_output_chars` is middle-truncated with an omission notice.
    #[tokio::test]
    async fn test_max_output_chars_truncation() {
        let env = RunEnv::sandbox(SandboxConfig {
            max_output_chars: 50,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();
        let handle = env.get().await.unwrap();
        let result = handle
            .exec_shell("printf '%200s' | tr ' ' x".to_string(), None)
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
        assert!(
            result.stdout.contains("characters omitted"),
            "expected truncation notice in stdout, got: {:?}",
            result.stdout
        );
    }

    // ── VM lifecycle ─────────────────────────────────────────────────────────

    /// VM definition survives `Machine::stop` (handle drop) when
    /// `persist = false` — subsequent starts on the same `Sandbox` must succeed.
    /// The definition is only removed when the `Sandbox` struct itself is dropped.
    #[tokio::test]
    async fn test_vm_definition_survives_machine_stop_with_persist_false() {
        let name = format!("at-vmdef-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: false,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        // 1st start/stop cycle
        {
            let h = env.get().await.unwrap();
            let r = h
                .exec_shell("echo a > ./m.txt".to_string(), None)
                .await
                .unwrap();
            assert_eq!(r.exit_code, 0);
        } // handle drop → Machine::stop is called

        // VM definition must still exist
        assert!(
            MsbSandbox::get(&name).await.is_ok(),
            "VM definition must survive handle drop when persist=false"
        );

        // 2nd start must succeed and file state must be preserved
        let h2 = env.get().await.unwrap();
        let r = h2
            .exec_shell("cat ./m.txt".to_string(), None)
            .await
            .unwrap();
        assert_eq!(r.exit_code, 0, "second start must succeed: {}", r.stderr);
        drop(h2);

        // VM definition must be gone after the Sandbox is dropped
        drop(env);
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        assert!(
            MsbSandbox::get(&name).await.is_err(),
            "VM definition must be removed when persist=false Sandbox is dropped"
        );
    }

    /// A single start serves multiple `exec` calls without the VM going down
    /// between them.
    #[tokio::test]
    async fn test_vm_stays_up_across_exec_calls() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        {
            let r = handle
                .exec(
                    "sh".to_string(),
                    vec!["-c".to_string(), "echo first > ./marker".to_string()],
                    None,
                )
                .await
                .unwrap();
            assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        }
        {
            let r = handle
                .exec("cat".to_string(), vec!["./marker".to_string()], None)
                .await
                .unwrap();
            assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
            assert!(r.stdout.contains("first"));
        }
    }

    /// Dropping the last handle stops the VM; `get()` starts it again.
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
            wait_until_stopped(&name, std::time::Duration::from_secs(30)).await,
            "VM should be stopped within 30s of handle drop"
        );

        let handle = env.get().await.unwrap();
        assert!(vm_is_running(&name).await, "VM should be running again");
        drop(handle);
        Sandbox::remove_persisted(&name).await.ok();
    }

    /// Repeated start/drop cycles are safe with `persist = false` — the VM
    /// definition survives each cycle and no resource leak or double-stop error
    /// occurs. Removal happens only when the `Sandbox` is dropped.
    #[tokio::test]
    async fn test_start_stop_idempotent() {
        let name = format!("at-si-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: false,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        for _ in 0..3 {
            let handle = env.get().await.unwrap();
            assert!(vm_is_running(&name).await);
            drop(handle);
            assert!(
                wait_until_stopped(&name, std::time::Duration::from_secs(30)).await,
                "VM should be stopped within 30s of handle drop"
            );
        }
        // VM definition is still present — removal happens on Sandbox drop.
        assert!(MsbSandbox::get(&name).await.is_ok());
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
        Sandbox::remove_persisted(&name).await.ok();
    }

    // ── exists ───────────────────────────────────────────────────────────────

    /// A randomly generated name that was never registered returns `false`.
    #[tokio::test]
    async fn test_exists_returns_false_for_unknown_name() {
        let name = format!("at-ex-unknown-{}", short_id());
        assert!(
            !Sandbox::exists(&name).await,
            "a never-registered name must not exist"
        );
    }

    /// `exists()` tracks the full lifecycle: `true` after creation, `false` after removal.
    #[tokio::test]
    async fn test_exists_lifecycle() {
        let name = format!("at-ex-{}", short_id());
        let _env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        assert!(Sandbox::exists(&name).await, "must exist after creation");
        Sandbox::remove_persisted(&name)
            .await
            .expect("remove failed");
        assert!(
            !Sandbox::exists(&name).await,
            "must not exist after removal"
        );
    }

    // ── concurrent access ────────────────────────────────────────────────────

    /// Concurrent `RunEnv::get()` calls all succeed and share the same started VM —
    /// the machine is started at most once.
    #[tokio::test]
    async fn test_concurrent_get() {
        let name = format!("at-cg-{}", short_id());
        let env = RunEnv::sandbox(SandboxConfig {
            name: Some(name.clone()),
            persist: true,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();

        let tasks: Vec<_> = (0..4)
            .map(|_| {
                let e = env.clone();
                tokio::spawn(async move { e.get().await })
            })
            .collect();

        let mut handles = Vec::new();
        for task in tasks {
            handles.push(task.await.expect("task panic").expect("get() failed"));
        }

        assert!(
            vm_is_running(&name).await,
            "VM should be running while handles are held"
        );
        for h in &handles {
            let r = h.exec_shell("echo ok".to_string(), None).await.unwrap();
            assert_eq!(r.exit_code, 0);
        }
        drop(handles);
        Sandbox::remove_persisted(&name).await.ok();
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
            .write(Path::new("./marker.txt"), b"sequential")
            .await
            .expect("write failed");

        for i in 0..10 {
            let bytes = handle
                .read(Path::new("./marker.txt"))
                .await
                .unwrap_or_else(|e| panic!("read failed on iteration {i}: {e}"));
            let content = String::from_utf8_lossy(&bytes);
            assert!(
                content.contains("sequential"),
                "iteration {i}: expected 'sequential' in content, got: {content:?}"
            );
        }
    }

    // ── file I/O ─────────────────────────────────────────────────────────────

    /// `get_cwd()` returns the sandbox workdir (default `/workspace`).
    #[tokio::test]
    async fn test_get_cwd() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        let cwd = handle.get_cwd().await.expect("get_cwd failed");
        assert_eq!(
            cwd,
            PathBuf::from("/workspace"),
            "default workdir should be /workspace, got: {cwd:?}"
        );
    }

    /// When `workdir` is customized, exec and relative read/write all resolve
    /// paths against the new directory.
    #[tokio::test]
    async fn test_custom_workdir() {
        let env = RunEnv::sandbox(SandboxConfig {
            workdir: "/custom_work".to_string(),
            ..SandboxConfig::default()
        })
        .await
        .unwrap();
        let handle = env.get().await.unwrap();

        let r = handle
            .exec_shell("pwd".to_string(), None)
            .await
            .expect("pwd failed");
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(
            r.stdout.contains("/custom_work"),
            "cwd should be /custom_work, got: {:?}",
            r.stdout
        );

        handle
            .write(Path::new("./cw.txt"), b"custom_work_ok")
            .await
            .expect("write failed");
        let bytes = handle
            .read(Path::new("./cw.txt"))
            .await
            .expect("read failed");
        assert!(
            String::from_utf8_lossy(&bytes).contains("custom_work_ok"),
            "file written relative to custom workdir must be readable"
        );
    }

    /// Writing to a path whose parent directories do not yet exist creates them
    /// automatically via `mkdir -p`.
    #[tokio::test]
    async fn test_write_nested_dir_creates_parents() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        handle
            .write(Path::new("./deep/nested/dir/data.txt"), b"nested_ok")
            .await
            .expect("write to nested path failed");
        let bytes = handle
            .read(Path::new("./deep/nested/dir/data.txt"))
            .await
            .expect("read back failed");
        assert!(
            String::from_utf8_lossy(&bytes).contains("nested_ok"),
            "nested file content mismatch"
        );
    }

    /// Reading a path that does not exist returns `Err`, not an empty `Ok`.
    #[tokio::test]
    async fn test_read_nonexistent_returns_error() {
        let env = make_env().await;
        let handle = env.get().await.unwrap();
        let result = handle
            .read(Path::new("./this_file_does_not_exist_xyz.txt"))
            .await;
        assert!(
            result.is_err(),
            "reading a non-existent file should return Err"
        );
    }

    // ── persistence ──────────────────────────────────────────────────────────

    /// Files written during one start survive the next — the VM definition is
    /// preserved across stop/start cycles as long as the `Sandbox` is alive,
    /// regardless of `persist`.
    #[tokio::test]
    async fn test_filesystem_persists_across_stop_start() {
        let env = make_env().await;

        {
            let handle = env.get().await.unwrap();
            handle
                .exec_shell("echo hello > ./test.txt".to_string(), None)
                .await
                .expect("write failed");
        }

        let handle = env.get().await.unwrap();
        let bytes = handle
            .read(Path::new("./test.txt"))
            .await
            .expect("read failed");
        let content = String::from_utf8_lossy(&bytes);
        assert!(
            content.contains("hello"),
            "file should survive stop/start cycle, got: {content:?}"
        );
    }

    #[tokio::test]
    async fn test_remove_persisted_idempotent() {
        let result = Sandbox::remove_persisted("ailoy-nonexistent-sandbox-xyz-12345").await;
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
                .write(Path::new("./marker.txt"), b"persist_marker")
                .await
                .expect("write failed");
        }

        Sandbox::remove_persisted(&name)
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
        let result = handle.read(Path::new("./marker.txt")).await;
        assert!(
            result.is_err(),
            "fresh VM should not contain the marker file from the previous run"
        );
        drop(handle);
        Sandbox::remove_persisted(&name).await.ok();
    }

    // ── volume mounts ────────────────────────────────────────────────────────

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
                    create_if_missing: false,
                    labels: Vec::new(),
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

        // Second mount opts into `create_if_missing` to also exercise the
        // `ensure_exists` reuse path against the volume the first mount used.
        let read_result = async {
            let env = RunEnv::sandbox(SandboxConfig {
                volumes: vec![VolumeMount::Named {
                    name: vol_name.clone(),
                    guest: "/mnt/vol".to_string(),
                    readonly: false,
                    create_if_missing: true,
                    labels: Vec::new(),
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

    /// `/tmp` is backed by an auto-injected named volume, so writes from one
    /// handle scope survive into the next within the same `Sandbox`.
    #[tokio::test]
    async fn test_tmp_persists_across_starts() {
        let env = make_env().await;

        {
            let h = env.get().await.unwrap();
            h.exec_shell("echo hello > /tmp/probe".to_string(), None)
                .await
                .expect("write failed");
        }

        let h = env.get().await.unwrap();
        let r = h
            .exec_shell(
                "cat /tmp/probe 2>/dev/null || echo MISSING".to_string(),
                None,
            )
            .await
            .expect("read failed");
        assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
        assert!(r.stdout.contains("hello"), "stdout: {:?}", r.stdout);
    }

    /// User-supplied `/tmp` mounts take precedence over the auto-injected
    /// named volume, so callers can opt out and get tmpfs semantics back.
    #[tokio::test]
    async fn test_user_tmpfs_overrides_default_tmp() {
        let env = RunEnv::sandbox(SandboxConfig {
            volumes: vec![VolumeMount::Tmpfs {
                guest: "/tmp".to_string(),
                size_mib: Some(64),
            }],
            ..SandboxConfig::default()
        })
        .await
        .expect("failed to create sandbox");

        {
            let h = env.get().await.unwrap();
            h.exec_shell("echo hello > /tmp/probe".to_string(), None)
                .await
                .expect("write failed");
        }

        let h = env.get().await.unwrap();
        let r = h
            .exec_shell(
                "cat /tmp/probe 2>/dev/null || echo MISSING".to_string(),
                None,
            )
            .await
            .expect("read failed");
        assert!(
            r.stdout.contains("MISSING"),
            "user tmpfs at /tmp must wipe between starts, stdout: {:?}",
            r.stdout
        );
    }

    /// Dropping a non-persist `Sandbox` removes its auto-injected `/tmp`
    /// volume along with the VM definition.
    #[tokio::test]
    async fn test_tmp_volume_removed_on_sandbox_drop() {
        use microsandbox::Volume;

        let name = format!("drop-test-{}", short_id());
        let vol = tmp_volume_name(&name);
        {
            let _env = RunEnv::sandbox(SandboxConfig {
                name: Some(name.clone()),
                persist: false,
                ..SandboxConfig::default()
            })
            .await
            .expect("failed to create sandbox");
            Volume::get(&vol)
                .await
                .expect("volume must exist while sandbox is alive");
        }

        // Drop runs cleanup on a background thread; poll until the volume
        // disappears or the deadline elapses.
        let start = Instant::now();
        loop {
            if Volume::get(&vol).await.is_err() {
                return;
            }
            if start.elapsed() >= std::time::Duration::from_secs(30) {
                panic!("volume '{vol}' was not removed within 30s of Sandbox drop");
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }

    /// The startup sweep removes `ailoy.managed` volumes whose owner sandbox
    /// is no longer present. Volumes without the label are left alone.
    #[tokio::test]
    async fn test_startup_sweep_targets_only_labelled_orphans() {
        use microsandbox::Volume;

        // Orphan: ailoy-tmp-* prefix + ailoy.managed label, no owner sandbox.
        let orphan_owner = format!("nonexistent-{}", Uuid::new_v4());
        let orphan_vol = tmp_volume_name(&orphan_owner);
        Volume::builder(&orphan_vol)
            .label(AILOY_VOLUME_LABEL_KEY, AILOY_VOLUME_LABEL_VALUE)
            .create()
            .await
            .expect("create orphan volume");

        // Bystander: ailoy-tmp-* prefix but no label, mimicking an external
        // volume coincidentally named like ours.
        let bystander_vol = format!("ailoy-tmp-external-{}", Uuid::new_v4());
        Volume::builder(&bystander_vol)
            .create()
            .await
            .expect("create bystander volume");

        sweep_orphan_tmp_volumes().await;

        let orphan_present = Volume::get(&orphan_vol).await.is_ok();
        let bystander_present = Volume::get(&bystander_vol).await.is_ok();

        if bystander_present {
            let _ = Volume::remove(&bystander_vol).await;
        }
        if orphan_present {
            let _ = Volume::remove(&orphan_vol).await;
        }

        assert!(!orphan_present, "labelled orphan must be swept");
        assert!(bystander_present, "unlabelled volume must be left alone");
    }

    // ── networking ───────────────────────────────────────────────────────────

    /// By default the sandbox can reach external hosts.
    ///
    /// Uses alpine (busybox `nc`) to avoid needing bash for the /dev/tcp trick.
    #[tokio::test]
    async fn test_network_reachable_by_default() {
        let env = RunEnv::sandbox(SandboxConfig {
            image: "alpine:latest".to_string(),
            ..SandboxConfig::default()
        })
        .await
        .unwrap();
        let handle = env.get().await.unwrap();
        // nc -zw5: zero-I/O scan mode, 5-second timeout. No raw-socket privilege needed.
        let result = handle
            .exec_shell("nc -zw5 8.8.8.8 53".to_string(), Some(10))
            .await
            .unwrap();
        assert_eq!(
            result.exit_code, 0,
            "TCP connect to 8.8.8.8:53 should succeed with default network settings, \
             stderr: {}",
            result.stderr
        );
    }

    /// `SandboxNetwork::HostOnly` grants egress to the sandbox host only; public
    /// egress stays blocked. Confirms the least-privilege VFS mode really does
    /// keep the guest shell off the internet.
    #[tokio::test]
    async fn test_host_only_network_blocks_public_egress() {
        let env = RunEnv::sandbox(SandboxConfig {
            image: "alpine:latest".to_string(),
            network: SandboxNetwork::HostOnly,
            ..SandboxConfig::default()
        })
        .await
        .unwrap();
        let handle = env.get().await.unwrap();
        // Probe a non-DNS port: microsandbox intercepts port 53 with its own
        // resolver, so :53 "connects" regardless of egress policy. :443 to a raw
        // public IP is a real egress test (no DNS lookup needed).
        let result = handle
            .exec_shell("nc -zw5 1.1.1.1 443 2>/dev/null".to_string(), Some(10))
            .await
            .unwrap();
        assert_ne!(
            result.exit_code, 0,
            "public TCP connect (1.1.1.1:443) must be blocked under SandboxNetwork::HostOnly"
        );
    }

    // ── fork ─────────────────────────────────────────────────────────────────
    //
    // Fork operates on the underlying `Sandbox`, not on a started handle, so
    // these tests bypass `RunEnv` and drive `Machine::start` / `Machine::stop`
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
        let _console = src.start().await.expect("start");
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
        src.stop().await;
        Sandbox::remove_persisted(&src_name).await.ok();
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
            let console = src.start().await.expect("start src");
            console
                .exec_shell("echo fork_content > ./note.txt".to_string(), None)
                .await
                .expect("write note");
        }
        src.stop().await;

        let mut child = src
            .fork(SandboxConfig {
                name: Some(child_name.clone()),
                persist: true,
                ..SandboxConfig::default()
            })
            .await
            .expect("fork");

        let child_console = child.start().await.expect("start child");
        let bytes = child_console
            .read(Path::new("./note.txt"))
            .await
            .expect("read from child");
        let content = String::from_utf8_lossy(&bytes);

        assert!(
            content.contains("fork_content"),
            "child should have note from source, got: {content:?}"
        );

        drop(child_console);
        child.stop().await;
        Sandbox::remove_persisted(&src_name).await.ok();
        Sandbox::remove_persisted(&child_name).await.ok();
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
            let console = src.start().await.expect("start src");
            console
                .exec_shell("echo original > ./data.txt".to_string(), None)
                .await
                .expect("write original");
        }
        src.stop().await;

        let mut child = src
            .fork(SandboxConfig {
                name: Some(child_name.clone()),
                persist: true,
                ..SandboxConfig::default()
            })
            .await
            .expect("fork");

        {
            let console = child.start().await.expect("start child");
            console
                .exec_shell("echo mutated > ./data.txt".to_string(), None)
                .await
                .expect("mutate child");
        }
        child.stop().await;

        let src_console = src.start().await.expect("start src after fork");
        let bytes = src_console
            .read(Path::new("./data.txt"))
            .await
            .expect("read source after child mutation");
        let src_content = String::from_utf8_lossy(&bytes);

        assert!(
            src_content.contains("original") && !src_content.contains("mutated"),
            "source must not be affected by child write, got: {src_content:?}"
        );

        drop(src_console);
        src.stop().await;
        Sandbox::remove_persisted(&src_name).await.ok();
        Sandbox::remove_persisted(&child_name).await.ok();
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

        let snap_dir = microsandbox::config::load_persisted_config_or_default()
            .unwrap()
            .home()
            .join("snapshots")
            .join(&snap_name);

        assert!(
            !snap_dir.exists(),
            "snapshot should be deleted after fork, still found at {snap_dir:?}"
        );

        child.stop().await;
        Sandbox::remove_persisted(&src_name).await.ok();
        Sandbox::remove_persisted(&child_name).await.ok();
    }
}
