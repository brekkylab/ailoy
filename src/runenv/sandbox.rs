//! Microsandbox-backed [`Machine`] implementation.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Context as _;
use async_trait::async_trait;
use microsandbox::{
    ExecOutput, MicrosandboxError, NetworkPolicy, Sandbox as MsbSandbox, SandboxConfig, Snapshot,
    sandbox::{ExecOptionsBuilder, IntoImage, MountBuilder, PullPolicy, validate_sandbox_name},
    snapshot::SaveOpts,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    runenv::{Console, ExecResult, Machine},
    util::truncate::middle_truncate,
};

/// A volume mount attached to a sandbox at creation time.
///
/// Narrow surface over microsandbox's mount model: only the three variants
/// agents need (`Bind`, `Named`, `Tmpfs`), with `readonly` as the only policy
/// knob. Lower-level options (stat virtualization, host permission
/// propagation, disk-image mounts) stay inside the sandbox module.
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
    /// or `$MSB_HOME/volumes/<name>/`). The volume persists across sandbox
    /// restarts and can be shared between sandboxes.
    Named {
        /// Name of the microsandbox volume.
        name: String,
        /// Absolute guest path.
        guest: String,
        /// When `true`, the guest cannot write to this mount.
        #[serde(default)]
        readonly: bool,
        /// Create the volume if missing (reusing a compatible existing one)
        /// instead of requiring it to already exist. Default: `false`.
        #[serde(default)]
        create_if_missing: bool,
        /// Labels applied when creating the volume; must match the existing
        /// volume's labels when reused.
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
    /// Attach a custom virtio-fs backend (e.g. an S3-backed workspace) resolved
    /// on the sandbox side by a registered factory.
    ///
    /// Requires the sandbox process (`MSB_PATH`) to be a binary that registered
    /// `backend_type` — e.g. cortex's `msb_cortex`, which registers
    /// `"cortex-s3"`. The backend is attached under `tag`; guest-side mounting
    /// of that tag at `guest` is not yet wired through agentd, so a tool-call
    /// script must `mount -t virtiofs <tag> <guest>` for now.
    FsBackend {
        /// virtio-fs device tag the guest mounts.
        tag: String,
        /// Absolute guest path the tag is intended to be mounted at.
        guest: String,
        /// Registered factory name (e.g. `"cortex-s3"`).
        backend_type: String,
        /// Opaque, factory-specific parameters (typically JSON).
        params: String,
    },
}

impl VolumeMount {
    /// Guest mount path. Matches the `guest` field across all variants.
    pub fn guest_path(&self) -> &str {
        match self {
            VolumeMount::Bind { guest, .. }
            | VolumeMount::Named { guest, .. }
            | VolumeMount::Tmpfs { guest, .. }
            | VolumeMount::FsBackend { guest, .. } => guest,
        }
    }
}

/// Resolve `MSB_HOME` (or `$HOME/.microsandbox` if unset). Sync so `Drop`
/// can reuse it without spinning up a runtime.
fn msb_home() -> anyhow::Result<PathBuf> {
    std::env::var_os("MSB_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".microsandbox")))
        .ok_or_else(|| anyhow::anyhow!("MSB_HOME and HOME both unset"))
}

/// Run `msb <args>...` synchronously, silencing stdout/stderr.
/// Returns `Err` on spawn failure or non-zero exit. Used from Drop impls.
fn run_msb_cli<I, S>(args: I) -> anyhow::Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let bin = msb_home()?.join("bin").join("msb");
    let status = std::process::Command::new(&bin)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .with_context(|| format!("spawn {}", bin.display()))?;
    if !status.success() {
        anyhow::bail!("`msb` exited with {status}");
    }
    Ok(())
}

async fn ensure_msb() -> anyhow::Result<PathBuf> {
    let home = msb_home()?;
    if !microsandbox::setup::is_installed() {
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
    Ok(home)
}

#[derive(Debug, Clone)]
pub struct SandboxBuilder {
    /// All microsandbox-native settings (name, image, cpus, memory, workdir,
    /// env, mounts, network, pull policy, ...) live here directly.
    config: SandboxConfig,

    /// Per-exec timeout in seconds. Default: `60`.
    default_timeout_secs: u64,

    /// Maximum characters to keep from stdout/stderr. Default: `30_000`.
    max_output_chars: usize,

    /// First error captured by a setter (currently only `image()`), surfaced
    /// at `build()` time. Mirrors microsandbox::SandboxBuilder's pattern so
    /// the setters can stay infallible and chainable. Stored as `String` so
    /// the builder remains `Clone`.
    build_error: Option<String>,
}

/// Convert an engine-side [`NetworkPolicy`] into the wire-format policy the
/// `SandboxSpec` stores. The two types share one JSON schema (microsandbox
/// converts its engine config into the spec through the same serde
/// round-trip), so the conversion is infallible.
fn wire_network_policy(policy: NetworkPolicy) -> microsandbox_types::NetworkPolicy {
    let value = serde_json::to_value(policy).expect("NetworkPolicy serializes to JSON");
    serde_json::from_value(value).expect("engine and wire NetworkPolicy share one JSON schema")
}

/// Guest network reachability. The sandbox host
/// (`host.microsandbox.internal`, e.g. an ailoy VFS forward server) is
/// reachable in **every** variant; they differ only in outside reach. There is
/// no fully-offline variant. All map to a [`NetworkPolicy`] with
/// `default_ingress: Allow` (published-port behavior).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SandboxNetwork {
    /// Host only — no public internet, LAN, loopback, or metadata.
    /// Least-privilege for VFS-in-sandbox.
    HostOnly,
    /// Host + public internet, but not private LAN, loopback, link-local,
    /// cloud-metadata, or multicast. The default.
    #[default]
    Public,
    /// Unrestricted egress/ingress — `Public` plus LAN, loopback, link-local,
    /// cloud-metadata, and multicast. Grant deliberately (reopens SSRF).
    Full,
}

impl SandboxNetwork {
    /// The microsandbox policy for this variant.
    fn policy(self) -> NetworkPolicy {
        use microsandbox_network::policy::{Action, Destination, DestinationGroup, Rule};
        // `allow_egress(Host)` permits any port to the host — including :53, so
        // the guest's `host.microsandbox.internal` DNS lookup works.
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

impl Default for SandboxBuilder {
    fn default() -> Self {
        let mut config = SandboxConfig::default();
        // 8 random bytes hex-encoded = 16 hex chars, ~64 bits of entropy. Short
        // enough to fit any reasonable socket-path budget.
        config.spec.name = format!("ailoy-{}", hex::encode(&Uuid::new_v4().as_bytes()[..8]));
        // `"ubuntu:latest"` is a stable OCI reference; conversion never fails.
        config.spec.image = "ubuntu:latest"
            .into_rootfs_source()
            .expect("'ubuntu:latest' parses as an OCI image reference");
        config.spec.resources.cpus = 2;
        config.spec.resources.memory_mib = 2048;
        config.spec.runtime.workdir = Some("/root".to_string());
        config.spec.pull_policy = PullPolicy::IfMissing;
        // Default network posture: host + public internet (see SandboxNetwork).
        config.spec.network.enabled = true;
        config.spec.network.policy = Some(wire_network_policy(SandboxNetwork::default().policy()));
        Self {
            config,
            default_timeout_secs: 60,
            max_output_chars: 30_000,
            build_error: None,
        }
    }
}

impl SandboxBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.spec.name = name.into();
        self
    }

    pub fn image(mut self, image: impl IntoImage) -> Self {
        match image.into_rootfs_source() {
            Ok(rfs) => self.config.spec.image = rfs,
            Err(e) => {
                if self.build_error.is_none() {
                    self.build_error = Some(format!("invalid image: {e}"));
                }
            }
        }
        self
    }

    pub fn cpus(mut self, cpus: u8) -> Self {
        self.config.spec.resources.cpus = cpus;
        self
    }

    pub fn memory_mib(mut self, memory_mib: u32) -> Self {
        self.config.spec.resources.memory_mib = memory_mib;
        self
    }

    pub fn workdir(mut self, workdir: impl Into<String>) -> Self {
        self.config.spec.runtime.workdir = Some(workdir.into());
        self
    }

    pub fn env(mut self, env: impl IntoIterator<Item = (String, String)>) -> Self {
        self.config.spec.env = env.into_iter().map(Into::into).collect();
        self
    }

    /// Set the guest network posture. The host is reachable in every variant;
    /// see [`SandboxNetwork`]. Defaults to [`SandboxNetwork::Public`].
    pub fn network(mut self, network: SandboxNetwork) -> Self {
        self.config.spec.network.enabled = true;
        self.config.spec.network.policy = Some(wire_network_policy(network.policy()));
        self
    }

    /// Append a volume mount.
    pub fn mount(mut self, mount: VolumeMount) -> Self {
        // Custom fs-backends have no host path, so they bypass `MountBuilder`
        // and ride the SDK's separate fs-backend channel into the launch config.
        if let VolumeMount::FsBackend {
            tag,
            guest: _,
            backend_type,
            params,
        } = mount
        {
            self.config.add_fs_backend(tag, backend_type, params);
            return self;
        }
        let builder = match mount {
            VolumeMount::Bind {
                host,
                guest,
                readonly,
            } => {
                let b = MountBuilder::new(guest).bind(host);
                if readonly { b.readonly() } else { b }
            }
            VolumeMount::Named {
                name,
                guest,
                readonly,
                create_if_missing,
                labels,
            } => {
                let b = if create_if_missing {
                    MountBuilder::new(guest).named_with(name, move |mut n| {
                        n = n.ensure_exists();
                        for (k, v) in labels {
                            n = n.label(k, v);
                        }
                        n
                    })
                } else {
                    MountBuilder::new(guest).named(name)
                };
                if readonly { b.readonly() } else { b }
            }
            VolumeMount::Tmpfs { guest, size_mib } => {
                let b = MountBuilder::new(guest).tmpfs();
                if let Some(s) = size_mib { b.size(s) } else { b }
            }
            VolumeMount::FsBackend { .. } => unreachable!("handled before the builder match"),
        };
        match builder.build() {
            Ok(vm) => self.config.spec.mounts.push(vm),
            Err(e) => {
                if self.build_error.is_none() {
                    self.build_error = Some(format!("invalid mount: {e}"));
                }
            }
        }
        self
    }

    pub fn default_timeout_secs(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    pub fn max_output_chars(mut self, chars: usize) -> Self {
        self.max_output_chars = chars;
        self
    }

    pub async fn build(self) -> anyhow::Result<Sandbox> {
        if let Some(e) = self.build_error {
            anyhow::bail!(e);
        }
        ensure_msb().await?;
        let Self {
            config,
            default_timeout_secs,
            max_output_chars,
            ..
        } = self;

        validate_sandbox_name(&config.spec.name)
            .map_err(|e| anyhow::anyhow!("sandbox name '{}': {e}", config.spec.name))?;

        Sandbox::try_new(config, default_timeout_secs, max_output_chars).await
    }
}

/// Remove a microsandbox snapshot directory by path. Best-effort: any
/// failure is logged at `warn!` and swallowed so callers can continue.
fn cleanup_snapshot_dir(path: &Path) {
    if let Err(e) = run_msb_cli([
        std::ffi::OsStr::new("snapshot"),
        std::ffi::OsStr::new("remove"),
        std::ffi::OsStr::new("-f"),
        path.as_os_str(),
    ]) {
        log::warn!("cleanup snapshot {}: {e}", path.display());
    }
}

pub struct Sandbox {
    name: String,
    default_timeout_secs: u64,
    max_output_chars: usize,
    console: Option<SandboxConsole>,
}

impl Sandbox {
    pub async fn try_new(
        config: SandboxConfig,
        default_timeout_secs: u64,
        max_output_chars: usize,
    ) -> anyhow::Result<Sandbox> {
        let name = config.spec.name.clone();
        let inner = microsandbox::Sandbox::create(config)
            .await
            .context("sandbox create")?;
        Ok(Self {
            name,
            default_timeout_secs,
            max_output_chars,
            console: Some(SandboxConsole {
                inner,
                default_timeout_secs,
                max_output_chars,
            }),
        })
    }

    /// Restore a sandbox from a `.tar.zst` (or `.tar`) archive previously
    /// produced by [`archive`](Self::archive). The restored sandbox is
    /// created in the stopped state, reusing the embedded snapshot name.
    /// Any existing sandbox with that name is replaced (stopped and removed
    /// first), so restore is idempotent. The intermediate microsandbox
    /// snapshot directory unpacked by this call is cleaned up before
    /// returning (success or failure).
    pub async fn try_from_archive(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        Self::try_from_archive_inner(path, SandboxNetwork::default()).await
    }

    /// Like [`try_from_archive`](Self::try_from_archive) but with an explicit
    /// [`SandboxNetwork`]. An archive is only a filesystem snapshot and doesn't
    /// carry the network policy, so the desired posture is re-applied here.
    pub async fn try_from_archive_with_network(
        path: impl AsRef<Path>,
        network: SandboxNetwork,
    ) -> anyhow::Result<Self> {
        Self::try_from_archive_inner(path, network).await
    }

    async fn try_from_archive_inner(
        path: impl AsRef<Path>,
        network: SandboxNetwork,
    ) -> anyhow::Result<Self> {
        ensure_msb().await?;

        let handle = Snapshot::load(path.as_ref(), None)
            .await
            .context("load snapshot archive")?;
        let snap = handle.open().await.context("open loaded snapshot")?;
        let snap_path = snap.path().to_path_buf();

        // The imported artifact directory is content-addressed (`sha256-…`),
        // so its file name is not the original sandbox name. The manifest's
        // `source_sandbox` records the name the snapshot was taken from; reuse
        // that. Fall back to a fresh generated name for name-less snapshots.
        let name = snap
            .manifest()
            .source_sandbox
            .clone()
            .unwrap_or_else(|| format!("ailoy-{}", hex::encode(&Uuid::new_v4().as_bytes()[..8])));

        let result = microsandbox::Sandbox::builder(&name)
            .from_snapshot(snap_path.to_string_lossy().into_owned())
            .pull_policy(PullPolicy::IfMissing)
            .replace()
            .network(|n| n.policy(network.policy()))
            .create()
            .await
            .context("create sandbox from snapshot");

        cleanup_snapshot_dir(&snap_path);

        let inner = result?;
        Ok(Self {
            name,
            default_timeout_secs: 60,
            max_output_chars: 30_000,
            console: Some(SandboxConsole {
                inner,
                default_timeout_secs: 60,
                max_output_chars: 30_000,
            }),
        })
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Snapshot this sandbox and bundle the result into a `.tar.zst` archive
    /// at `path`. The sandbox must be stopped — call [`Machine::stop`] first,
    /// or rely on the stopped state that [`SandboxBuilder::build`] leaves it
    /// in. The intermediate microsandbox snapshot directory created by this
    /// call is cleaned up before returning (success or failure).
    pub async fn archive(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let snap = Snapshot::builder(&self.name)
            .from_sandbox(&self.name)
            .create()
            .await
            .context("create snapshot")?;
        let snap_path = snap.path().to_path_buf();

        let result = Snapshot::save(
            snap_path.to_string_lossy().as_ref(),
            path.as_ref(),
            SaveOpts::default(),
        )
        .await
        .context("save snapshot archive");

        cleanup_snapshot_dir(&snap_path);

        result?;
        Ok(())
    }

    /// Fork this sandbox into a new one initialized from a filesystem
    /// snapshot of `self`. The new sandbox inherits `default_timeout_secs`
    /// and `max_output_chars` from `self`; the name is auto-generated.
    /// Returns the new sandbox in the running state.
    ///
    /// `self` must be stopped — call [`Machine::stop`] first if needed.
    pub async fn fork(&self) -> anyhow::Result<Sandbox> {
        if self.is_running() {
            anyhow::bail!("cannot fork running sandbox '{}'; stop it first", self.name);
        }

        let new_name = format!("ailoy-{}", hex::encode(&Uuid::new_v4().as_bytes()[..8]));
        let snap_name = format!("fork-{new_name}");

        let handle = MsbSandbox::get(&self.name)
            .await
            .map_err(|e| anyhow::anyhow!("fork: source sandbox not found: {e}"))?;

        let snap = handle
            .snapshot(&snap_name)
            .await
            .map_err(|e| anyhow::anyhow!("fork: snapshot failed: {e}"))?;
        let snap_path = snap.path().to_path_buf();

        let result = microsandbox::Sandbox::builder(&new_name)
            .from_snapshot(snap_path.to_string_lossy().into_owned())
            .pull_policy(PullPolicy::IfMissing)
            .create()
            .await
            .context("fork: create from snapshot");

        // Clean up the temp snapshot regardless of outcome.
        if let Err(e) = Snapshot::remove(&snap_name, true).await {
            log::warn!("fork: failed to clean up snapshot '{snap_name}': {e}");
        }

        let inner = match result {
            Ok(inner) => inner,
            Err(e) => {
                // Best-effort: clean up any partially-created sandbox record.
                let _ = MsbSandbox::remove(&new_name).await;
                return Err(e);
            }
        };

        Ok(Sandbox {
            name: new_name,
            default_timeout_secs: self.default_timeout_secs,
            max_output_chars: self.max_output_chars,
            console: Some(SandboxConsole {
                inner,
                default_timeout_secs: self.default_timeout_secs,
                max_output_chars: self.max_output_chars,
            }),
        })
    }

    /// Returns `true` if a sandbox with the given name already exists, without
    /// creating or starting it.
    ///
    /// Lightweight existence probe — never modifies sandbox state. Returns
    /// `false` on any error (e.g. the microsandbox runtime is not installed).
    pub async fn exists(name: &str) -> bool {
        MsbSandbox::get(name).await.is_ok()
    }

    /// Remove a sandbox by name without holding a [`Sandbox`] instance.
    ///
    /// Intended for explicit cleanup when the [`Sandbox`] object is no longer
    /// available (e.g. after a process restart). Force-removes via the `msb`
    /// CLI in a fresh process, matching how [`Drop`] removes the VM (the
    /// in-process microsandbox DB pool is bound to the parent runtime).
    ///
    /// Idempotent: if the named sandbox does not exist, returns `Ok(())`.
    pub async fn remove_persisted(name: &str) -> anyhow::Result<()> {
        if MsbSandbox::get(name).await.is_err() {
            return Ok(());
        }
        run_msb_cli(["remove", "-f", name])
    }
}

#[async_trait]
impl Machine for Sandbox {
    type Console = SandboxConsole;

    fn is_running(&self) -> bool {
        self.console.is_some()
    }

    async fn start<'a>(&'a mut self) -> anyhow::Result<&'a Self::Console> {
        if self.console.is_none() {
            let inner = microsandbox::Sandbox::start(&self.name)
                .await
                .context("sandbox start")?;
            self.console = Some(SandboxConsole {
                inner,
                default_timeout_secs: self.default_timeout_secs,
                max_output_chars: self.max_output_chars,
            });
        }
        Ok(self.console.as_ref().expect("just set"))
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(console) = self.console.as_ref() {
            console.inner.stop_and_wait().await?;
        }
        self.console = None;
        Ok(())
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // If the VM is still running, gracefully stop it through agentd via
        // the in-process handle before the field drops — otherwise the
        // safety-net SIGTERM costs ~5s of libkrun shutdown. Awaits aren't
        // allowed in Drop, so we hop to a worker thread with a fresh
        // runtime. We can't reuse the in-process API for `remove` because
        // microsandbox's DB pool is bound to the parent runtime; the CLI
        // spawns a fresh process with its own pool.
        if let Some(console) = self.console.take() {
            let name = self.name.clone();
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            std::thread::spawn(move || {
                if let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    && let Err(e) = rt.block_on(console.inner.stop_and_wait())
                {
                    log::warn!("Drop stop_and_wait `{name}`: {e}");
                }
                let _ = tx.send(());
            });
            let _ = rx.recv_timeout(std::time::Duration::from_secs(5));
        }
        if let Err(e) = run_msb_cli(["remove", "-f", self.name.as_str()]) {
            log::warn!("cleanup sandbox `{}`: {e}", self.name);
        }
    }
}

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

#[cfg(test)]
mod tests {
    use microsandbox::sandbox::SandboxStatus;

    use super::*;

    /// Smoke test: a fresh sandbox can run a shell command and return its
    /// stdout/exit code.
    #[tokio::test]
    async fn test_exec() {
        let mut sandbox = SandboxBuilder::new().build().await.expect("build");
        let console = sandbox.start().await.expect("start");
        let r = console
            .exec_shell("echo hello".to_string(), None)
            .await
            .expect("exec");
        assert_eq!(r.exit_code, 0, "non-zero exit: stderr={}", r.stderr);
        assert_eq!(r.stdout.trim(), "hello");
    }

    /// `Machine::stop` should actually transition the underlying microsandbox
    /// VM to the `Stopped` state, not just clear our internal handle.
    #[tokio::test]
    async fn test_stop() {
        let mut sandbox = SandboxBuilder::new().build().await.expect("build");
        let name = sandbox.get_name().to_string();

        let handle = MsbSandbox::get(&name)
            .await
            .expect("vm record should exist after build");
        assert_eq!(
            handle.status_snapshot(),
            SandboxStatus::Running,
            "vm should be Running right after build",
        );

        sandbox.stop().await.expect("stop");

        let handle = MsbSandbox::get(&name)
            .await
            .expect("vm record should still exist after stop");
        assert_eq!(
            handle.status_snapshot(),
            SandboxStatus::Stopped,
            "vm should be Stopped after Machine::stop",
        );
    }

    /// Dropping `Sandbox` should leave nothing behind in microsandbox —
    /// the VM record must be gone so the name can be reused.
    #[tokio::test]
    async fn test_clean_drop() {
        let sandbox = SandboxBuilder::new().build().await.expect("build");
        let name = sandbox.get_name().to_string();

        assert!(
            MsbSandbox::get(&name).await.is_ok(),
            "vm record should exist before drop",
        );

        drop(sandbox);

        assert!(
            MsbSandbox::get(&name).await.is_err(),
            "vm record should be gone after Drop",
        );
    }

    // ── exists / remove_persisted ─────────────────────────────────────────────

    /// A name that was never registered reports `false` without touching state.
    #[tokio::test]
    async fn test_exists_returns_false_for_unknown_name() {
        let name = format!("ailoy-nx-{}", hex::encode(&Uuid::new_v4().as_bytes()[..6]));
        assert!(
            !Sandbox::exists(&name).await,
            "a never-registered name must not exist"
        );
    }

    /// `exists()` tracks the lifecycle: `true` after build, `false` after the
    /// VM is removed on `Drop`.
    #[tokio::test]
    async fn test_exists_true_after_build_false_after_drop() {
        let sandbox = SandboxBuilder::new().build().await.expect("build");
        let name = sandbox.get_name().to_string();

        assert!(Sandbox::exists(&name).await, "must exist after build");

        drop(sandbox);

        assert!(
            !Sandbox::exists(&name).await,
            "must not exist after Drop removes the VM"
        );
    }

    /// `remove_persisted` on an unknown name is a no-op that returns `Ok`.
    #[tokio::test]
    async fn test_remove_persisted_unknown_is_ok() {
        let name = format!("ailoy-nx-{}", hex::encode(&Uuid::new_v4().as_bytes()[..6]));
        Sandbox::remove_persisted(&name)
            .await
            .expect("remove_persisted on unknown name should be Ok");
    }

    // ── network policy ─────────────────────────────────────────────────────────

    /// Probe a raw public IP on :443 from inside the guest. Uses `alpine`
    /// (busybox `nc`) and a non-DNS port — microsandbox intercepts :53 with its
    /// own resolver, so only a raw-IP connect actually exercises egress policy.
    /// Returns the shell exit code (0 = connected).
    async fn public_egress_exit_code(network: SandboxNetwork) -> i32 {
        let mut sandbox = SandboxBuilder::new()
            .image("alpine:latest")
            .network(network)
            .build()
            .await
            .expect("build");
        let console = sandbox.start().await.expect("start");
        console
            .exec_shell("nc -zw5 1.1.1.1 443 2>/dev/null".to_string(), Some(10))
            .await
            .expect("exec")
            .exit_code
    }

    /// `SandboxNetwork::HostOnly` grants egress to the sandbox host only —
    /// public egress stays blocked (the least-privilege VFS posture).
    #[tokio::test]
    async fn test_host_only_blocks_public_egress() {
        assert_ne!(
            public_egress_exit_code(SandboxNetwork::HostOnly).await,
            0,
            "public TCP connect (1.1.1.1:443) must be blocked under HostOnly"
        );
    }

    /// Contrast: `SandboxNetwork::Public` (the default) allows public egress,
    /// confirming HostOnly's block is the policy — not a broken network.
    #[tokio::test]
    async fn test_public_allows_public_egress() {
        assert_eq!(
            public_egress_exit_code(SandboxNetwork::Public).await,
            0,
            "public TCP connect (1.1.1.1:443) must succeed under Public"
        );
    }

    /// Round-trip: build → write files → archive → restore → read files.
    /// Confirms the snapshot artifact is produced, `from_archive` restores
    /// under the same name, and files written before archiving survive.
    #[tokio::test]
    async fn test_archive_and_restore() {
        let mut sandbox = SandboxBuilder::new().build().await.expect("build sandbox");
        let original_name = sandbox.get_name().to_string();

        // Write a couple of files into the rootfs so we can verify they
        // survive the snapshot round-trip.
        {
            let console = sandbox.start().await.expect("start sandbox");
            let r = console
                .exec_shell(
                    "echo hello > /root/file1.txt && echo world > /root/file2.txt".to_string(),
                    None,
                )
                .await
                .expect("write files");
            assert_eq!(r.exit_code, 0, "write failed: {}", r.stderr);
        }

        // Snapshot needs a quiesced VM.
        sandbox.stop().await.expect("stop before archive");

        let tmp = tempfile::tempdir().expect("tempdir");
        let archive_path = tmp.path().join("sandbox.tar.zst");
        sandbox
            .archive(&archive_path)
            .await
            .expect("archive sandbox");
        assert!(
            archive_path.is_file(),
            "archive file is missing: {}",
            archive_path.display()
        );

        let mut restored = Sandbox::try_from_archive(&archive_path)
            .await
            .expect("restore from archive");
        assert_eq!(
            restored.get_name(),
            original_name,
            "restored sandbox should reuse the archive's embedded name"
        );

        // Files written before archiving should still be present.
        let console = restored.start().await.expect("start restored");
        let r1 = console
            .exec_shell("cat /root/file1.txt".to_string(), None)
            .await
            .expect("read file1");
        assert_eq!(r1.exit_code, 0, "cat file1 failed: {}", r1.stderr);
        assert_eq!(r1.stdout.trim(), "hello");

        let r2 = console
            .exec_shell("cat /root/file2.txt".to_string(), None)
            .await
            .expect("read file2");
        assert_eq!(r2.exit_code, 0, "cat file2 failed: {}", r2.stderr);
        assert_eq!(r2.stdout.trim(), "world");
    }

    // 1. fork has a distinct name from source
    // 2. fork starts with source's files (/root/marker.txt: from-source)
    // 3. overwriting in the fork doesn't bleed back into source — isolation
    #[tokio::test]
    async fn test_fork() {
        let mut src = SandboxBuilder::new().build().await.expect("build src");

        // Plant a file in the source, then stop so we can fork.
        {
            let console = src.start().await.expect("start src");
            let r = console
                .exec_shell("echo from-source > /root/marker.txt".to_string(), None)
                .await
                .expect("write marker");
            assert_eq!(r.exit_code, 0, "write failed: {}", r.stderr);
        }
        src.stop().await.expect("stop src");

        let mut fork = src.fork().await.expect("fork");
        assert_ne!(
            fork.get_name(),
            src.get_name(),
            "fork must have a distinct name",
        );

        // The marker written in the source should be visible in the fork.
        let console = fork.start().await.expect("start fork");
        let r = console
            .exec_shell("cat /root/marker.txt".to_string(), None)
            .await
            .expect("read marker in fork");
        assert_eq!(r.exit_code, 0, "cat failed: {}", r.stderr);
        assert_eq!(r.stdout.trim(), "from-source");

        // Mutating the fork must not affect the source.
        let r = console
            .exec_shell("echo from-fork > /root/marker.txt".to_string(), None)
            .await
            .expect("overwrite in fork");
        assert_eq!(r.exit_code, 0, "overwrite failed: {}", r.stderr);
        fork.stop().await.expect("stop fork");

        let console = src.start().await.expect("restart src");
        let r = console
            .exec_shell("cat /root/marker.txt".to_string(), None)
            .await
            .expect("read marker in src");
        assert_eq!(r.exit_code, 0, "cat failed: {}", r.stderr);
        assert_eq!(
            r.stdout.trim(),
            "from-source",
            "source must be unaffected by writes in the fork",
        );
    }
}
