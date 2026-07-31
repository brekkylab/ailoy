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

/// How far outside itself the guest can reach. There is no fully-offline
/// variant; every variant permits DNS to the sandbox gateway so the guest can
/// resolve names at all.
///
/// Reaching a service on the host is a separate grant, expressed per port
/// through [`NetworkPosture::host_ports`]. That split is deliberate: a posture
/// on its own says nothing about which host services are exposed, so widening
/// the outside reach cannot silently widen host reach too.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SandboxNetwork {
    /// Nothing but the host ports explicitly granted, plus gateway DNS. The
    /// default, and the least-privilege posture for VFS-in-sandbox.
    #[default]
    HostOnly,
    /// The public internet, but not private LAN, loopback, link-local,
    /// cloud-metadata, or multicast.
    Public,
    /// Unrestricted egress *and* ingress — `Public` plus LAN, loopback,
    /// link-local, cloud-metadata, multicast, and every host port regardless of
    /// [`NetworkPosture::host_ports`]. Grant deliberately; it reopens SSRF.
    Full,
}

impl SandboxNetwork {
    /// This posture plus egress to the listed host TCP ports.
    pub fn with_host_ports(self, ports: impl IntoIterator<Item = u16>) -> NetworkPosture {
        NetworkPosture::from(self).with_host_ports(ports)
    }

    /// This posture plus egress to the listed domain suffixes.
    pub fn with_domain_suffixes(
        self,
        suffixes: impl IntoIterator<Item = impl Into<String>>,
    ) -> NetworkPosture {
        NetworkPosture::from(self).with_domain_suffixes(suffixes)
    }
}

/// A guest network posture: how far out the guest can reach, and which host TCP
/// ports it may open.
///
/// `host_ports` has to be named explicitly because the convenience constructor
/// for a host rule (`Rule::allow_egress(Group(Host))`) leaves the port set
/// empty, and an empty port set means every port. A guest that needs one host
/// service would otherwise be handed the whole host: SSH, a database, a
/// container daemon, anything bound to `0.0.0.0`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct NetworkPosture {
    /// Outside reach. Defaulted so a config that names only the fields it
    /// cares about — `{"host_ports": [8080]}`, or `{}` — deserializes to the
    /// narrow posture instead of failing on a missing field.
    #[serde(default)]
    pub network: SandboxNetwork,
    /// Host TCP ports the guest may connect to. Gateway DNS is always allowed
    /// and does not need to be listed.
    #[serde(default)]
    pub host_ports: Vec<u16>,
    /// Domain suffixes the guest may reach, matching the apex and any
    /// subdomain. Lets a session that has to install packages name its
    /// registries instead of taking [`SandboxNetwork::Public`] and the whole
    /// internet with it.
    ///
    /// Matching works through the gateway resolver's hostname cache, so it
    /// covers names the guest looked up there — which is every name it can
    /// resolve under these postures.
    #[serde(default)]
    pub domain_suffixes: Vec<String>,
}

impl From<SandboxNetwork> for NetworkPosture {
    fn from(network: SandboxNetwork) -> Self {
        Self {
            network,
            host_ports: Vec::new(),
            domain_suffixes: Vec::new(),
        }
    }
}

impl NetworkPosture {
    /// Grant egress to these host TCP ports, replacing any already set.
    pub fn with_host_ports(mut self, ports: impl IntoIterator<Item = u16>) -> Self {
        self.host_ports = ports.into_iter().collect();
        self
    }

    /// Grant egress to these domain suffixes, replacing any already set.
    pub fn with_domain_suffixes(
        mut self,
        suffixes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.domain_suffixes = suffixes.into_iter().map(Into::into).collect();
        self
    }

    /// The microsandbox policy for this posture.
    fn policy(&self) -> NetworkPolicy {
        use std::str::FromStr as _;

        use microsandbox_network::policy::{
            Action, Destination, DestinationGroup, Direction, DomainName, PortRange, Protocol, Rule,
        };

        if matches!(self.network, SandboxNetwork::Full) {
            return NetworkPolicy::allow_all();
        }

        // `Rule::allow_dns()` is narrow: UDP/TCP :53 to the gateway addresses
        // only, not to arbitrary resolvers the guest might aim at. It has to
        // come first, because under deny-by-default a policy without it refuses
        // every DNS query, including the one for `host.microsandbox.internal`.
        let mut rules = vec![Rule::allow_dns()];
        if matches!(self.network, SandboxNetwork::Public) {
            rules.push(Rule::allow_egress(Destination::Group(
                DestinationGroup::Public,
            )));
        }
        rules.extend(self.host_ports.iter().map(|&port| Rule {
            direction: Direction::Egress,
            destination: Destination::Group(DestinationGroup::Host),
            protocols: vec![Protocol::Tcp],
            ports: vec![PortRange::single(port)],
            action: Action::Allow,
        }));
        rules.extend(self.domain_suffixes.iter().filter_map(|suffix| {
            match DomainName::from_str(suffix) {
                Ok(name) => Some(Rule::allow_egress(Destination::DomainSuffix(name))),
                // Dropping the entry means the domain simply is not reachable,
                // so a typo costs a failed install rather than an unintended
                // grant. Logged loudly because the failure is otherwise opaque.
                Err(e) => {
                    log::warn!("sandbox network: ignoring invalid domain suffix '{suffix}': {e}");
                    None
                }
            }
        }));

        NetworkPolicy {
            default_egress: Action::Deny,
            // microsandbox's own serde default. It only governs connections
            // inbound to a published guest port, and nothing here publishes
            // one, so this is fail-closed for a capability that does not exist
            // yet rather than a restriction on anything in use.
            default_ingress: Action::Deny,
            rules,
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
        // Default posture: gateway DNS and nothing else (see SandboxNetwork).
        // Anything wider is opt-in, so a caller that never thinks about the
        // network does not get outbound reach by accident.
        config.spec.network.enabled = true;
        config.spec.network.policy = Some(wire_network_policy(NetworkPosture::default().policy()));
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

    /// Set the guest network posture. Accepts a bare [`SandboxNetwork`] for
    /// outside reach alone, or [`SandboxNetwork::with_host_ports`] to also grant
    /// specific host TCP ports. Defaults to [`SandboxNetwork::HostOnly`] with no
    /// host ports.
    pub fn network(mut self, posture: impl Into<NetworkPosture>) -> Self {
        let posture = posture.into();
        self.config.spec.network.enabled = true;
        self.config.spec.network.policy = Some(wire_network_policy(posture.policy()));
        self
    }

    /// Append a volume mount.
    pub fn mount(mut self, mount: VolumeMount) -> Self {
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
        Self::try_from_archive_inner(path, NetworkPosture::default()).await
    }

    /// Like [`try_from_archive`](Self::try_from_archive) but with an explicit
    /// posture. An archive is only a filesystem snapshot and doesn't carry the
    /// network policy, so the desired posture is re-applied here — including its
    /// host ports, which are otherwise lost across the round trip.
    pub async fn try_from_archive_with_network(
        path: impl AsRef<Path>,
        posture: impl Into<NetworkPosture>,
    ) -> anyhow::Result<Self> {
        Self::try_from_archive_inner(path, posture.into()).await
    }

    async fn try_from_archive_inner(
        path: impl AsRef<Path>,
        posture: NetworkPosture,
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
        let name =
            snap.manifest().source_sandbox.clone().unwrap_or_else(|| {
                format!("ailoy-{}", hex::encode(&Uuid::new_v4().as_bytes()[..8]))
            });

        let result = microsandbox::Sandbox::builder(&name)
            .from_snapshot(snap_path.to_string_lossy().into_owned())
            .pull_policy(PullPolicy::IfMissing)
            .replace()
            .network(|n| n.policy(posture.policy()))
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

    /// `SandboxNetwork::HostOnly` grants gateway DNS and whatever host ports
    /// were asked for, nothing else — public egress stays blocked.
    #[tokio::test]
    async fn test_host_only_blocks_public_egress() {
        assert_ne!(
            public_egress_exit_code(SandboxNetwork::HostOnly).await,
            0,
            "public TCP connect (1.1.1.1:443) must be blocked under HostOnly"
        );
    }

    /// Contrast: `SandboxNetwork::Public` allows public egress, confirming
    /// HostOnly's block is the policy — not a broken network.
    #[tokio::test]
    async fn test_public_allows_public_egress() {
        assert_eq!(
            public_egress_exit_code(SandboxNetwork::Public).await,
            0,
            "public TCP connect (1.1.1.1:443) must succeed under Public"
        );
    }

    /// The port set has to be enforced by microsandbox, not merely recorded in
    /// the spec. Two host listeners, one port granted and one not: the guest
    /// must reach exactly one of them. This is the grant a host-side forward
    /// server depends on, so it is worth proving against a real VM rather than
    /// only against the evaluator.
    #[tokio::test]
    async fn test_host_ports_are_enforced_from_inside_the_guest() {
        // Bound on all interfaces because the guest arrives via the gateway
        // address, not loopback.
        let granted = tokio::net::TcpListener::bind("0.0.0.0:0")
            .await
            .expect("bind granted port");
        let ungranted = tokio::net::TcpListener::bind("0.0.0.0:0")
            .await
            .expect("bind ungranted port");
        let granted_port = granted.local_addr().expect("granted addr").port();
        let ungranted_port = ungranted.local_addr().expect("ungranted addr").port();
        // Keep accepting, so a connect the policy permits actually completes.
        tokio::spawn(async move { while granted.accept().await.is_ok() {} });
        tokio::spawn(async move { while ungranted.accept().await.is_ok() {} });

        let mut sandbox = SandboxBuilder::new()
            .image("alpine:latest")
            .network(SandboxNetwork::HostOnly.with_host_ports([granted_port]))
            .build()
            .await
            .expect("build");
        let console = sandbox.start().await.expect("start");
        let probe = |port: u16| format!("nc -zw5 host.microsandbox.internal {port} 2>/dev/null");

        let allowed = console
            .exec_shell(probe(granted_port), Some(15))
            .await
            .expect("exec granted probe")
            .exit_code;
        let blocked = console
            .exec_shell(probe(ungranted_port), Some(15))
            .await
            .expect("exec ungranted probe")
            .exit_code;

        assert_eq!(
            allowed, 0,
            "host port {granted_port} was granted and must be reachable"
        );
        assert_ne!(
            blocked, 0,
            "host port {ungranted_port} was not granted and must be unreachable"
        );
    }

    // ── network policy, evaluated directly ─────────────────────────────────────
    //
    // The tests above boot a VM to observe the policy's effect, which is slow
    // and can only probe one destination per run. These ask microsandbox's own
    // evaluator what a posture decides, so a rule that is wider than intended
    // shows up as a failing assertion rather than as an open port nobody looked
    // at.

    /// Shared state carrying a gateway IP, which is what `DestinationGroup::Host`
    /// rules match against.
    fn policy_test_state() -> microsandbox_network::shared::SharedState {
        let shared = microsandbox_network::shared::SharedState::new(8);
        shared.set_gateway_ips(Some(std::net::Ipv4Addr::new(10, 0, 2, 2)), None);
        shared
    }

    /// Decide one guest → host TCP connect under `posture`.
    fn host_port_action(
        posture: &NetworkPosture,
        port: u16,
    ) -> microsandbox_network::policy::Action {
        use microsandbox_network::policy::Protocol;

        let shared = policy_test_state();
        posture.policy().evaluate_egress(
            std::net::SocketAddr::from((std::net::Ipv4Addr::new(10, 0, 2, 2), port)),
            Protocol::Tcp,
            &shared,
        )
    }

    /// A posture that grants one host port must grant only that port. Every
    /// other host service — SSH, a database, a container daemon — has to stay
    /// out of reach, which is what an empty port set in a host rule would give
    /// away.
    #[test]
    fn host_ports_grant_only_the_listed_port() {
        use microsandbox_network::policy::Action;

        let posture = SandboxNetwork::HostOnly.with_host_ports([9000]);
        assert_eq!(host_port_action(&posture, 9000), Action::Allow);
        for port in [22, 2375, 5432, 9200, 11434] {
            assert_eq!(
                host_port_action(&posture, port),
                Action::Deny,
                "host port {port} must not be reachable"
            );
        }
    }

    /// No posture reaches a host port that was not asked for, `Public` included
    /// — widening outside reach must not widen host reach.
    #[test]
    fn no_posture_reaches_unlisted_host_ports() {
        use microsandbox_network::policy::Action;

        for network in [SandboxNetwork::HostOnly, SandboxNetwork::Public] {
            let posture = NetworkPosture::from(network);
            for port in [22, 2375, 5432, 9200, 11434] {
                assert_eq!(
                    host_port_action(&posture, port),
                    Action::Deny,
                    "{network:?} must not reach host port {port}"
                );
            }
        }
    }

    /// Gateway DNS survives the narrowing. Without it a deny-by-default policy
    /// refuses every lookup, including the one for `host.microsandbox.internal`,
    /// and the guest has no working network at all.
    #[test]
    fn every_posture_allows_gateway_dns() {
        use microsandbox_network::policy::{Action, Protocol};

        for network in [SandboxNetwork::HostOnly, SandboxNetwork::Public] {
            let policy = NetworkPosture::from(network).policy();
            let shared = policy_test_state();
            for protocol in [Protocol::Udp, Protocol::Tcp] {
                assert_eq!(
                    policy.evaluate_egress(
                        std::net::SocketAddr::from((std::net::Ipv4Addr::new(10, 0, 2, 2), 53)),
                        protocol,
                        &shared,
                    ),
                    Action::Allow,
                    "{network:?} must allow gateway DNS over {protocol:?}"
                );
            }
        }
    }

    /// Inbound connections are refused unless a rule says otherwise. Nothing
    /// publishes a guest port today; this keeps the first one that does from
    /// being reachable by every peer on the LAN by default.
    #[test]
    fn unmatched_ingress_is_denied() {
        use microsandbox_network::policy::{Action, Protocol};

        for network in [SandboxNetwork::HostOnly, SandboxNetwork::Public] {
            let policy = NetworkPosture::from(network).policy();
            let shared = policy_test_state();
            assert_eq!(
                policy.evaluate_ingress(
                    std::net::SocketAddr::from((std::net::Ipv4Addr::new(192, 168, 0, 14), 54321)),
                    8080,
                    Protocol::Tcp,
                    &shared,
                ),
                Action::Deny,
                "{network:?} must refuse an unmatched inbound connection"
            );
        }
    }

    /// The default posture is the narrow one, so a caller that never mentions
    /// the network gets no outbound reach beyond DNS.
    #[test]
    fn default_posture_is_host_only_with_no_host_ports() {
        use microsandbox_network::policy::Action;

        let posture = NetworkPosture::default();
        assert_eq!(posture.network, SandboxNetwork::HostOnly);
        assert!(posture.host_ports.is_empty());

        let shared = policy_test_state();
        assert_eq!(
            posture.policy().evaluate_egress(
                "1.1.1.1:443".parse().expect("literal socket address"),
                microsandbox_network::policy::Protocol::Tcp,
                &shared,
            ),
            Action::Deny,
            "the default posture must not reach the public internet"
        );
    }

    /// A stored posture that omits a field falls back to the narrow default
    /// rather than failing to parse. `network` is the field worth pinning: it
    /// is the one that decides outside reach, so a config which only lists
    /// host ports must still come back as `HostOnly`.
    #[test]
    fn omitted_posture_fields_deserialize_to_the_narrow_default() {
        let empty: NetworkPosture = serde_json::from_str("{}").expect("`{}` must deserialize");
        assert_eq!(empty, NetworkPosture::default());

        let ports_only: NetworkPosture =
            serde_json::from_str(r#"{"host_ports":[8080]}"#).expect("host_ports alone must parse");
        assert_eq!(ports_only.network, SandboxNetwork::HostOnly);
        assert_eq!(ports_only.host_ports, vec![8080]);
        assert!(ports_only.domain_suffixes.is_empty());

        // An explicit value still wins over the default.
        let explicit: NetworkPosture =
            serde_json::from_str(r#"{"network":"public"}"#).expect("explicit network must parse");
        assert_eq!(explicit.network, SandboxNetwork::Public);
    }

    /// A domain allowlist reaches the named registry and its subdomains without
    /// opening the rest of the internet. The evaluator matches these through the
    /// resolver's hostname cache, so the test seeds the cache the way a guest
    /// lookup would.
    #[test]
    fn domain_suffixes_allow_only_the_named_domains() {
        use std::{net::IpAddr, time::Duration};

        use microsandbox_network::{
            policy::{Action, Protocol},
            shared::ResolvedHostnameFamily,
        };

        let posture = SandboxNetwork::HostOnly.with_domain_suffixes(["pypi.org"]);
        let policy = posture.policy();
        let shared = policy_test_state();

        let allowed: IpAddr = "151.101.0.223".parse().expect("literal address");
        let denied: IpAddr = "203.0.114.5".parse().expect("literal address");
        shared.cache_resolved_hostname(
            "files.pythonhosted.pypi.org",
            ResolvedHostnameFamily::Ipv4,
            [allowed],
            Duration::from_secs(60),
        );
        shared.cache_resolved_hostname(
            "registry.evil.example",
            ResolvedHostnameFamily::Ipv4,
            [denied],
            Duration::from_secs(60),
        );

        assert_eq!(
            policy.evaluate_egress(
                std::net::SocketAddr::from((allowed, 443)),
                Protocol::Tcp,
                &shared
            ),
            Action::Allow,
            "a subdomain of an allowlisted suffix must be reachable"
        );
        assert_eq!(
            policy.evaluate_egress(
                std::net::SocketAddr::from((denied, 443)),
                Protocol::Tcp,
                &shared
            ),
            Action::Deny,
            "a domain outside the allowlist must stay blocked"
        );
    }

    /// An unparseable suffix must not become a wider grant. It is dropped, so
    /// the posture behaves as if it had never been listed.
    #[test]
    fn invalid_domain_suffix_is_dropped_rather_than_widening_the_policy() {
        let bad = SandboxNetwork::HostOnly.with_domain_suffixes(["not a domain"]);
        assert_eq!(
            bad.policy().rules.len(),
            NetworkPosture::from(SandboxNetwork::HostOnly)
                .policy()
                .rules
                .len(),
            "an invalid suffix must add no rule"
        );
    }

    /// `Full` stays the deliberate escape hatch: everything allowed, in both
    /// directions, host ports included.
    #[test]
    fn full_allows_everything() {
        use microsandbox_network::policy::Action;

        let posture = NetworkPosture::from(SandboxNetwork::Full);
        assert_eq!(host_port_action(&posture, 22), Action::Allow);
        let shared = policy_test_state();
        assert_eq!(
            posture.policy().evaluate_egress(
                "1.1.1.1:443".parse().expect("literal socket address"),
                microsandbox_network::policy::Protocol::Tcp,
                &shared,
            ),
            Action::Allow
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
