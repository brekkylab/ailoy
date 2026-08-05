//! A minimal ephemeral sandbox on raw `msb_krun` that mounts cortex volumes.
//!
//! Separation of concerns: **cortex owns the filesystems** — a caller mounts
//! them by passing [`cortex::VolumeSpec`]s, which cortex realizes into virtio-fs
//! backends via [`cortex::VolumeSpec::build`]. ailoy never names a specific
//! volume kind, so new cortex volumes need no change here. **ailoy owns the
//! sandbox**: booting the microVM, capturing output, persisting state, and
//! mounting each volume at its guest path.
//!
//! Each [`Sandbox::exec`] boots a fresh microVM (base rootfs over virtio-fs + a
//! persistent virtio-blk upper at `/data`), mounts every configured volume,
//! runs one command, and captures its output. `msb_krun::enter()` never returns
//! (it `_exit`s on guest shutdown), so the boot runs in a **child process** —
//! the same binary re-invoked, gated by [`boot_if_requested`], which consuming
//! binaries must call first in `main`.

use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use cortex::{PosixFs, VolumeSpec, Workspace, WorkspaceSpec};
use msb_krun::{DiskImageFormat, VmBuilder};
use serde::{Deserialize, Serialize};

use super::{Console, ExecResult};

const BOOT_ENV: &str = "AILOY_KRUN_BOOT";
const WORKSPACE_ENV: &str = "AILOY_KRUN_WORKSPACE";
const OUT_MARKER: &str = "__AILOY_OUT__";
const RC_MARKER: &str = "__AILOY_RC__";

const ALPINE_URL: &str = "https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/aarch64/alpine-minirootfs-3.24.1-aarch64.tar.gz";
const ALPINE_SHA256: &str = "f55a90f69052c5bd6f92cb09a8f47065970830b194c917a006fb94028e721259";

/// Guest path the init binary is placed and exec'd at.
const GUEST_INIT_PATH: &str = "/.ailoy-init";

/// The static guest init (see `crates/ailoy-guest-init`): mounts the block/
/// virtio-fs devices via `mount(2)` so glibc images (whose util-linux `mount`
/// the libkrun kernel rejects) work like busybox ones, and builds the overlay
/// root. Cross-compiled to the guest-arch musl target by `build.rs` and embedded
/// from `OUT_DIR` — no committed binary asset.
const GUEST_INIT_BIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ailoy-guest-init"));

/// Write the guest init into `rootfs` at [`GUEST_INIT_PATH`] (executable),
/// rewriting only when the bytes differ so a shared/cached rootfs is not churned.
fn ensure_guest_init(rootfs: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let dest = rootfs.join(GUEST_INIT_PATH.trim_start_matches('/'));
    let up_to_date = std::fs::read(&dest).map(|b| b == GUEST_INIT_BIN).unwrap_or(false);
    if !up_to_date {
        std::fs::write(&dest, GUEST_INIT_BIN)?;
        std::fs::set_permissions(&dest, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

/// Entitlements the boot process needs on macOS: create a VM via
/// Hypervisor.framework, and load libkrunfw (a differently-signed dylib).
#[cfg(target_os = "macos")]
const MACOS_ENTITLEMENTS: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "https://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.hypervisor</key>
    <true/>
    <key>com.apple.security.cs.disable-library-validation</key>
    <true/>
</dict>
</plist>
"#;

/// The executable [`Sandbox::exec`] re-invokes to boot the VM.
///
/// Elsewhere the running binary boots as-is. On macOS it usually lacks the
/// hypervisor entitlement — a plain `cargo run`/`cargo test` binary is unsigned —
/// so boot an ad-hoc-signed *copy* instead, cached by the source's path, mtime,
/// and size so the sign happens once per build. The consumer's own binary never
/// has to be signed.
#[cfg(not(target_os = "macos"))]
fn boot_helper_exe() -> io::Result<PathBuf> {
    std::env::current_exe()
}

#[cfg(target_os = "macos")]
fn boot_helper_exe() -> io::Result<PathBuf> {
    use std::hash::{Hash, Hasher};

    let src = std::env::current_exe()?;
    let meta = std::fs::metadata(&src)?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    src.hash(&mut hasher);
    let key = format!("{:016x}-{mtime}-{}", hasher.finish(), meta.len());

    let cache = krun_home()?.join("boot-helper");
    std::fs::create_dir_all(&cache)?;
    let dest = cache.join(format!("boot-{key}"));
    if dest.exists() {
        return Ok(dest);
    }

    // Copy to a temp path in the same dir, sign it, then rename into place so a
    // concurrent exec never sees a half-signed helper.
    let tmp = cache.join(format!("boot-{key}.{}.tmp", std::process::id()));
    std::fs::copy(&src, &tmp)?;
    let plist = cache.join("entitlements.plist");
    std::fs::write(&plist, MACOS_ENTITLEMENTS)?;
    let status = Command::new("codesign")
        .args(["-s", "-", "--force", "--entitlements"])
        .arg(&plist)
        .arg(&tmp)
        .status()?;
    if !status.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err(io::Error::other("codesign of boot helper failed"));
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

/// Boot the VM when this process is a re-invoked boot child, from a link-time
/// constructor that runs before `main` — so no consumer, and no test harness,
/// has to call [`boot_if_requested`]. A normal process (no [`BOOT_ENV`]) pays
/// only a single `getenv` here and continues to its own `main`.
#[ctor::ctor]
fn boot_on_start() {
    boot_if_requested();
}

/// Base dir for ailoy's krun assets — `AILOY_KRUN_HOME`, else `$HOME/.ailoy/krun`.
fn krun_home() -> io::Result<PathBuf> {
    std::env::var_os("AILOY_KRUN_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".ailoy/krun")))
        .ok_or_else(|| io::Error::other("AILOY_KRUN_HOME and HOME both unset"))
}

/// Resolve the libkrunfw kernel: `AILOY_KRUN_KERNEL`, else standard installs.
fn resolve_kernel() -> io::Result<PathBuf> {
    if let Some(k) = std::env::var_os("AILOY_KRUN_KERNEL") {
        return Ok(PathBuf::from(k));
    }
    let mut candidates = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(PathBuf::from(home).join(".microsandbox/lib/libkrunfw.dylib"));
    }
    candidates.push(PathBuf::from("/opt/homebrew/lib/libkrunfw.dylib"));
    candidates.push(PathBuf::from("/usr/local/lib/libkrunfw.dylib"));
    candidates.push(PathBuf::from("/usr/lib/libkrunfw.so"));
    candidates
        .into_iter()
        .find(|p| p.exists())
        .ok_or_else(|| io::Error::other("libkrunfw not found; set AILOY_KRUN_KERNEL"))
}

/// Provision (once) and return the base rootfs as a read-only **EROFS** image —
/// the overlay lower the guest mounts as a block device. A minimal Alpine
/// minirootfs today; `AILOY_KRUN_ROOTFS_EROFS` overrides with a prebuilt image.
fn ensure_base_erofs() -> io::Result<PathBuf> {
    if let Some(r) = std::env::var_os("AILOY_KRUN_ROOTFS_EROFS") {
        return Ok(PathBuf::from(r));
    }
    let home = krun_home()?;
    let erofs = home.join("alpine.erofs");
    if !erofs.exists() {
        let tarball = home.join("alpine-minirootfs.tar.gz");
        if !tarball.exists() {
            run_cmd(Command::new("curl").args(["-fsSL", ALPINE_URL, "-o"]).arg(&tarball))?;
        }
        verify_sha256(&tarball, ALPINE_SHA256);
        erofs_from_tarball(&tarball, &erofs)?;
    }
    Ok(erofs)
}

/// Encode a gzipped rootfs tarball as an EROFS image, reusing microsandbox-image
/// (`ingest_compressed_tar` + `write_erofs`). The ingest is async; drive it on a
/// throwaway runtime on a fresh thread so this stays callable from sync code even
/// when a Tokio runtime is already running on the caller's thread.
fn erofs_from_tarball(tarball: &Path, out: &Path) -> io::Result<()> {
    use microsandbox_image::erofs::write_erofs;
    use microsandbox_image::tar::{Compression, ingest_compressed_tar};
    use microsandbox_image::tree::ResourceLimits;

    let (tarball, out) = (tarball.to_path_buf(), out.to_path_buf());
    std::thread::spawn(move || -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(async {
            let f = tokio::fs::File::open(&tarball).await?;
            let res = ingest_compressed_tar(f, Compression::Gzip, &ResourceLimits::default(), None)
                .await
                .map_err(|e| io::Error::other(format!("ingest {}: {e:?}", tarball.display())))?;
            let tmp = out.with_extension("erofs.tmp");
            write_erofs(&res.tree, &tmp).map_err(|e| io::Error::other(format!("{e:?}")))?;
            std::fs::rename(&tmp, &out)?;
            Ok(())
        })
    })
    .join()
    .map_err(|_| io::Error::other("erofs build thread panicked"))?
}

/// The minimal virtio-fs root the VM boots into: it holds only the guest init at
/// [`GUEST_INIT_PATH`], which then overlays the EROFS lower + ext4 upper and
/// `pivot_root`s into the real image. Shared across sandboxes (content is fixed).
fn ensure_init_root() -> io::Result<PathBuf> {
    let dir = krun_home()?.join("init-root");
    std::fs::create_dir_all(&dir)?;
    ensure_guest_init(&dir)?;
    Ok(dir)
}

fn run_cmd(cmd: &mut Command) -> io::Result<()> {
    let status = cmd.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("{cmd:?} failed: {status}")))
    }
}

/// Best-effort integrity check (skips if no hashing tool is available).
fn verify_sha256(file: &Path, expected: &str) {
    for tool in ["shasum", "sha256sum"] {
        let mut cmd = Command::new(tool);
        if tool == "shasum" {
            cmd.arg("-a").arg("256");
        }
        if let Ok(out) = cmd.arg(file).output() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if let Some(got) = stdout.split_whitespace().next() {
                if got != expected {
                    eprintln!("ailoy krun: WARNING sha256 mismatch for {}", file.display());
                }
                return;
            }
        }
    }
}

/// The cortex workspace mounted into the sandbox: one unified tree served as a
/// single virtio-fs device at `guest_root`. Its sub-mounts (`WorkspaceSpec`)
/// appear under that root, routed internally by cortex — ailoy carries the spec
/// opaquely across the process boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct WorkspaceWire {
    /// virtio-fs device tag (assigned by ailoy).
    tag: String,
    /// Absolute guest path the unified tree mounts at.
    guest_root: String,
    /// The workspace to realize on the sandbox side.
    spec: WorkspaceSpec,
}

/// Result of running one command in the sandbox.
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub stdout: String,
    pub exit_code: i32,
    /// The command exceeded its timeout and the VM was killed.
    pub timed_out: bool,
}

/// Guest outbound-reach posture. The guest reaches the network through an
/// in-process smoltcp userspace stack ([`microsandbox_network`]) that NATs to
/// the host — no external proxy.
///
/// This names *outside* reach only. Reachable host TCP ports are a separate,
/// explicit grant via [`NetworkPosture::host_ports`]: the convenience host rule
/// (`Group(Host)`) leaves its port set empty, which matches *every* port, so
/// widening outside reach must never silently widen host reach too (see
/// brekkylab/ailoy#443).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxNetwork {
    /// No network device at all; the guest is fully offline.
    Disabled,
    /// Gateway DNS plus whatever host ports are granted, and nothing else — no
    /// public internet. The default: a caller that never thinks about the
    /// network gets no outbound reach by accident.
    #[default]
    HostOnly,
    /// `HostOnly` plus public-internet egress; private LAN, loopback,
    /// link-local, and metadata stay denied.
    Public,
    /// Unrestricted egress (`allow_all`). The deliberate escape hatch.
    Full,
}

impl SandboxNetwork {
    /// Pair this reach with the host TCP ports the guest may open.
    pub fn with_host_ports(self, ports: impl IntoIterator<Item = u16>) -> NetworkPosture {
        NetworkPosture::from(self).with_host_ports(ports)
    }

    /// Pair this reach with allowed outbound domain suffixes (e.g. a package
    /// registry) instead of taking all of `Public`.
    pub fn with_domain_suffixes(
        self,
        suffixes: impl IntoIterator<Item = impl Into<String>>,
    ) -> NetworkPosture {
        NetworkPosture::from(self).with_domain_suffixes(suffixes)
    }
}

/// A full network posture: outside [`reach`](Self::reach) plus the explicit host
/// TCP ports and domain suffixes the guest may reach. `host_ports` has to be
/// named explicitly — the host convenience rule leaves its port set empty,
/// which matches every port, so a bare posture grants no host reach beyond DNS.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NetworkPosture {
    /// Outside-reach posture.
    pub reach: SandboxNetwork,
    /// Host TCP ports the guest may open (beyond gateway DNS). Each reopens a
    /// path to a host service, so grant deliberately.
    pub host_ports: Vec<u16>,
    /// Outbound domain suffixes to allow without taking all of `Public`.
    pub domain_suffixes: Vec<String>,
}

impl From<SandboxNetwork> for NetworkPosture {
    fn from(reach: SandboxNetwork) -> Self {
        NetworkPosture {
            reach,
            host_ports: Vec::new(),
            domain_suffixes: Vec::new(),
        }
    }
}

impl NetworkPosture {
    /// Set the host TCP ports the guest may open.
    pub fn with_host_ports(mut self, ports: impl IntoIterator<Item = u16>) -> Self {
        self.host_ports = ports.into_iter().collect();
        self
    }

    /// Set the allowed outbound domain suffixes.
    pub fn with_domain_suffixes(
        mut self,
        suffixes: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.domain_suffixes = suffixes.into_iter().map(Into::into).collect();
        self
    }

    /// Realize this posture as a smoltcp egress policy. Never called for
    /// [`SandboxNetwork::Disabled`] (which attaches no device at all).
    #[cfg(feature = "sandbox")]
    fn policy(&self) -> microsandbox_network::policy::NetworkPolicy {
        use microsandbox_network::policy::{
            Action, Destination, DestinationGroup, Direction, DomainName, NetworkPolicy, PortRange,
            Protocol, Rule,
        };

        if matches!(self.reach, SandboxNetwork::Full) {
            return NetworkPolicy::allow_all();
        }

        // `Rule::allow_dns()` is narrow — UDP/TCP :53 to the gateway only — and
        // MUST come first: under deny-by-default a policy without it refuses
        // every DNS query, including the one for `host.microsandbox.internal`.
        let mut rules = vec![Rule::allow_dns()];
        if matches!(self.reach, SandboxNetwork::Public) {
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
            match suffix.parse::<DomainName>() {
                Ok(name) => Some(Rule::allow_egress(Destination::DomainSuffix(name))),
                // A dropped entry just isn't reachable, so a typo costs a failed
                // install rather than an unintended grant.
                Err(e) => {
                    eprintln!("ailoy krun: ignoring invalid domain suffix '{suffix}': {e:?}");
                    None
                }
            }
        }));

        NetworkPolicy {
            default_egress: Action::Deny,
            // Fail-closed for a capability that does not exist yet: nothing here
            // publishes an inbound guest port.
            default_ingress: Action::Deny,
            rules,
        }
    }
}

/// Guest vCPU count when the caller doesn't override it. Matches the old
/// microsandbox-engine default; the agents run Python/pandas workloads that a
/// single vCPU starves.
const DEFAULT_VCPUS: u8 = 2;
/// Guest memory (MiB) default. `pip install` + data tooling OOMs well under this.
const DEFAULT_MEMORY_MIB: u32 = 2048;
/// Per-exec wall-clock cap (seconds) when the caller passes no timeout. Mirrors
/// the old engine's `default_timeout_secs`.
const DEFAULT_TIMEOUT_SECS: u64 = 60;

/// A per-sandbox ext4 upper image, deleted from the host when the last clone of
/// its owning [`Sandbox`] drops. [`Sandbox::snapshot`] copies it out beforehand
/// to persist the session.
struct TempUpper(PathBuf);

impl Drop for TempUpper {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A fresh, empty ext4 upper at a unique temp path (formatted in pure Rust).
fn fresh_upper() -> io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let upper = std::env::temp_dir().join(format!("ailoy-upper-{}-{n}.ext4", std::process::id()));
    microsandbox_image::ext4::format_ext4(
        &upper,
        &microsandbox_image::ext4::Ext4FormatOptions {
            size_bytes: 2 << 30,
            journal_blocks: 4096,
        },
    )
    .map_err(|e| io::Error::other(format!("format upper ext4: {e}")))?;
    Ok(upper)
}

/// An ephemeral sandbox recipe: booted fresh per [`exec`](Sandbox::exec).
#[derive(Clone)]
pub struct Sandbox {
    kernel: PathBuf,
    /// Read-only base image as an EROFS block device — the overlay lower.
    rootfs_erofs: PathBuf,
    /// The writable ext4 overlay upper (the session's filesystem delta).
    upper: PathBuf,
    /// Held only for its `Drop`: deletes the temp upper on last clone drop
    /// (`Some` for a `Sandbox`-owned upper, `None` never happens today).
    _upper_guard: Option<Arc<TempUpper>>,
    /// The cortex workspace to mount, as `(guest_root, spec)`. The whole tree is
    /// served as one virtio-fs device; sub-mounts appear under `guest_root`.
    workspace: Option<(String, WorkspaceSpec)>,
    /// Guest vCPU count.
    vcpus: u8,
    /// Guest memory in MiB.
    memory_mib: u32,
    /// Guest network posture (outside reach + host ports + domain suffixes).
    network: NetworkPosture,
    /// Serializes boots so concurrent execs don't write the shared upper at once.
    exec_lock: Arc<tokio::sync::Mutex<()>>,
}

impl Sandbox {
    /// A fresh sandbox with an empty, ephemeral writable upper — an ext4 overlay
    /// over the (read-only) base rootfs. Writes anywhere in the guest persist
    /// across the per-exec VMs, but the upper is discarded when the sandbox (and
    /// all its clones) drop; persist it first with [`snapshot`](Self::snapshot).
    ///
    /// ailoy owns the base rootfs (a provisioned Alpine EROFS, or an image via
    /// [`with_image`](Self::with_image)) and the kernel.
    pub fn new() -> io::Result<Self> {
        let upper = fresh_upper()?;
        Self::assemble(upper.clone(), Some(Arc::new(TempUpper(upper))))
    }

    /// Restore a sandbox whose upper is seeded from a [`snapshot`](Self::snapshot),
    /// resuming that captured filesystem state. The caller must pair it with the
    /// same base image the snapshot was taken over.
    pub fn from_snapshot(snapshot: impl AsRef<Path>) -> io::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let upper =
            std::env::temp_dir().join(format!("ailoy-upper-r{}-{n}.ext4", std::process::id()));
        std::fs::copy(snapshot.as_ref(), &upper)?;
        Self::assemble(upper.clone(), Some(Arc::new(TempUpper(upper))))
    }

    fn assemble(upper: PathBuf, upper_guard: Option<Arc<TempUpper>>) -> io::Result<Self> {
        Ok(Self {
            kernel: resolve_kernel()?,
            rootfs_erofs: ensure_base_erofs()?,
            upper,
            _upper_guard: upper_guard,
            workspace: None,
            vcpus: DEFAULT_VCPUS,
            memory_mib: DEFAULT_MEMORY_MIB,
            network: NetworkPosture::default(),
            exec_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// Persist the current upper (the session's filesystem delta) to `dest` so a
    /// later [`from_snapshot`](Self::from_snapshot) can resume it. Waits for any
    /// in-flight exec so the image is captured consistently.
    pub async fn snapshot(&self, dest: impl AsRef<Path>) -> io::Result<()> {
        let _guard = self.exec_lock.lock().await;
        std::fs::copy(&self.upper, dest.as_ref())?;
        Ok(())
    }

    /// Set the guest network posture. Accepts a bare [`SandboxNetwork`] (outside
    /// reach only) or a [`NetworkPosture`] (`SandboxNetwork::Public
    /// .with_host_ports([..])`). Defaults to [`SandboxNetwork::HostOnly`] with no
    /// host ports — anything wider is opt-in.
    pub fn with_network(mut self, network: impl Into<NetworkPosture>) -> Self {
        self.network = network.into();
        self
    }

    /// Override the guest vCPU count (default [`DEFAULT_VCPUS`]).
    pub fn with_vcpus(mut self, vcpus: u8) -> Self {
        self.vcpus = vcpus;
        self
    }

    /// Override the guest memory in MiB (default [`DEFAULT_MEMORY_MIB`]).
    pub fn with_memory_mib(mut self, memory_mib: u32) -> Self {
        self.memory_mib = memory_mib;
        self
    }

    /// Override the base rootfs with a prebuilt read-only EROFS image.
    pub fn with_rootfs_erofs(mut self, erofs: impl Into<PathBuf>) -> Self {
        self.rootfs_erofs = erofs.into();
        self
    }

    /// Use an OCI image (e.g. `python:3.12-slim`) as the base rootfs, pulling it
    /// (once) as a read-only EROFS under `<krun_home>/images/<ref>.erofs`. The
    /// guest overlays a writable ext4 upper over it, so the image stays pristine.
    ///
    /// Public Docker Hub images only for now — see [`super::oci`].
    pub async fn with_image(self, reference: &str) -> io::Result<Self> {
        let sanitized: String = reference
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let erofs = krun_home()?
            .join("images")
            .join(format!("{sanitized}.erofs"));
        super::oci::pull_erofs(reference, &erofs).await?;
        Ok(self.with_rootfs_erofs(erofs))
    }

    /// Override the libkrunfw kernel path.
    pub fn with_kernel(mut self, kernel: impl Into<PathBuf>) -> Self {
        self.kernel = kernel.into();
        self
    }

    /// Mount a cortex [`WorkspaceSpec`] as one unified tree at `guest_root`. The
    /// workspace's sub-mounts appear under that root (routed internally by
    /// cortex), served as a single virtio-fs device — the same tree a WebDAV or
    /// host-FUSE frontend built from the same spec would show.
    pub fn with_workspace(
        mut self,
        guest_root: impl Into<String>,
        spec: WorkspaceSpec,
    ) -> Self {
        self.workspace = Some((guest_root.into(), spec));
        self
    }

    /// The workspace mount as wire data (a fixed device tag), if any.
    fn wire_workspace(&self) -> Option<WorkspaceWire> {
        self.workspace.as_ref().map(|(guest_root, spec)| WorkspaceWire {
            tag: "ailoyws".to_string(),
            guest_root: guest_root.clone(),
            spec: spec.clone(),
        })
    }

    /// Boot a fresh microVM, run `cmd`, and capture its output. `/data` is the
    /// persistent upper; each configured volume is mounted at its guest path.
    pub fn exec(&self, cmd: &str, timeout: Option<u64>) -> io::Result<ExecOutput> {
        let console = tempfile::NamedTempFile::new()?;
        let console_path = console.path().to_path_buf();
        let exe = boot_helper_exe()?;
        let ws = self.wire_workspace();
        let payload = build_payload(cmd);
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let workspace_json = serde_json::to_string(&ws)
            .map_err(|e| io::Error::other(format!("serialize workspace: {e}")))?;
        let network_json = serde_json::to_string(&self.network)
            .map_err(|e| io::Error::other(format!("serialize network: {e}")))?;

        let mut child = Command::new(exe)
            .env(BOOT_ENV, "1")
            .env("AILOY_KRUN_KERNEL", &self.kernel)
            .env("AILOY_KRUN_ROOTFS_EROFS", &self.rootfs_erofs)
            .env("AILOY_KRUN_UPPER", &self.upper)
            .env("AILOY_KRUN_CONSOLE", &console_path)
            .env("AILOY_KRUN_B64", &b64)
            .env("AILOY_KRUN_VCPUS", self.vcpus.to_string())
            .env("AILOY_KRUN_MEMORY_MIB", self.memory_mib.to_string())
            .env("AILOY_KRUN_NETWORK", &network_json)
            .env(WORKSPACE_ENV, &workspace_json)
            .spawn()?;

        // Bound the boot child by wall-clock: a runaway guest command otherwise
        // hangs the whole exec forever. Poll `try_wait` to a deadline, then kill.
        let secs = timeout.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
        let (status, timed_out) = loop {
            if let Some(status) = child.try_wait()? {
                break (status, false);
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let status = child.wait()?;
                break (status, true);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        };

        let raw = std::fs::read_to_string(&console_path).unwrap_or_default();
        let mut out = parse_output(&raw, status.code());
        out.timed_out = timed_out;
        Ok(out)
    }
}

/// Single-quote a shell word so arbitrary characters survive.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn shell_join(program: &str, args: &[String]) -> String {
    let mut cmd = shell_quote(program);
    for a in args {
        cmd.push(' ');
        cmd.push_str(&shell_quote(a));
    }
    cmd
}

/// The krun sandbox as an ailoy exec backend: each `exec` boots a fresh microVM
/// (with the cortex volumes mounted) and captures the command's output.
#[async_trait]
impl Console for Sandbox {
    fn get_os(&self) -> &str {
        "linux"
    }

    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        let cmd = if program == "sh" && args.len() == 2 && args[0] == "-c" {
            args[1].clone()
        } else {
            shell_join(&program, &args)
        };
        let _guard = self.exec_lock.lock().await;
        let sb = self.clone();
        let out = tokio::task::spawn_blocking(move || sb.exec(&cmd, timeout))
            .await
            .map_err(|e| anyhow::anyhow!("krun exec join: {e}"))??;
        Ok(ExecResult {
            stdout: out.stdout,
            stderr: String::new(),
            exit_code: out.exit_code,
            timed_out: out.timed_out,
        })
    }
}

/// Guest program: run the command between markers, syncing before shutdown so
/// the overlay upper's writes flush to the host image. The filesystem is already
/// set up by the guest init (overlay root + virtio-fs shares) before this runs.
fn build_payload(cmd: &str) -> String {
    format!(
        "echo {OUT_MARKER}\n\
         {cmd}\n\
         __ailoy_rc=$?\n\
         sync\n\
         echo {RC_MARKER}$__ailoy_rc\n"
    )
}

/// Boot the configured VM (mounting each cortex volume via
/// [`cortex::VolumeSpec::build`]) and never return, when [`BOOT_ENV`] is set;
/// otherwise return immediately.
///
/// A link-time constructor calls this before `main` (see `boot_on_start`), so
/// consumers no longer need to call it themselves — it is kept public only for
/// explicit/embedded use. `Sandbox::exec` is the code that spawns a boot child
/// with `BOOT_ENV` set.
pub fn boot_if_requested() {
    if std::env::var(BOOT_ENV).is_err() {
        return;
    }
    let get = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("missing {k}"));
    let kernel = PathBuf::from(get("AILOY_KRUN_KERNEL"));
    let rootfs_erofs = PathBuf::from(get("AILOY_KRUN_ROOTFS_EROFS"));
    let upper = PathBuf::from(get("AILOY_KRUN_UPPER"));
    let console = PathBuf::from(get("AILOY_KRUN_CONSOLE"));
    let b64 = get("AILOY_KRUN_B64");
    let workspace: Option<WorkspaceWire> = serde_json::from_str(&get(WORKSPACE_ENV))
        .unwrap_or_else(|e| panic!("parse {WORKSPACE_ENV}: {e}"));

    // The payload can be large (e.g. a `write` embedding base64 file bytes), so
    // it CANNOT ride in the exec argv — msb_krun places exec args in the kernel
    // cmdline, which has a hard size limit (`TooLarge`). Instead, decode it to a
    // host file and carry it in over a dedicated virtio-fs control mount; the
    // exec argv then stays a tiny fixed bootstrap.
    let payload = base64::engine::general_purpose::STANDARD
        .decode(&b64)
        .unwrap_or_else(|e| panic!("decode AILOY_KRUN_B64: {e}"));
    let ctrl_dir = std::env::temp_dir().join(format!("ailoy-ctrl-{}", std::process::id()));
    std::fs::create_dir_all(&ctrl_dir).unwrap_or_else(|e| panic!("create ctrl dir: {e}"));
    std::fs::write(ctrl_dir.join("run.sh"), &payload)
        .unwrap_or_else(|e| panic!("write ctrl run.sh: {e}"));
    let ctrl_backend = VolumeSpec::Local { host: ctrl_dir }
        .build()
        .unwrap_or_else(|e| panic!("build ctrl volume: {e}"));

    // The VM boots into a minimal virtio-fs root holding only the guest init; the
    // init overlays the EROFS lower + ext4 upper (via mount(2)) and pivots into
    // the real image before the shell runs the payload.
    let init_root = ensure_init_root().unwrap_or_else(|e| panic!("prepare init root: {e}"));
    let bootstrap = "sh /.ailoyctrl/run.sh";

    // Resource sizing travels from the parent's `Sandbox` via env.
    let vcpus: u8 = std::env::var("AILOY_KRUN_VCPUS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_VCPUS);
    let memory_mib: u32 = std::env::var("AILOY_KRUN_MEMORY_MIB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEMORY_MIB);

    let mut builder = VmBuilder::new()
        .machine(|m| m.vcpus(vcpus).memory_mib(memory_mib as usize))
        .kernel(|k| k.krunfw_path(&kernel))
        .fs(|fs| fs.root(&init_root))
        .fs(move |fs| fs.tag("ailoyctrl").custom(ctrl_backend))
        // vda: the writable ext4 upper. vdb: the read-only base image (EROFS).
        // Attach order fixes the guest device names.
        .disk(|d| d.path(&upper).format(DiskImageFormat::Raw))
        .disk(|d| {
            d.path(&rootfs_erofs)
                .read_only(true)
                .format(DiskImageFormat::Raw)
        })
        .console(|c| c.output(&console));

    // The mount spec the guest init needs for the workspace share (tag:guest_root),
    // captured before `workspace` is consumed below.
    let init_ws_env = workspace
        .as_ref()
        .map(|w| format!("{}:{}", w.tag, w.guest_root));

    // Realize the whole cortex workspace as one virtio-fs device: `from_spec`
    // rebuilds the unified tree, `PosixFs` binds it to msb_krun's `DynFileSystem`.
    // The guest's single mount at `guest_root` sees every sub-mount, routed
    // internally by cortex.
    if let Some(w) = workspace {
        let ws = Workspace::from_spec(&w.spec).unwrap_or_else(|e| {
            eprintln!("ailoy krun: build workspace: {e}");
            std::process::exit(1);
        });
        let backend: Box<dyn msb_krun::DynFileSystem + Send + Sync> = Box::new(PosixFs::new(ws));
        let tag = w.tag.clone();
        builder = builder.fs(move |fs| fs.tag(&tag).custom(backend));
    }

    // Guest networking. An in-process smoltcp userspace stack NATs guest egress
    // to the host under the posture's policy; it is driven by a tokio runtime
    // that must outlive the VM. `enter()` below diverges (never unwinds,
    // `_exit`s on guest shutdown), so these guards live for the VM's lifetime.
    let posture: NetworkPosture = serde_json::from_str(
        &std::env::var("AILOY_KRUN_NETWORK").unwrap_or_default(),
    )
    .unwrap_or_default();
    let mut net_prelude = String::new();
    let _net_guard: Option<(tokio::runtime::Runtime, microsandbox_network::network::SmoltcpNetwork)>;
    if posture.reach == SandboxNetwork::Disabled {
        _net_guard = None;
    } else {
        // The smoltcp stack's TLS/DNS machinery expects a rustls crypto provider.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let mut netcfg = microsandbox_network::config::NetworkConfig::default();
        netcfg.policy = posture.policy();
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap_or_else(|e| panic!("ailoy krun: net runtime: {e}"));
        let mut stack = microsandbox_network::network::SmoltcpNetwork::new(netcfg, 0);
        stack.start(rt.handle().clone());
        let guest_mac = stack.guest_mac();
        let backend = stack.take_backend();
        net_prelude = guest_net_setup(&stack.guest_env_vars());
        builder = builder.net(move |n| n.mac(guest_mac).custom(backend));
        _net_guard = Some((rt, stack));
    }

    // The guest configures eth0 (if networking is on) before running the payload.
    let script = format!("{net_prelude}{bootstrap}");

    // Exec the guest init, not the shell directly: it mounts the ctrl/workspace
    // shares and the upper via mount(2), then hands off to `/bin/sh -c <script>`.
    // Mount targets travel in the environment (small, fixed); the large payload
    // stays on the ctrl share the init mounts.
    let result = builder
        .exec(|e| {
            // The init overlays the ext4 upper (/dev/vda) over the read-only image
            // root, then mounts the ctrl/workspace virtio-fs shares, before the
            // shell runs the payload.
            let mut e = e
                .path(GUEST_INIT_PATH)
                .env("AILOY_INIT_LOWER", "/dev/vdb")
                .env("AILOY_INIT_UPPER", "/dev/vda")
                .env("AILOY_INIT_CTRL", "ailoyctrl:/.ailoyctrl");
            if let Some(ws) = &init_ws_env {
                e = e.env("AILOY_INIT_WS", ws);
            }
            e.args(["/bin/sh", "-c", script.as_str()])
        })
        .build()
        .and_then(|vm| vm.enter());
    match result {
        Ok(never) => match never {},
        Err(e) => {
            eprintln!("ailoy krun boot: {e}");
            std::process::exit(1);
        }
    }
}

/// Emit the guest shell commands that bring up `eth0` with the static IPv4
/// address, gateway, and DNS the smoltcp stack assigned (carried in the stack's
/// guest env vars as `addr=<ip>/30,gw=<gw>,dns=<gw>`). The base rootfs has no
/// network agent, so the guest must configure the interface itself.
fn guest_net_setup(env_vars: &[(String, String)]) -> String {
    for (_key, val) in env_vars {
        // The IPv4 entry is the one carrying both `addr=` and `gw=`.
        if !(val.contains("addr=") && val.contains("gw=")) {
            continue;
        }
        let mut addr = None;
        let mut gw = None;
        for part in val.split(',') {
            if let Some(a) = part.strip_prefix("addr=") {
                addr = Some(a);
            } else if let Some(g) = part.strip_prefix("gw=") {
                gw = Some(g);
            }
        }
        if let (Some(addr), Some(gw)) = (addr, gw) {
            // Must be a single line: this is spliced into the exec argv, which
            // msb_krun writes to the kernel cmdline (newlines → `InvalidAscii`).
            return format!(
                "ip link set eth0 up 2>/dev/null; \
                 ip addr add {addr} dev eth0 2>/dev/null; \
                 ip route add default via {gw} dev eth0 2>/dev/null; \
                 printf 'nameserver {gw}\\n' > /etc/resolv.conf 2>/dev/null; "
            );
        }
    }
    String::new()
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for d in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&d) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn parse_output(raw: &str, child_exit: Option<i32>) -> ExecOutput {
    let clean = strip_ansi(raw);
    match clean.split(OUT_MARKER).nth(1) {
        Some(rest) => {
            let stdout = rest.split(RC_MARKER).next().unwrap_or("").trim().to_string();
            let exit_code = rest
                .split(RC_MARKER)
                .nth(1)
                .and_then(|t| t.trim().lines().next())
                .and_then(|n| n.trim().parse::<i32>().ok())
                .unwrap_or(-1);
            ExecOutput {
                stdout,
                exit_code,
                timed_out: false,
            }
        }
        None => ExecOutput {
            stdout: clean.trim().to_string(),
            exit_code: child_exit.unwrap_or(-1),
            timed_out: false,
        },
    }
}
