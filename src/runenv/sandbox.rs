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

/// Default base rootfs when no [`with_image`](Sandbox::with_image) /
/// `AILOY_KRUN_ROOTFS_EROFS` is given: the `alpine` OCI image, pulled through
/// the same path as any other image.
const DEFAULT_BASE_IMAGE: &str = "alpine:3.24.1";

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

    // Copy to a per-call-unique temp path in the same dir, sign it, then rename
    // into place so a concurrent exec never sees a half-signed helper — and two
    // boots in the *same* process (same pid, same key) don't sign one shared tmp.
    // The final `dest` is identical for both, so the atomic rename just wins once.
    let uniq = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        SEQ.fetch_add(1, Ordering::Relaxed)
    };
    let tmp = cache.join(format!("boot-{key}.{}.{uniq}.tmp", std::process::id()));
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

/// Resolve the libkrunfw kernel: `AILOY_KRUN_KERNEL`, else a known install, else
/// download the pinned build to `<krun_home>/lib`.
///
/// Checks our managed copy first (so a download is reused), then Homebrew/system
/// lib dirs — by both the ABI-versioned name the release ships
/// (`libkrunfw.5.dylib`) and the plain name a package manager installs
/// (`libkrunfw.dylib`).
fn resolve_kernel() -> io::Result<PathBuf> {
    if let Some(k) = std::env::var_os("AILOY_KRUN_KERNEL") {
        return Ok(PathBuf::from(k));
    }

    let os = std::env::consts::OS;
    let versioned = microsandbox_utils::libkrunfw_filename(os);
    let plain = match os {
        "macos" => "libkrunfw.dylib",
        "windows" => "libkrunfw.dll",
        _ => "libkrunfw.so",
    };
    let managed = krun_home()?.join("lib").join(&versioned);

    let mut candidates = vec![managed.clone()];
    for dir in ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"] {
        candidates.push(PathBuf::from(dir).join(&versioned));
        candidates.push(PathBuf::from(dir).join(plain));
    }
    if let Some(p) = candidates.into_iter().find(|p| p.exists()) {
        return Ok(p);
    }

    download_libkrunfw(&managed)?;
    Ok(managed)
}

/// Download libkrunfw for the host arch/OS to `dest` (temp-then-rename so a
/// partial download is never used). curl-downloaded, so no macOS quarantine to
/// block `dlopen`. The release is tagged by the microsandbox version
/// ([`microsandbox_utils::PREBUILT_VERSION`]) — the libkrunfw it bundles matches
/// the `microsandbox-*` crates we pin.
fn download_libkrunfw(dest: &Path) -> io::Result<()> {
    let url = microsandbox_utils::libkrunfw_download_url(
        microsandbox_utils::PREBUILT_VERSION,
        std::env::consts::ARCH,
        std::env::consts::OS,
    );
    download_file(&url, dest)
}

/// Provision (once) and return the base rootfs as a read-only **EROFS** image —
/// the overlay lower the guest mounts as a block device. Defaults to the
/// [`DEFAULT_BASE_IMAGE`] OCI image, pulled through the same path as
/// [`with_image`](Sandbox::with_image); `AILOY_KRUN_ROOTFS_EROFS` overrides with
/// a prebuilt image.
fn ensure_base_erofs() -> io::Result<PathBuf> {
    if let Some(r) = std::env::var_os("AILOY_KRUN_ROOTFS_EROFS") {
        return Ok(PathBuf::from(r));
    }
    pull_image_erofs(DEFAULT_BASE_IMAGE)
}

/// The on-disk EROFS cache path for an image reference, under `<krun_home>/images`.
fn image_erofs_path(reference: &str) -> io::Result<PathBuf> {
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
    Ok(krun_home()?.join("images").join(format!("{sanitized}.erofs")))
}

/// Pull an OCI image to its cached EROFS (once) and return the path. Drives the
/// async pull on a throwaway runtime on a fresh thread so it stays callable from
/// sync code even when a Tokio runtime is already running on the caller's thread.
fn pull_image_erofs(reference: &str) -> io::Result<PathBuf> {
    let dest = image_erofs_path(reference)?;
    if dest.exists() {
        return Ok(dest);
    }
    let (reference, dest2) = (reference.to_string(), dest.clone());
    std::thread::spawn(move || -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(super::oci::pull_erofs(&reference, &dest2))
    })
    .join()
    .map_err(|_| io::Error::other("image pull thread panicked"))??;
    Ok(dest)
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

/// Download `url` to `dest` with reqwest, driven on a throwaway runtime on a
/// fresh thread so it is safe to call from sync code even under an ambient Tokio
/// runtime. Temp-then-rename so a partial download is never used.
fn download_file(url: &str, dest: &Path) -> io::Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = dest.with_extension("download.tmp");
    let (url, tmp2) = (url.to_string(), tmp.clone());
    std::thread::spawn(move || -> io::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        rt.block_on(async {
            let resp = reqwest::get(&url)
                .await
                .map_err(io::Error::other)?
                .error_for_status()
                .map_err(io::Error::other)?;
            let bytes = resp.bytes().await.map_err(io::Error::other)?;
            std::fs::write(&tmp2, &bytes)
        })
    })
    .join()
    .map_err(|_| io::Error::other("download thread panicked"))??;
    std::fs::rename(&tmp, dest)?;
    Ok(())
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
        // MUST come first: under deny-by-default a policy without it refuses every
        // DNS query, so even an otherwise-allowed egress domain never resolves.
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

/// A unique temp path for a per-sandbox upper image.
fn fresh_upper_path() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("ailoy-upper-{}-{n}.ext4", std::process::id()))
}

/// A per-sandbox virtio-fs boot root (holds only the guest init). Created and
/// populated on the host per [`Sandbox`], removed when the last clone drops — so
/// concurrent boots never share one writable root (each boots into its own).
struct TempInitRoot(PathBuf);

impl Drop for TempInitRoot {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A fresh boot root at a unique temp dir, populated with the guest init. Owned
/// by the parent `Sandbox`; the boot child receives only its path (by env).
fn fresh_init_root() -> io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("ailoy-initroot-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    ensure_guest_init(&dir)?;
    Ok(dir)
}

/// A fresh, empty ext4 upper at a unique temp path (formatted in pure Rust).
fn fresh_upper() -> io::Result<PathBuf> {
    let upper = fresh_upper_path();
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

/// Snapshot container magic (`AILOYSN` + format version).
const SNAP_MAGIC: &[u8; 8] = b"AILOYSN1";

/// Write `src` (a sparse ext4 upper) to `dest` as a gzip snapshot holding only
/// the allocated extents — O(data written), not O(logical size) (cf.
/// microsandbox PR #1150). Layout (all little-endian, gzip-compressed):
/// `MAGIC | logical_len u64 | n_extents u64 | (offset u64, len u64, bytes)*`.
fn write_snapshot(src: &Path, dest: &Path) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut f = std::fs::File::open(src)?;
    let logical = f.metadata()?.len();
    let extents = match microsandbox_utils::extent::ExtentMap::scan_file(&f)? {
        Some(m) => m.extents,
        // Filesystem can't report holes — fall back to one dense extent.
        None => vec![(0, logical)],
    };

    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(dest)?,
        flate2::Compression::default(),
    );
    enc.write_all(SNAP_MAGIC)?;
    enc.write_all(&logical.to_le_bytes())?;
    enc.write_all(&(extents.len() as u64).to_le_bytes())?;

    let mut buf = vec![0u8; 1 << 20];
    for (off, len) in extents {
        enc.write_all(&off.to_le_bytes())?;
        enc.write_all(&len.to_le_bytes())?;
        f.seek(SeekFrom::Start(off))?;
        let mut left = len;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            let n = f.read(&mut buf[..want])?;
            if n == 0 {
                return Err(io::Error::other("snapshot: short read in extent"));
            }
            enc.write_all(&buf[..n])?;
            left -= n as u64;
        }
    }
    enc.finish()?;
    Ok(())
}

/// Rebuild a sparse ext4 upper at `dest` from a [`write_snapshot`] archive:
/// truncate to the logical size (all holes) then write back each extent, so the
/// holes stay unallocated.
fn read_snapshot(src: &Path, dest: &Path) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};

    let mut dec = flate2::read::GzDecoder::new(std::fs::File::open(src)?);
    let mut magic = [0u8; 8];
    dec.read_exact(&mut magic)?;
    if &magic != SNAP_MAGIC {
        return Err(io::Error::other("not an ailoy snapshot"));
    }
    let mut u64buf = [0u8; 8];
    dec.read_exact(&mut u64buf)?;
    let logical = u64::from_le_bytes(u64buf);
    dec.read_exact(&mut u64buf)?;
    let n_extents = u64::from_le_bytes(u64buf);

    let mut out = std::fs::File::create(dest)?;
    out.set_len(logical)?;
    let mut buf = vec![0u8; 1 << 20];
    for _ in 0..n_extents {
        dec.read_exact(&mut u64buf)?;
        let off = u64::from_le_bytes(u64buf);
        dec.read_exact(&mut u64buf)?;
        let len = u64::from_le_bytes(u64buf);
        out.seek(SeekFrom::Start(off))?;
        let mut left = len;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            dec.read_exact(&mut buf[..want])?;
            out.write_all(&buf[..want])?;
            left -= want as u64;
        }
    }
    Ok(())
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
    /// The per-sandbox virtio-fs boot root (guest init only), handed to the boot
    /// child by path. Unique per sandbox, so concurrent boots share no root.
    init_root: PathBuf,
    /// Held only for its `Drop`: removes the temp init root on last clone drop.
    _init_root_guard: Option<Arc<TempInitRoot>>,
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
        let upper = fresh_upper_path();
        read_snapshot(snapshot.as_ref(), &upper)?;
        Self::assemble(upper.clone(), Some(Arc::new(TempUpper(upper))))
    }

    fn assemble(upper: PathBuf, upper_guard: Option<Arc<TempUpper>>) -> io::Result<Self> {
        let init_root = fresh_init_root()?;
        Ok(Self {
            kernel: resolve_kernel()?,
            rootfs_erofs: ensure_base_erofs()?,
            upper,
            _upper_guard: upper_guard,
            init_root: init_root.clone(),
            _init_root_guard: Some(Arc::new(TempInitRoot(init_root))),
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
        let (src, dest) = (self.upper.clone(), dest.as_ref().to_path_buf());
        tokio::task::spawn_blocking(move || write_snapshot(&src, &dest))
            .await
            .map_err(|e| io::Error::other(format!("snapshot join: {e}")))?
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
        let erofs = image_erofs_path(reference)?;
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
            .env("AILOY_KRUN_INIT_ROOT", &self.init_root)
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
    // The parent `Sandbox` creates and owns a unique boot root per sandbox and
    // passes its path here; fall back to the shared root only for a boot not
    // launched through `Sandbox::exec`.
    let init_root = match std::env::var_os("AILOY_KRUN_INIT_ROOT") {
        Some(p) => PathBuf::from(p),
        None => ensure_init_root().unwrap_or_else(|e| panic!("prepare init root: {e}")),
    };
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
