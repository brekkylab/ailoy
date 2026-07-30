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
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use cortex::VolumeSpec;
use msb_krun::{DiskImageFormat, VmBuilder};
use serde::{Deserialize, Serialize};

use super::{Console, ExecResult};

const BOOT_ENV: &str = "AILOY_KRUN_BOOT";
const MOUNTS_ENV: &str = "AILOY_KRUN_MOUNTS";
const OUT_MARKER: &str = "__AILOY_OUT__";
const RC_MARKER: &str = "__AILOY_RC__";

/// One volume mounted into the sandbox: a cortex volume at a guest path.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct MountWire {
    /// virtio-fs device tag (assigned by ailoy).
    tag: String,
    /// Absolute guest path to mount at.
    guest_path: String,
    /// The cortex volume to realize on the sandbox side (opaque to ailoy).
    spec: VolumeSpec,
}

/// Result of running one command in the sandbox.
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub stdout: String,
    pub exit_code: i32,
}

/// An ephemeral sandbox recipe: booted fresh per [`exec`](Sandbox::exec).
#[derive(Clone)]
pub struct Sandbox {
    kernel: PathBuf,
    rootfs: PathBuf,
    upper: PathBuf,
    /// Cortex volumes to mount, as `(guest_path, spec)`.
    volumes: Vec<(String, VolumeSpec)>,
    /// Serializes boots so concurrent execs don't write the shared upper at once.
    exec_lock: Arc<tokio::sync::Mutex<()>>,
}

impl Sandbox {
    /// `rootfs`: base directory served read-mostly over virtio-fs. `upper`: a
    /// host disk image (created sparse, 2 GiB logical, if missing) mounted at
    /// `/data` for persistent state.
    pub fn new(
        rootfs: impl Into<PathBuf>,
        upper: impl Into<PathBuf>,
        kernel: impl Into<PathBuf>,
    ) -> io::Result<Self> {
        let upper = upper.into();
        if !upper.exists() {
            let f = std::fs::File::create(&upper)?;
            f.set_len(2 << 30)?;
        }
        Ok(Self {
            kernel: kernel.into(),
            rootfs: rootfs.into(),
            upper,
            volumes: Vec::new(),
            exec_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// Mount a cortex volume at `guest_path`. Any [`cortex::VolumeSpec`] works —
    /// ailoy stays agnostic to the volume kind.
    pub fn mount(mut self, guest_path: impl Into<String>, spec: VolumeSpec) -> Self {
        self.volumes.push((guest_path.into(), spec));
        self
    }

    /// Assign virtio-fs tags and produce the wire mounts.
    fn wire_mounts(&self) -> Vec<MountWire> {
        self.volumes
            .iter()
            .enumerate()
            .map(|(i, (guest_path, spec))| MountWire {
                tag: format!("vol{i}"),
                guest_path: guest_path.clone(),
                spec: spec.clone(),
            })
            .collect()
    }

    /// Boot a fresh microVM, run `cmd`, and capture its output. `/data` is the
    /// persistent upper; each configured volume is mounted at its guest path.
    pub fn exec(&self, cmd: &str) -> io::Result<ExecOutput> {
        let console = tempfile::NamedTempFile::new()?;
        let console_path = console.path().to_path_buf();
        let exe = std::env::current_exe()?;
        let mounts = self.wire_mounts();
        let payload = build_payload(cmd, &mounts);
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);
        let mounts_json = serde_json::to_string(&mounts)
            .map_err(|e| io::Error::other(format!("serialize mounts: {e}")))?;

        let status = Command::new(exe)
            .env(BOOT_ENV, "1")
            .env("AILOY_KRUN_KERNEL", &self.kernel)
            .env("AILOY_KRUN_ROOTFS", &self.rootfs)
            .env("AILOY_KRUN_UPPER", &self.upper)
            .env("AILOY_KRUN_CONSOLE", &console_path)
            .env("AILOY_KRUN_B64", &b64)
            .env(MOUNTS_ENV, &mounts_json)
            .status()?;

        let raw = std::fs::read_to_string(&console_path).unwrap_or_default();
        Ok(parse_output(&raw, status.code()))
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
        _timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        let cmd = if program == "sh" && args.len() == 2 && args[0] == "-c" {
            args[1].clone()
        } else {
            shell_join(&program, &args)
        };
        let _guard = self.exec_lock.lock().await;
        let sb = self.clone();
        let out = tokio::task::spawn_blocking(move || sb.exec(&cmd))
            .await
            .map_err(|e| anyhow::anyhow!("krun exec join: {e}"))??;
        Ok(ExecResult {
            stdout: out.stdout,
            stderr: String::new(),
            exit_code: out.exit_code,
            timed_out: false,
        })
    }
}

/// Guest program: mount the upper at `/data` (formatting on first use) and each
/// cortex volume at its guest path, then run the command between markers,
/// syncing before shutdown so upper writes persist.
fn build_payload(cmd: &str, mounts: &[MountWire]) -> String {
    let mut vol_mounts = String::new();
    for m in mounts {
        vol_mounts.push_str(&format!(
            "mkdir -p {g}\nmount -t virtiofs {tag} {g} 2>/dev/null\n",
            g = m.guest_path,
            tag = m.tag
        ));
    }
    format!(
        "mkdir -p /data\n\
         mount /dev/vda /data 2>/dev/null || ( (mkfs.ext4 -F -q /dev/vda || mkfs.vfat /dev/vda) >/dev/null 2>&1; sync; mount /dev/vda /data 2>/dev/null )\n\
         {vol_mounts}\
         echo {OUT_MARKER}\n\
         {cmd}\n\
         __ailoy_rc=$?\n\
         sync\n\
         umount /data 2>/dev/null\n\
         echo {RC_MARKER}$__ailoy_rc\n"
    )
}

/// Child entry point. If [`BOOT_ENV`] is set, boot the configured VM — mounting
/// each cortex volume (realized via [`cortex::VolumeSpec::build`]) — and never
/// return. Consuming binaries must call this at the top of `main()`.
pub fn boot_if_requested() {
    if std::env::var(BOOT_ENV).is_err() {
        return;
    }
    let get = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("missing {k}"));
    let kernel = PathBuf::from(get("AILOY_KRUN_KERNEL"));
    let rootfs = PathBuf::from(get("AILOY_KRUN_ROOTFS"));
    let upper = PathBuf::from(get("AILOY_KRUN_UPPER"));
    let console = PathBuf::from(get("AILOY_KRUN_CONSOLE"));
    let b64 = get("AILOY_KRUN_B64");
    let mounts: Vec<MountWire> =
        serde_json::from_str(&get(MOUNTS_ENV)).unwrap_or_else(|e| panic!("parse {MOUNTS_ENV}: {e}"));

    let script = format!("echo {b64} | base64 -d > /run.sh; sh /run.sh");

    let mut builder = VmBuilder::new()
        .machine(|m| m.vcpus(1).memory_mib(512))
        .kernel(|k| k.krunfw_path(&kernel))
        .fs(|fs| fs.root(&rootfs))
        .disk(|d| d.path(&upper).format(DiskImageFormat::Raw))
        .console(|c| c.output(&console));

    // Realize each cortex volume and attach it as a virtio-fs device. cortex
    // maps the spec to a concrete backend — ailoy stays volume-agnostic.
    for m in mounts {
        let backend = m.spec.build().unwrap_or_else(|e| {
            eprintln!("ailoy krun: build volume {:?}: {e}", m.tag);
            std::process::exit(1);
        });
        let tag = m.tag.clone();
        builder = builder.fs(move |fs| fs.tag(&tag).custom(backend));
    }

    let result = builder
        .exec(|e| e.path("/bin/sh").args(["-c", script.as_str()]))
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
            ExecOutput { stdout, exit_code }
        }
        None => ExecOutput {
            stdout: clean.trim().to_string(),
            exit_code: child_exit.unwrap_or(-1),
        },
    }
}
