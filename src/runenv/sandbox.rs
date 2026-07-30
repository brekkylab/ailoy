//! A minimal ephemeral sandbox on raw `msb_krun` that mounts a cortex VFS.
//!
//! Separation of concerns: **cortex provides the filesystem** (a `Mountable`
//! adapted to `msb_krun::DynFileSystem` via `cortex::PosixAdapter`); **ailoy
//! owns the sandbox** — booting the microVM, capturing output, persisting
//! state, and mounting the cortex VFS into the guest.
//!
//! Each [`Sandbox::exec`] boots a fresh microVM (base rootfs over virtio-fs +
//! a persistent virtio-blk upper at `/data`), optionally mounts a cortex VFS
//! (e.g. S3) at a guest path, runs one command, and captures its output.
//! `msb_krun::enter()` never returns (it `_exit`s on guest shutdown), so the
//! boot runs in a **child process** — the same binary re-invoked, gated by
//! [`boot_if_requested`], which consuming binaries must call first in `main`.

use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use cortex::{PosixAdapter, S3Config, S3Volume};
use msb_krun::{DiskImageFormat, VmBuilder};

use super::{Console, ExecResult};

const BOOT_ENV: &str = "AILOY_KRUN_BOOT";
const OUT_MARKER: &str = "__AILOY_OUT__";
const RC_MARKER: &str = "__AILOY_RC__";
/// virtio-fs tag the cortex VFS is attached under.
const VFS_TAG: &str = "wsvfs";

/// A cortex S3 VFS to mount into the sandbox at `guest_path`.
#[derive(Clone, Debug)]
pub struct S3Vfs {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint: Option<String>,
    pub key_prefix: Option<String>,
    /// Absolute guest path to mount the VFS at (e.g. `/workspace`).
    pub guest_path: String,
}

impl S3Vfs {
    fn to_config(&self) -> S3Config {
        S3Config {
            bucket: self.bucket.clone(),
            region: self.region.clone(),
            access_key_id: self.access_key_id.clone(),
            secret_access_key: self.secret_access_key.clone(),
            endpoint: self.endpoint.clone(),
            key_prefix: self.key_prefix.clone(),
        }
    }
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
    s3: Option<S3Vfs>,
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
            s3: None,
            exec_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// Mount a cortex S3 VFS into the guest at `s3.guest_path`.
    pub fn with_s3(mut self, s3: S3Vfs) -> Self {
        self.s3 = Some(s3);
        self
    }

    /// Boot a fresh microVM, run `cmd`, and capture its output. `/data` is the
    /// persistent upper; the cortex VFS (if any) is mounted at its guest path.
    pub fn exec(&self, cmd: &str) -> io::Result<ExecOutput> {
        let console = tempfile::NamedTempFile::new()?;
        let console_path = console.path().to_path_buf();
        let exe = std::env::current_exe()?;
        let vfs_guest = self.s3.as_ref().map(|s| s.guest_path.clone());
        let payload = build_payload(cmd, vfs_guest.as_deref());
        let b64 = base64::engine::general_purpose::STANDARD.encode(payload);

        let mut child = Command::new(exe);
        child
            .env(BOOT_ENV, "1")
            .env("AILOY_KRUN_KERNEL", &self.kernel)
            .env("AILOY_KRUN_ROOTFS", &self.rootfs)
            .env("AILOY_KRUN_UPPER", &self.upper)
            .env("AILOY_KRUN_CONSOLE", &console_path)
            .env("AILOY_KRUN_B64", &b64);
        if let Some(s3) = &self.s3 {
            child
                .env("AILOY_KRUN_S3_BUCKET", &s3.bucket)
                .env("AILOY_KRUN_S3_REGION", &s3.region)
                .env("AILOY_KRUN_S3_KEY", &s3.access_key_id)
                .env("AILOY_KRUN_S3_SECRET", &s3.secret_access_key)
                .env("AILOY_KRUN_S3_GUEST", &s3.guest_path);
            if let Some(ep) = &s3.endpoint {
                child.env("AILOY_KRUN_S3_ENDPOINT", ep);
            }
            if let Some(p) = &s3.key_prefix {
                child.env("AILOY_KRUN_S3_PREFIX", p);
            }
        }

        let status = child.status()?;
        let raw = std::fs::read_to_string(&console_path).unwrap_or_default();
        Ok(parse_output(&raw, status.code()))
    }
}

/// Single-quote a shell word so arbitrary characters survive.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Build a shell command line from a program + args.
fn shell_join(program: &str, args: &[String]) -> String {
    let mut cmd = shell_quote(program);
    for a in args {
        cmd.push(' ');
        cmd.push_str(&shell_quote(a));
    }
    cmd
}

/// The krun sandbox as an ailoy exec backend: each `exec` boots a fresh microVM
/// (with the cortex VFS mounted) and captures the command's output. No
/// persistent VM, no agentd — just `get_os` + `exec`; `Console`'s `read`/`write`/
/// `exec_shell` defaults ride on top.
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
        // `exec_shell` (and thus read/write) arrive as ("sh", ["-c", script]);
        // pass the script through verbatim. Direct exec shell-escapes its argv.
        let cmd = if program == "sh" && args.len() == 2 && args[0] == "-c" {
            args[1].clone()
        } else {
            shell_join(&program, &args)
        };
        // Serialize boots — they share one upper disk.
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

/// Guest program: mount the upper at `/data` (formatting on first use) and the
/// cortex VFS (if present) at `vfs_guest`, then run the command between markers,
/// syncing before shutdown so upper writes persist.
fn build_payload(cmd: &str, vfs_guest: Option<&str>) -> String {
    let vfs_mount = match vfs_guest {
        Some(g) => format!("mkdir -p {g}\nmount -t virtiofs {VFS_TAG} {g} 2>/dev/null\n"),
        None => String::new(),
    };
    format!(
        "mkdir -p /data\n\
         mount /dev/vda /data 2>/dev/null || ( (mkfs.ext4 -F -q /dev/vda || mkfs.vfat /dev/vda) >/dev/null 2>&1; sync; mount /dev/vda /data 2>/dev/null )\n\
         {vfs_mount}\
         echo {OUT_MARKER}\n\
         {cmd}\n\
         __ailoy_rc=$?\n\
         sync\n\
         umount /data 2>/dev/null\n\
         echo {RC_MARKER}$__ailoy_rc\n"
    )
}

/// Child entry point. If [`BOOT_ENV`] is set, boot the configured VM (mounting
/// the cortex VFS if S3 env is present) and never return. Consuming binaries
/// must call this at the top of `main()`.
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
    let script = format!("echo {b64} | base64 -d > /run.sh; sh /run.sh");

    let mut builder = VmBuilder::new()
        .machine(|m| m.vcpus(1).memory_mib(512))
        .kernel(|k| k.krunfw_path(&kernel))
        .fs(|fs| fs.root(&rootfs))
        .disk(|d| d.path(&upper).format(DiskImageFormat::Raw))
        .console(|c| c.output(&console));

    // Mount the cortex VFS as a second virtio-fs device if S3 config was passed.
    if let Ok(bucket) = std::env::var("AILOY_KRUN_S3_BUCKET") {
        let cfg = S3Config {
            bucket,
            region: std::env::var("AILOY_KRUN_S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
            access_key_id: get("AILOY_KRUN_S3_KEY"),
            secret_access_key: get("AILOY_KRUN_S3_SECRET"),
            endpoint: std::env::var("AILOY_KRUN_S3_ENDPOINT").ok(),
            key_prefix: std::env::var("AILOY_KRUN_S3_PREFIX").ok(),
        };
        match S3Volume::new(&cfg) {
            Ok(vol) => {
                let adapter = PosixAdapter::new(vol);
                builder =
                    builder.fs(move |fs| fs.tag(VFS_TAG).custom(Box::new(adapter)));
            }
            Err(e) => {
                eprintln!("ailoy krun: S3 VFS connect failed: {e}");
                std::process::exit(1);
            }
        }
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
