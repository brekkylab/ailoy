//! Microsandbox-backed [`Machine`] implementation.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Context as _;
use async_trait::async_trait;
use microsandbox::{
    ExecOutput, MicrosandboxError, NetworkPolicy, Sandbox as MsbSandbox, SandboxConfig, Snapshot,
    sandbox::{ExecOptionsBuilder, IntoImage, MountBuilder, PullPolicy},
    snapshot::ExportOpts,
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

impl Default for SandboxBuilder {
    fn default() -> Self {
        let mut config = SandboxConfig::default();
        // 8 random bytes hex-encoded = 16 hex chars, ~64 bits of entropy. Short
        // enough to fit any reasonable socket-path budget.
        config.name = format!("ailoy-{}", hex::encode(&Uuid::new_v4().as_bytes()[..8]));
        // `"ubuntu:latest"` is a stable OCI reference; conversion never fails.
        config.image = "ubuntu:latest"
            .into_rootfs_source()
            .expect("'ubuntu:latest' parses as an OCI image reference");
        config.cpus = 2;
        config.memory_mib = 2048;
        config.workdir = Some("/root".to_string());
        config.pull_policy = PullPolicy::IfMissing;
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
        self.config.name = name.into();
        self
    }

    pub fn image(mut self, image: impl IntoImage) -> Self {
        match image.into_rootfs_source() {
            Ok(rfs) => self.config.image = rfs,
            Err(e) => {
                if self.build_error.is_none() {
                    self.build_error = Some(format!("invalid image: {e}"));
                }
            }
        }
        self
    }

    pub fn cpus(mut self, cpus: u8) -> Self {
        self.config.cpus = cpus;
        self
    }

    pub fn memory_mib(mut self, memory_mib: u32) -> Self {
        self.config.memory_mib = memory_mib;
        self
    }

    pub fn workdir(mut self, workdir: impl Into<String>) -> Self {
        self.config.workdir = Some(workdir.into());
        self
    }

    pub fn env(mut self, env: impl IntoIterator<Item = (String, String)>) -> Self {
        self.config.env = env.into_iter().collect();
        self
    }

    pub fn disable_network(mut self, disable: bool) -> Self {
        if disable {
            self.config.network.enabled = false;
            self.config.network.policy = NetworkPolicy::none();
        }
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
            } => {
                let b = MountBuilder::new(guest).named(name);
                if readonly { b.readonly() } else { b }
            }
            VolumeMount::Tmpfs { guest, size_mib } => {
                let b = MountBuilder::new(guest).tmpfs();
                if let Some(s) = size_mib { b.size(s) } else { b }
            }
        };
        match builder.build() {
            Ok(vm) => self.config.mounts.push(vm),
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
        let name = config.name.clone();
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
    /// The intermediate microsandbox snapshot directory unpacked by this
    /// call is cleaned up before returning (success or failure).
    pub async fn try_from_archive(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        ensure_msb().await?;

        let handle = Snapshot::import(path.as_ref(), None)
            .await
            .context("import snapshot archive")?;
        let snap = handle.open().await.context("open imported snapshot")?;
        let snap_path = snap.path().to_path_buf();

        let name = snap_path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                anyhow::anyhow!("snapshot path has no file name: {}", snap_path.display())
            })?
            .to_string();

        let result = microsandbox::Sandbox::builder(&name)
            .from_snapshot(snap_path.to_string_lossy().into_owned())
            .pull_policy(PullPolicy::IfMissing)
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
            .name(&self.name)
            .create()
            .await
            .context("create snapshot")?;
        let snap_path = snap.path().to_path_buf();

        let result = Snapshot::export(
            snap_path.to_string_lossy().as_ref(),
            path.as_ref(),
            ExportOpts::default(),
        )
        .await
        .context("export snapshot archive");

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
        if let Some(console) = self.console.take() {
            console.inner.stop_and_wait().await?;
        }
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
            let _ = rx.recv();
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
            handle.status(),
            SandboxStatus::Running,
            "vm should be Running right after build",
        );

        sandbox.stop().await.expect("stop");

        let handle = MsbSandbox::get(&name)
            .await
            .expect("vm record should still exist after stop");
        assert_eq!(
            handle.status(),
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
