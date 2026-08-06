use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::{Console, ExecResult};

/// `Console` that runs commands directly on the host — the default exec backend.
///
/// With the `local-fuse` feature it can instead run *inside* a cortex
/// [`Workspace`](cortex::Workspace): like the sandbox, it holds a
/// [`WorkspaceSpec`](cortex::WorkspaceSpec) (see [`with_workspace`](Self::with_workspace))
/// rather than an already-open mount, and realizes it — a host-FUSE mount — lazily
/// on first use. Every command, read, and write then sees the same unified tree a
/// sandboxed agent would, without a VM.
pub struct LocalConsole {
    /// The workspace to run in, held declaratively as `(mountpoint, spec)` — the
    /// same shape the sandbox keeps. `None` is the plain host console (process
    /// cwd). The mount below is realized from this on first use.
    #[cfg(feature = "local-fuse")]
    workspace: Option<(PathBuf, cortex::WorkspaceSpec)>,

    /// The realized host-FUSE mount, built from `workspace` on first use and held
    /// for the console's lifetime (unmounts on drop). Interior-mutable because the
    /// `Console` methods take `&self`.
    #[cfg(feature = "local-fuse")]
    mount: std::sync::Mutex<Option<cortex::CortexMount>>,
}

impl Default for LocalConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalConsole {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "local-fuse")]
            workspace: None,
            #[cfg(feature = "local-fuse")]
            mount: std::sync::Mutex::new(None),
        }
    }

    /// The directory exec/read/write resolve relative paths under: the workspace
    /// mountpoint when one is configured, else `None` (the process cwd).
    fn root(&self) -> Option<&Path> {
        #[cfg(feature = "local-fuse")]
        {
            return self.workspace.as_ref().map(|(mp, _)| mp.as_path());
        }
        #[cfg(not(feature = "local-fuse"))]
        None
    }

    /// Resolve a request path: relative paths are taken under [`root`](Self::root)
    /// when one is set; absolute paths and the unmounted case pass through.
    fn resolve(&self, path: &Path) -> PathBuf {
        match self.root() {
            Some(root) if path.is_relative() => root.join(path),
            _ => path.to_path_buf(),
        }
    }

    /// Realize the configured workspace as a host-FUSE mount, once. A no-op when
    /// there is no workspace or it is already mounted. Every `Console` op calls
    /// this first, so the mount comes up on the first command/read/write.
    fn ensure_mounted(&self) -> anyhow::Result<()> {
        #[cfg(feature = "local-fuse")]
        {
            let Some((mp, spec)) = &self.workspace else {
                return Ok(());
            };
            let mut guard = self.mount.lock().unwrap();
            if guard.is_some() {
                return Ok(());
            }
            std::fs::create_dir_all(mp)
                .map_err(|e| anyhow::anyhow!("create mountpoint {}: {e}", mp.display()))?;
            let ws = cortex::Workspace::from_spec(spec)
                .map_err(|e| anyhow::anyhow!("build workspace: {e}"))?;
            let m = cortex::CortexMount::spawn(cortex::PosixFs::new(ws), mp)
                .map_err(|e| anyhow::anyhow!("host-mount workspace at {}: {e}", mp.display()))?;
            *guard = Some(m);
        }
        Ok(())
    }
}

#[cfg(feature = "local-fuse")]
impl LocalConsole {
    /// Run inside a cortex [`WorkspaceSpec`](cortex::WorkspaceSpec), host-mounted
    /// over FUSE at `mountpoint`. Mirrors the sandbox's
    /// [`with_workspace`](super::Sandbox::with_workspace): the console *holds the
    /// spec* and realizes the mount lazily (on the first command/read/write),
    /// tearing it down when dropped.
    ///
    /// The mount needs a libfuse provider (macFUSE / FUSE-T on macOS, `/dev/fuse`
    /// on Linux) — the `local-fuse` feature's build-time requirement.
    pub fn with_workspace(
        mut self,
        mountpoint: impl Into<PathBuf>,
        spec: cortex::WorkspaceSpec,
    ) -> Self {
        self.workspace = Some((mountpoint.into(), spec));
        self
    }
}

#[async_trait]
impl Console for LocalConsole {
    fn get_os(&self) -> &str {
        std::env::consts::OS
    }

    async fn exec(
        &self,
        program: String,
        args: Vec<String>,
        timeout: Option<u64>,
    ) -> anyhow::Result<ExecResult> {
        self.ensure_mounted()?;
        let mut command = tokio::process::Command::new(program);
        command.args(args).kill_on_drop(true);
        // Run inside the mounted workspace when there is one, so relative paths
        // and a bare `pwd` resolve to the same tree read/write see.
        if let Some(root) = self.root() {
            command.current_dir(root);
        }
        let result = if let Some(secs) = timeout {
            tokio::time::timeout(std::time::Duration::from_secs(secs), command.output()).await
        } else {
            Ok(command.output().await)
        };
        match result {
            Ok(Ok(out)) => Ok(ExecResult {
                stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
                exit_code: out.status.code().unwrap_or(-1),
                timed_out: false,
            }),
            Ok(Err(e)) => Ok(ExecResult {
                stdout: String::new(),
                stderr: e.to_string(),
                exit_code: -1,
                timed_out: false,
            }),
            Err(_) => Ok(ExecResult {
                stdout: String::new(),
                stderr: String::new(),
                exit_code: -1,
                timed_out: true,
            }),
        }
    }

    async fn get_cwd(&self) -> anyhow::Result<PathBuf> {
        if let Some(root) = self.root() {
            return Ok(root.to_path_buf());
        }
        std::env::current_dir().map_err(|e| anyhow::anyhow!("get_cwd: {e}"))
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        self.ensure_mounted()?;
        let real = self.resolve(path);
        tokio::fs::read(&real)
            .await
            .map_err(|e| anyhow::anyhow!("read {}: {e}", real.display()))
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
        self.ensure_mounted()?;
        let real = self.resolve(path);
        if let Some(parent) = real.parent()
            && !parent.as_os_str().is_empty()
        {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| anyhow::anyhow!("write {}: mkdir parent: {e}", real.display()))?;
        }
        tokio::fs::write(&real, content)
            .await
            .map_err(|e| anyhow::anyhow!("write {}: {e}", real.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(cmd: &str) -> (String, Vec<String>) {
        ("sh".to_string(), vec!["-c".to_string(), cmd.to_string()])
    }

    #[tokio::test]
    async fn test_exec_stdout() {
        let local = LocalConsole::new();
        let console = &local;
        let (prog, args) = sh("echo hello");
        let result = console.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_exec_exit_code() {
        let local = LocalConsole::new();
        let console = &local;
        let (prog, args) = sh("exit 42");
        let result = console.exec(prog, args, None).await.unwrap();
        assert_eq!(result.exit_code, 42);
    }

    #[tokio::test]
    async fn test_exec_stderr() {
        let local = LocalConsole::new();
        let console = &local;
        let (prog, args) = sh("echo err >&2");
        let result = console.exec(prog, args, None).await.unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn test_exec_timeout() {
        let local = LocalConsole::new();
        let console = &local;
        let (prog, args) = sh("sleep 10");
        let result = console.exec(prog, args, Some(1)).await.unwrap();
        assert!(result.timed_out);
    }

}

#[cfg(all(test, feature = "local-fuse"))]
mod fuse_tests {
    use super::*;
    use cortex::{VolumeSpec, WorkspaceSpec};

    /// A `LocalConsole` mounted over FUSE runs inside a cortex `Workspace`: its
    /// shell reads workspace files and its writes land on the backing source.
    /// Ignored — needs a libfuse provider (macFUSE/FUSE-T) and a real mount:
    ///
    ///   PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cargo test -p ailoy \
    ///     --features local-fuse local_console_mount -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "needs a libfuse provider + a real host mount"]
    async fn local_console_mount_sees_and_writes_the_workspace() {
        let pid = std::process::id();
        let src = std::env::temp_dir().join(format!("ailoy-lf-src-{pid}"));
        let mp = std::env::temp_dir().join(format!("ailoy-lf-mp-{pid}"));
        let _ = std::fs::remove_dir_all(&src);
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("hello.txt"), b"HELLO_LOCAL_FUSE").unwrap();

        let spec =
            WorkspaceSpec::default().mount("files", VolumeSpec::Local { host: src.clone() });
        // Holds the spec; the mount comes up lazily on the first command below.
        let console = LocalConsole::new().with_workspace(&mp, spec);

        // The shell sees the workspace file through the mount.
        let out = console
            .exec_shell("cat files/hello.txt".to_string(), Some(20))
            .await
            .unwrap();
        assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
        assert!(out.stdout.contains("HELLO_LOCAL_FUSE"), "stdout: {}", out.stdout);

        // A write through the console lands on the backing source dir.
        console
            .write(Path::new("files/new.txt"), b"WROTE_VIA_MOUNT")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read(src.join("new.txt")).unwrap(),
            b"WROTE_VIA_MOUNT"
        );

        drop(console); // unmounts
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&mp);
    }
}
