use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::{Console, ExecResult};

/// `Console` that runs commands directly on the host — the default exec backend.
///
/// With the `local-fuse` feature it can instead run *inside* a cortex
/// [`Workspace`](cortex::Workspace) host-mounted over FUSE (see
/// [`mounting`](Self::mounting)): every command, read, and write then sees the
/// same unified tree a sandboxed agent would, without a VM.
pub struct LocalConsole {
    /// Base directory exec/read/write resolve relative paths under. `None` means
    /// the process's own working directory (the plain host console).
    root: Option<PathBuf>,

    /// A host-FUSE mount of a cortex workspace, held so it stays mounted for the
    /// console's lifetime (it unmounts on drop). `root` points at its mountpoint.
    #[cfg(feature = "local-fuse")]
    _mount: Option<cortex::CortexMount>,
}

impl Default for LocalConsole {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalConsole {
    pub fn new() -> Self {
        Self {
            root: None,
            #[cfg(feature = "local-fuse")]
            _mount: None,
        }
    }

    /// Resolve a request path: relative paths are taken under [`root`](Self::root)
    /// (the mount) when one is set; absolute paths and the unmounted case pass
    /// through unchanged.
    fn resolve(&self, path: &Path) -> PathBuf {
        match &self.root {
            Some(root) if path.is_relative() => root.join(path),
            _ => path.to_path_buf(),
        }
    }
}

#[cfg(feature = "local-fuse")]
impl LocalConsole {
    /// Build a cortex [`Workspace`](cortex::Workspace) from `spec`, host-mount it
    /// over FUSE at `mountpoint`, and run every command/read/write under it. The
    /// mount is torn down when the returned console is dropped.
    ///
    /// `mountpoint` is created if absent. Needs a libfuse provider (macFUSE /
    /// FUSE-T on macOS, `/dev/fuse` on Linux) — the `local-fuse` feature's
    /// build-time requirement.
    pub fn mounting(
        spec: &cortex::WorkspaceSpec,
        mountpoint: impl Into<PathBuf>,
    ) -> anyhow::Result<Self> {
        let mp = mountpoint.into();
        std::fs::create_dir_all(&mp)
            .map_err(|e| anyhow::anyhow!("create mountpoint {}: {e}", mp.display()))?;
        let ws = cortex::Workspace::from_spec(spec)
            .map_err(|e| anyhow::anyhow!("build workspace: {e}"))?;
        let mount = cortex::CortexMount::spawn(cortex::PosixFs::new(ws), &mp)
            .map_err(|e| anyhow::anyhow!("host-mount workspace at {}: {e}", mp.display()))?;
        Ok(Self {
            root: Some(mp),
            _mount: Some(mount),
        })
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
        let mut command = tokio::process::Command::new(program);
        command.args(args).kill_on_drop(true);
        // Run inside the mounted workspace when there is one, so relative paths
        // and a bare `pwd` resolve to the same tree read/write see.
        if let Some(root) = &self.root {
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
        if let Some(root) = &self.root {
            return Ok(root.clone());
        }
        std::env::current_dir().map_err(|e| anyhow::anyhow!("get_cwd: {e}"))
    }

    async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        let real = self.resolve(path);
        tokio::fs::read(&real)
            .await
            .map_err(|e| anyhow::anyhow!("read {}: {e}", real.display()))
    }

    async fn write(&self, path: &Path, content: &[u8]) -> anyhow::Result<()> {
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
        let console = LocalConsole::mounting(&spec, &mp).expect("host-mount workspace");

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
