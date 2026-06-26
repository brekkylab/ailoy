use crate::runenv::RunEnv;
use crate::vfs::VfsMount;
#[cfg(feature = "sandbox")]
use crate::vfs::sandbox::SandboxVfs;

/// Opaque per-agent VFS handle. The agent holds one of these for its lifetime;
/// dropping it tears everything down (unmounts the host FUSE, or — for sandbox —
/// stops the VM and aborts the forward server). Both variants present the same
/// surface to the agent — only [`ensure_mounted`](AgentVfs::ensure_mounted) and
/// `Drop` differ internally.
pub enum AgentVfs {
    /// Non-sandbox: a host FUSE mount, already mounted at build time.
    Host(VfsMount),
    /// Sandbox: a host forward server plus a one-time in-guest mount. Only
    /// available with the `sandbox` feature.
    #[cfg(feature = "sandbox")]
    Sandbox(SandboxVfs),
}

impl AgentVfs {
    pub fn host(mount: VfsMount) -> Self {
        AgentVfs::Host(mount)
    }

    #[cfg(feature = "sandbox")]
    pub fn sandbox(
        forward: crate::vfs::sandbox::VfsForward,
        mount_root: String,
        port: u16,
        token: String,
    ) -> Self {
        AgentVfs::Sandbox(SandboxVfs::new(forward, mount_root, port, token))
    }

    /// Ensure the VFS is mounted in the agent's runenv.
    ///
    /// Host mounts are up at build time (no-op). Sandbox mounts live inside the
    /// guest VM, which is stopped whenever no handle is held — so the in-guest
    /// forwarder cannot outlive an agent runtime (a frequent pattern: build →
    /// use → drop → rebuild against the same persisted sandbox). For the sandbox
    /// variant this reacquires the VM handle and (re)bootstraps the forwarder as
    /// needed; see [`SandboxVfs::ensure_mounted`].
    pub async fn ensure_mounted(&mut self, runenv: &RunEnv) -> anyhow::Result<()> {
        #[cfg(feature = "sandbox")]
        if let AgentVfs::Sandbox(s) = self {
            return s.ensure_mounted(runenv).await;
        }
        #[cfg(not(feature = "sandbox"))]
        let _ = runenv;
        Ok(())
    }
}
