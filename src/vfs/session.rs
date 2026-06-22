use std::sync::Arc;

use crate::runenv::{RunEnv, RunEnvHandle};
use crate::vfs::{VfsForward, VfsMount, bootstrap_guest_forwarder};

/// Opaque per-agent VFS handle. The agent holds one of these for its lifetime;
/// dropping it tears everything down (unmounts the host FUSE, or stops the
/// sandbox VM and aborts the forward server). Both variants present the same
/// surface to the agent — only [`ensure_mounted`](AgentVfs::ensure_mounted)
/// and `Drop` differ internally.
pub enum AgentVfs {
    /// Non-sandbox: a host FUSE mount, already mounted at build time.
    Host(VfsMount),
    /// Sandbox: a host forward server plus a one-time in-guest mount.
    Sandbox(SandboxVfs),
}

pub struct SandboxVfs {
    /// Held for the session; aborts on drop.
    _forward: VfsForward,
    /// Consumed on the first `ensure_mounted`.
    bootstrap: Option<Bootstrap>,
    /// Strong runenv handle kept after mounting so the VM (and the in-guest
    /// forwarder process / mount) stays up across tool calls. Dropping the last
    /// handle stops the VM, which would kill the mount.
    _handle: Option<Arc<RunEnvHandle>>,
}

struct Bootstrap {
    mount_root: String,
    port: u16,
    token: String,
}

impl AgentVfs {
    pub fn host(mount: VfsMount) -> Self {
        AgentVfs::Host(mount)
    }

    pub fn sandbox(forward: VfsForward, mount_root: String, port: u16, token: String) -> Self {
        AgentVfs::Sandbox(SandboxVfs {
            _forward: forward,
            bootstrap: Some(Bootstrap {
                mount_root,
                port,
                token,
            }),
            _handle: None,
        })
    }

    /// Ensure the VFS is mounted in the agent's runenv. Host mounts are already
    /// up at build time (no-op); sandbox mounts are bootstrapped once on the
    /// first call, after which the runenv handle is held to keep the VM alive.
    pub async fn ensure_mounted(&mut self, runenv: &RunEnv) -> anyhow::Result<()> {
        match self {
            AgentVfs::Host(_) => Ok(()),
            AgentVfs::Sandbox(s) => {
                let Some(b) = s.bootstrap.take() else {
                    return Ok(());
                };
                let handle = runenv.get().await?;
                bootstrap_guest_forwarder(&handle, &b.mount_root, b.port, &b.token).await?;
                s._handle = Some(handle);
                Ok(())
            }
        }
    }
}
