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
    /// Endpoint the in-guest forwarder is (re)pointed at on every (re)mount.
    bootstrap: Bootstrap,
    /// Strong runenv handle held while mounted so the VM (and the in-guest
    /// forwarder / mount) stays up across tool calls. Reacquired on each
    /// `ensure_mounted`, which restarts the VM if it was stopped since the
    /// previous attach.
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
            bootstrap: Bootstrap {
                mount_root,
                port,
                token,
            },
            _handle: None,
        })
    }

    /// Ensure the VFS is mounted in the agent's runenv.
    ///
    /// Host mounts are up at build time (no-op). Sandbox mounts live inside the
    /// guest VM, which is stopped whenever no handle is held — so the in-guest
    /// forwarder cannot outlive an agent runtime (a frequent pattern: build →
    /// use → drop → rebuild against the same persisted sandbox). On each call
    /// this reacquires the VM handle (restarting the VM if it was stopped) and,
    /// when the mount is not live, (re)bootstraps the forwarder against the
    /// current host server. Bootstrap is idempotent, so a stale forwarder/mount
    /// left by a previous boot is torn down first.
    pub async fn ensure_mounted(&mut self, runenv: &RunEnv) -> anyhow::Result<()> {
        let AgentVfs::Sandbox(s) = self else {
            return Ok(());
        };
        let handle = runenv.get().await?;
        // `get()` can return before the guest agent accepts exec (a freshly
        // started VM); microsandbox maps an exec against a not-ready guest to an
        // immediate timeout. Wait for readiness so the probe/bootstrap below
        // don't spuriously fail.
        wait_exec_ready(handle.as_ref()).await;
        if !mount_is_live(handle.as_ref(), &s.bootstrap.mount_root).await {
            bootstrap_guest_forwarder(
                &handle,
                &s.bootstrap.mount_root,
                s.bootstrap.port,
                &s.bootstrap.token,
            )
            .await?;
        }
        s._handle = Some(handle);
        Ok(())
    }
}

/// Block until the guest accepts exec (a just-started VM may not yet), so the
/// first real exec doesn't fail with a spurious timeout. Gives up after ~10s
/// and lets the caller proceed (and surface a real error if still unready).
async fn wait_exec_ready(handle: &RunEnvHandle) {
    for _ in 0..40 {
        let ready = handle
            .exec_shell("true".into(), Some(10))
            .await
            .map(|o| o.exit_code == 0)
            .unwrap_or(false);
        if ready {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    }
}

/// Whether the in-guest FUSE mount is present and connected. A VM stopped /
/// resumed / recreated since the last attach reports it absent (the forwarder
/// process is gone, or the mountpoint is a defunct endpoint), which triggers a
/// re-mount.
async fn mount_is_live(handle: &RunEnvHandle, mount_root: &str) -> bool {
    handle
        .exec_shell(format!("mountpoint -q {mount_root}"), Some(10))
        .await
        .map(|o| o.exit_code == 0)
        .unwrap_or(false)
}
