//! Sandbox-only VFS frontend.
//!
//! When the agent runs in a microsandbox guest, the [`Vfs`](crate::vfs::Vfs)
//! core stays on the host and is exposed to the guest by two cooperating pieces,
//! both compiled only with the `sandbox` feature:
//!
//! - [`VfsForward`] — a tiny host-side HTTP server the in-guest forwarder calls
//!   over `allow@host` egress.
//! - [`bootstrap_guest_forwarder`] — deploys and starts the static in-guest FUSE
//!   forwarder, mounting the VFS inside the guest.
//!
//! [`SandboxVfs`] ties them together for an agent's lifetime and re-establishes
//! the in-guest mount on each attach (the VM stops while idle, so the forwarder
//! cannot outlive an agent runtime).

use crate::runenv::{RunEnv, RunEnvHandle};

mod forward;
mod guest;

pub use forward::VfsForward;
pub use guest::bootstrap_guest_forwarder;

/// Per-agent sandbox VFS state: a host forward server plus a one-time (per
/// attach) in-guest mount.
pub struct SandboxVfs {
    /// Held for the session; aborts on drop.
    _forward: VfsForward,
    /// Endpoint the in-guest forwarder is (re)pointed at on every (re)mount.
    bootstrap: Bootstrap,
    /// Strong runenv handle held while mounted so the VM (and the in-guest
    /// forwarder / mount) stays up across tool calls. Reacquired on each
    /// `ensure_mounted`, which restarts the VM if it was stopped since the
    /// previous attach.
    _handle: Option<std::sync::Arc<RunEnvHandle>>,
}

struct Bootstrap {
    mount_root: String,
    port: u16,
    token: String,
}

impl SandboxVfs {
    pub fn new(forward: VfsForward, mount_root: String, port: u16, token: String) -> Self {
        SandboxVfs {
            _forward: forward,
            bootstrap: Bootstrap {
                mount_root,
                port,
                token,
            },
            _handle: None,
        }
    }

    /// Reacquire the VM handle (restarting the VM if it was stopped) and, when
    /// the mount is not live, (re)bootstrap the forwarder against the current
    /// host server. Bootstrap is idempotent, so a stale forwarder/mount left by
    /// a previous boot is torn down first.
    pub async fn ensure_mounted(&mut self, runenv: &RunEnv) -> anyhow::Result<()> {
        let handle = runenv.get().await?;
        // `get()` can return before the guest agent accepts exec (a freshly
        // started VM); microsandbox maps an exec against a not-ready guest to an
        // immediate timeout. Wait for readiness so the probe/bootstrap below
        // don't spuriously fail.
        wait_exec_ready(handle.as_ref()).await;
        if !mount_is_live(handle.as_ref(), &self.bootstrap.mount_root).await {
            bootstrap_guest_forwarder(
                &handle,
                &self.bootstrap.mount_root,
                self.bootstrap.port,
                &self.bootstrap.token,
            )
            .await?;
        }
        self._handle = Some(handle);
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

/// Whether the in-guest FUSE mount is present AND functional. Beyond
/// `mountpoint -q` (which a stale mount left by a previous attach still passes
/// even though its forwarder points at a now-dead host server), this lists the
/// mount root — a readdir that round-trips to the current host forward server.
/// A dead/stale forwarder makes that hang or fail within the short timeout, so
/// we treat it as not-live and re-bootstrap against the current server.
async fn mount_is_live(handle: &RunEnvHandle, mount_root: &str) -> bool {
    handle
        .exec_shell(
            format!("mountpoint -q {mount_root} && ls {mount_root} >/dev/null 2>&1"),
            Some(8),
        )
        .await
        .map(|o| o.exit_code == 0 && !o.timed_out)
        .unwrap_or(false)
}
