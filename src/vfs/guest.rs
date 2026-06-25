use std::path::Path;

use crate::runenv::RunEnvHandle;

/// In-guest Python FUSE forwarder source, shipped as a data asset and written
/// into the guest at boot. Arch-independent.
const FORWARDER_SRC: &str = include_str!("assets/guest_fuse_fwd.py");
const GUEST_FWD_PATH: &str = "/opt/ailoy/vfs_fwd.py";

/// Inject and start the in-guest FUSE forwarder, mounting the VFS at
/// `mount_root` inside the sandbox. The forwarder talks to the host forward
/// server via `host.microsandbox.internal:<port>` (requires the sandbox to be
/// created with `allow_host_egress = true`). Blocks until the mount appears.
pub async fn bootstrap_guest_forwarder(
    handle: &RunEnvHandle,
    mount_root: &str,
    port: u16,
    token: &str,
) -> anyhow::Result<()> {
    handle
        .write(Path::new(GUEST_FWD_PATH), FORWARDER_SRC.as_bytes())
        .await
        .map_err(|e| anyhow::anyhow!("write guest forwarder: {e}"))?;

    let script = format!(
        r#"set -e
mkdir -p {mount_root} /opt/ailoy
# Idempotent: unmount any stale/defunct mount left by a previous boot or attach
# so the fresh mount points at the current host server (port/token below).
# Unmounting also detaches any surviving forwarder process from the mountpoint.
# NOTE: do NOT pkill by the forwarder path — this script's own `sh -c` argv
# contains that path, so `pkill -f <path>` would SIGTERM the bootstrap itself.
fusermount3 -u {mount_root} 2>/dev/null || umount -l {mount_root} 2>/dev/null || true
if ! command -v python3 >/dev/null 2>&1 || ! command -v fusermount3 >/dev/null 2>&1; then
  # Best-effort install for non-baked images, ONLY on the first mount of a fresh
  # sandbox (deps then persist on the rootfs, so restarts skip this). Each apt is
  # bounded by `timeout` so a slow mirror or stuck lock fails fast instead of
  # hanging to the exec timeout; DPkg::Lock::Timeout waits out boot-time
  # apt-daily; retried twice for transient contention.
  # Production should bake python3/fuse3/mfusepy into the guest image to skip
  # this entirely (fast, offline, deterministic).
  for _ in 1 2; do
    timeout 45 apt-get -o DPkg::Lock::Timeout=40 update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive timeout 80 apt-get -o DPkg::Lock::Timeout=40 \
      install -y -qq python3 python3-pip fuse3 >/dev/null 2>&1 || true
    command -v python3 >/dev/null 2>&1 && command -v fusermount3 >/dev/null 2>&1 && break
  done
fi
python3 -c 'import mfusepy' 2>/dev/null || pip3 install --break-system-packages -q mfusepy >/dev/null 2>&1 || true
export VFS_HOST="http://host.microsandbox.internal:{port}"
export VFS_TOKEN="{token}"
# Detach into its own session so it survives this exec session teardown.
setsid sh -c 'python3 {GUEST_FWD_PATH} {mount_root} </dev/null >/tmp/ailoy-vfs.log 2>&1' </dev/null >/dev/null 2>&1 &
for _ in $(seq 1 80); do
  if grep -q " {mount_root} " /proc/mounts 2>/dev/null; then exit 0; fi
  sleep 0.25
done
echo "ailoy vfs: mount did not appear at {mount_root}" >&2
cat /tmp/ailoy-vfs.log >&2 2>/dev/null || true
exit 1
"#
    );

    let out = handle
        .exec_shell(script, Some(300))
        .await
        .map_err(|e| anyhow::anyhow!("exec guest vfs setup: {e}"))?;
    if out.exit_code != 0 {
        anyhow::bail!(
            "guest vfs mount failed (exit {}): {}",
            out.exit_code,
            out.stderr.trim()
        );
    }
    Ok(())
}
