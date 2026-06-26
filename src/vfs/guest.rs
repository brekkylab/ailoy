use std::path::Path;

use crate::runenv::RunEnvHandle;

/// Static, dependency-free in-guest FUSE forwarder binaries (one per guest
/// arch), shipped as data assets. The guest arch equals the host arch under
/// libkrun (same-arch virtualization), so the host picks by its own arch.
/// Source: `tools/vfs-forwarder`.
const FWD_BIN_AARCH64: &[u8] = include_bytes!("assets/ailoy-vfs-fwd.aarch64");
const FWD_BIN_X86_64: &[u8] = include_bytes!("assets/ailoy-vfs-fwd.x86_64");
const GUEST_FWD_BIN: &str = "/opt/ailoy/vfs-fwd";

/// The static forwarder binary matching the (guest == host) arch, if shipped.
fn static_forwarder() -> Option<&'static [u8]> {
    match std::env::consts::ARCH {
        "aarch64" => Some(FWD_BIN_AARCH64),
        "x86_64" => Some(FWD_BIN_X86_64),
        _ => None,
    }
}

/// Inject and start the in-guest FUSE forwarder, mounting the VFS at
/// `mount_root` inside the sandbox. The forwarder talks to the host forward
/// server via `host.microsandbox.internal:<port>` (requires the sandbox to be
/// created with `allow_host_egress = true`). Blocks until the mount appears.
///
/// Uses the static dependency-free binary (no python/fuse3/apt — mounts
/// `/dev/fuse` directly as root). The guest arch equals the host arch under
/// libkrun, and a binary ships for every supported arch (aarch64, x86_64).
pub async fn bootstrap_guest_forwarder(
    handle: &RunEnvHandle,
    mount_root: &str,
    port: u16,
    token: &str,
) -> anyhow::Result<()> {
    let bin = static_forwarder().ok_or_else(|| {
        anyhow::anyhow!(
            "no static vfs forwarder shipped for guest arch '{}'",
            std::env::consts::ARCH
        )
    })?;
    try_static_forwarder(handle, mount_root, port, token, bin).await
}

/// Deploy + run the static binary; mount via direct `/dev/fuse` (no deps).
async fn try_static_forwarder(
    handle: &RunEnvHandle,
    mount_root: &str,
    port: u16,
    token: &str,
    bin: &[u8],
) -> anyhow::Result<()> {
    handle
        .write(Path::new(GUEST_FWD_BIN), bin)
        .await
        .map_err(|e| anyhow::anyhow!("write static forwarder: {e}"))?;

    let script = format!(
        r#"set -e
mkdir -p /opt/ailoy
# Detach any stale/defunct mount from a previous boot/attach FIRST — including a
# dead-daemon mount (forwarder crashed/killed) left in the "Transport endpoint is
# not connected" state. `mkdir`/stat on such a mountpoint errors, so clearing it
# before touching {mount_root} is what lets a crashed forwarder be re-mounted.
umount -l {mount_root} 2>/dev/null || fusermount3 -u {mount_root} 2>/dev/null || true
mkdir -p {mount_root}
chmod +x {GUEST_FWD_BIN}
export VFS_HOST="http://host.microsandbox.internal:{port}"
export VFS_TOKEN="{token}"
# Detach into its own session so it survives this exec session teardown.
setsid sh -c '{GUEST_FWD_BIN} {mount_root} </dev/null >/tmp/ailoy-vfs.log 2>&1' </dev/null >/dev/null 2>&1 &
for _ in $(seq 1 100); do
  if grep -q " {mount_root} " /proc/mounts 2>/dev/null; then exit 0; fi
  sleep 0.1
done
echo "static forwarder: mount did not appear at {mount_root}" >&2
cat /tmp/ailoy-vfs.log >&2 2>/dev/null || true
exit 1
"#
    );
    let out = handle
        .exec_shell(script, Some(30))
        .await
        .map_err(|e| anyhow::anyhow!("exec static forwarder: {e}"))?;
    if out.exit_code != 0 {
        anyhow::bail!("static forwarder mount failed (exit {}): {}", out.exit_code, out.stderr.trim());
    }
    Ok(())
}
