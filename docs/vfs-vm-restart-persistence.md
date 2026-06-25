# VFS over a sandbox VM: surviving VM restarts (agent-k use case)

> Status: **solved & verified**. 2026-06-26.

## Problem

agent-k's lifecycle builds an `Agent` against a **persisted (by-name) sandbox**,
uses it, drops it, and later builds a **new** agent against the **same** sandbox.
While idle the sandbox VM is **stopped** (`RunEnvHandle::Drop` stops it; `persist`
keeps only its registration). The VFS mount inside the guest is served by an
in-guest `mfusepy` forwarder **process** — which dies when the VM stops. So the
mount cannot simply "survive"; it must be **re-established** whenever the VM
comes back. The requirement: continuous provider-filesystem access through the
VM across these restarts.

## Why reviving the same process is impossible (and what we do instead)

A FUSE mount needs a live userspace server. A VM stop is effectively a poweroff —
the process and the mount are gone, and on the next start the guest is a fresh
process space. There is nothing to "revive". So the design is:

**Re-establish the mount automatically on each agent attach.**

`AgentVfs::ensure_mounted` runs at the start of every agent `run()` and:
1. reacquires the VM handle via `RunEnv::get()` (starting the VM if stopped),
2. waits until the guest accepts exec (a just-started VM may not yet),
3. **functionally** probes the mount (`mountpoint -q && ls <root>` — a readdir
   that round-trips to the *current* host forward server; a stale mount whose
   forwarder points at a dead server fails this),
4. if not live, **(re)bootstraps** the in-guest forwarder against the current
   host forward server (idempotent: unmounts any stale/defunct mount first).

The host forward server is per-agent-session (ephemeral port + token); the
re-bootstrap injects the current endpoint each time, so a new session's guest
always talks to the live server.

## Bugs that were blocking this (all fixed)

| Symptom | Root cause | Fix |
|---|---|---|
| bootstrap dies ~1s, exit -1 | teardown `pkill -f /opt/ailoy/vfs_fwd.py` matched the bootstrap's own `sh -c '<script>'` argv → SIGTERM self | drop the pkill; unmount via `fusermount3 -u` only |
| bootstrap hangs ~300s | boot-time `apt-daily` held the dpkg lock when the first-boot apt ran | `DPkg::Lock::Timeout` + `timeout` bound + retry; **bake deps in production** |
| reconnect fails "already running" | `Sandbox::new` propagated `SandboxStillRunning` when the previous owner's async stop hadn't finished | force-stop + restart on `SandboxStillRunning` |
| read clamped to wrong size | rendered files (Notion page.json) report listing size 0 | stat verifies size for size-0 files; gdrive read/stat share one render |
| crate vs runtime mismatch | crate 0.5.6 vs installed CLI/runtime 0.5.10 | bump crate to 0.5.10 |

## Proof

`tests/vfs_e2e.rs::vfs_sandbox_remount_after_restart` (live, `#[ignore]`):
build agent on a persisted sandbox → read provider `page.json` (489 B) → drop
(VM stops) → build a **new** agent on the same sandbox → it transparently
re-mounts and reads again (489 B). Passes repeatedly.

Key property: forwarder **deps persist on the rootfs**, so `apt` runs only on the
**first** mount of a fresh sandbox. VM **restarts** (the actual requirement) find
deps present → fast, deterministic, no network for setup.

## Production guidance

**Bake `python3`, `fuse3`, and `mfusepy` into the guest image.** Then the
bootstrap skips `apt` entirely — the first mount is also fast/offline/reliable.
The runtime `apt` path is only a best-effort fallback for un-baked dev images
(it is inherently flaky: apt-daily lock contention, slow mirrors).

To wire it in agent-k: build the coworker agent with `.vfs(VfsConfig { mounts })`
(provider credentials stay host-side) on a sandbox runenv. No custom tools — the
existing shell/read/write tools operate on `/mnt/vfs/...`.

## Rejected / future design directions

- **Host-FUSE bound into the guest via a virtiofs volume (no in-guest process).**
  Probed (`vfs_host_fuse_bind_into_guest`): on **macOS**, libkrun virtiofs cannot
  export a macFUSE mountpoint — the guest sees the bind but `ls` → ENOENT. So the
  in-guest forwarder is required on macOS. *May* work on a Linux host (libfuse) —
  untested; would eliminate the in-guest process and re-mount entirely.
- **Static, apt-free forwarder (future).** Replace the Python/mfusepy forwarder
  with a statically-linked Rust binary (the `fuser` crate, mounting `/dev/fuse`
  directly as root — FUSE is built into the guest kernel). Baked as an
  arch-specific asset, copied in at boot. Removes the Python + apt dependency
  entirely and makes even the first boot deterministic. Higher effort than image
  baking; pursue if baking is not viable for a deployment.
