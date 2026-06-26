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
| intermittent multi-minute hang on `cat`/`ls` (the mount, and any process touching it, wedged forever) | **two unbounded I/O paths behind the FUSE forward server** — (a) provider HTTP clients had no timeout (`reqwest::Client::new()`; default object_store options), so a hung upstream API call never returned and the host never answered the forwarder; (b) the forwarder read via `read_to_end` + `SO_RCVTIMEO`, which on the musl build did not reliably abort a stalled read (10-min hang despite a 120s timeout). Found via a native `sample` of the hung process + a `VFS_DIAG`-gated forwarder log. | providers: 30s request + 10s connect timeout (notion/gdrive `reqwest` builder, S3 `object_store` ClientOptions). forwarder: 8s connect timeout + a **manual read loop with a hard 45s wall-clock deadline** — a guaranteed upper bound on any FUSE op regardless of host-side cause |
| one slow op froze the whole mount | single-threaded `fuser` dispatch serialized every FUSE op | worker-pool forwarder (8 workers) offloads each blocking HTTP round trip; panics isolated per-job so a worker can't die and shrink the pool |
| intermittent (~13%) multi-minute hang on **reconnect** under rapid drop/recreate | *below the vfs layer* — microsandbox's embedded SQLite state layer occasionally blocks acquiring a connection (`sqlx ConnectionWorker::establish`), so `start_detached` never returns; also a startup race where a previous owner's async stop SIGKILLs the VM as the new start brings it up ("process exited before agent relay became available"). Found via a `soak` (not targeted tests) + native `sample` of the hung process. | `start_detached_resilient`: bound each `start_detached` with a 25s timeout, retry up to 4x (force-stop between attempts + backoff). Rides over the transient hang/startup-race; a wedged DB now fails fast instead of hanging. Soak: ~13% multi-minute hangs → 20/20 pass (6 recovered via retry, max 41s) |

## Proof

`tests/vfs_e2e.rs::vfs_sandbox_remount_after_restart` (live, `#[ignore]`):
build agent on a persisted sandbox → read provider `page.json` (489 B) → drop
(VM stops) → build a **new** agent on the same sandbox → it transparently
re-mounts and reads again (489 B). Passes repeatedly.

Key property: forwarder **deps persist on the rootfs**, so `apt` runs only on the
**first** mount of a fresh sandbox. VM **restarts** (the actual requirement) find
deps present → fast, deterministic, no network for setup.

Verified across all dimensions (all `tests/vfs_e2e.rs`, live, `#[ignore]`):

- `vfs_sandbox_remount_after_restart` — 25 consecutive clean attaches (6–10 s each).
- `vfs_static_forwarder_large_read` — 300 KB chunked read, sampled bytes match.
- `vfs_static_forwarder_write_unlink` — write → read-back → unlink → gone.
- `vfs_concurrent_access_stress` — 8 simultaneous readers, all return identical
  complete data (the worker-pool forwarder's purpose).
- `vfs_sandbox_reconnect_race` — 12 rapid drop→reconnect cycles, no wedge.

**Liveness guarantee:** every provider/forwarder I/O is now bounded, so a FUSE
operation can never block indefinitely — worst case it fails fast (≤45 s) and
recoverably. A transient failure is per-op (the mount stays; the next op recovers),
and `AgentVfs::ensure_mounted` re-bootstraps on each attach. The liveness probe
(`mount_is_live`) round-trips a root `ls` through the forwarder, so it now also
detects a dead host server (fast connect-refused) and re-mounts.

## Forwarder: static binary (default) with a Python fallback

The default forwarder is now a **static, dependency-free Rust binary**
(`tools/vfs-forwarder`, shipped per guest arch in `src/vfs/assets/`). It mounts
`/dev/fuse` directly as root (FUSE is built into the guest kernel) and needs
**no python/fuse3/mfusepy/apt** — a clean guest image just works, and the first
mount is fast (no install). `guest.rs` picks the binary by the host arch (guest
arch == host arch under libkrun), and falls back to the Python `mfusepy`
forwarder only for arches without a shipped binary or if the static mount fails.

Result: the full lifecycle runs in ~6.5 s on a clean image (was ~30 s with apt),
with no image customization required.

To wire it in agent-k: build the coworker agent with `.vfs(VfsConfig { mounts })`
(provider credentials stay host-side) on a sandbox runenv. No custom tools — the
existing shell/read/write tools operate on `/mnt/vfs/...`. Nothing to bake.

## Rejected / future design directions

- **Host-FUSE bound into the guest via a virtiofs volume (no in-guest process).**
  Probed (`vfs_host_fuse_bind_into_guest`): on **macOS**, libkrun virtiofs cannot
  export a macFUSE mountpoint — the guest sees the bind but `ls` → ENOENT. So the
  in-guest forwarder is required on macOS. *May* work on a Linux host (libfuse) —
  untested; would eliminate the in-guest process and re-mount entirely.
- **Static, apt-free forwarder (ADOPTED — now the default).** See the
  "Forwarder" section above. The remaining open item is purely how the per-arch
  binary is produced for release: it currently ships as committed assets
  (`src/vfs/assets/ailoy-vfs-fwd.{aarch64,x86_64}`, ~1.2 MB total); a CI
  cross-build step could replace the committed blobs later. Build/cross-compile
  recipe and gotchas are in `tools/vfs-forwarder/README.md`.
