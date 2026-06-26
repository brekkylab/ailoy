# ailoy-vfs-forwarder

A **static, dependency-free** in-guest FUSE forwarder for the ailoy VFS sandbox
path. It mounts a FUSE filesystem in the sandbox guest and forwards operations
to the host forward server (`VfsForward`) over plain HTTP, then serves provider
files (S3 / Notion / GDrive) to the guest shell.

## Why this exists

The original forwarder was a Python `mfusepy` script requiring `python3` +
`fuse3` + `mfusepy` in the guest — installed via `apt`/`pip` on first boot. That
runtime install was **fragile** (boot-time `apt-daily` dpkg-lock contention, slow
mirrors) and slow, so it was replaced by this binary.

This binary needs **none of that**. It is statically linked (musl) and mounts via
the kernel's `/dev/fuse` directly as root (the `fuser` pure-Rust mount path — no
`fusermount3`, no `libfuse`). FUSE is built into the sandbox guest kernel, so the
binary works on **any** guest image with **zero** setup, eliminating the apt
fragility and the need to bake deps.

Proven by `tests/vfs_e2e.rs::vfs_static_forwarder_full`: in a clean guest with
`python3` and `fusermount3` absent, it mounts `/mnt/vfs` and serves a Notion
`page.json` (489 B) over `allow@host` egress.

## Runtime contract

```
ailoy-vfs-fwd <mountpoint>
  env VFS_HOST=http://host.microsandbox.internal:<port>   # host forward server
  env VFS_TOKEN=<token>                                    # x-vfs-token
```
Must run as **root** (the guest is). No other guest dependencies.

## Building (per guest arch, static musl)

The sandbox guest arch matches the host: aarch64 on Apple Silicon, x86_64 on
Intel/AMD Linux. Build for the guest's arch.

### On a Linux build host (CI — simplest, no patches)
`fuser` without `libfuse` builds natively on Linux:
```sh
rustup target add aarch64-unknown-linux-musl   # or x86_64-unknown-linux-musl
cargo build --release --target aarch64-unknown-linux-musl
# -> target/aarch64-unknown-linux-musl/release/ailoy-vfs-fwd  (~600 KB static ELF)
```

### Cross-compiling from macOS (dev convenience, via zig)
Two upstream gotchas, both worked around with config (no source changes here):
1. `fuser`'s `build.rs` checks the **host** OS, not the target, and refuses the
   pure-Rust path when the host isn't Linux. Patch its `build.rs` to gate on
   `CARGO_CFG_TARGET_OS` instead (use a `[patch.crates-io]` vendored copy), or
   just build on Linux.
2. zig + Rust musl duplicate the `crt1` startup object. Add
   `rustflags = ["-C", "link-self-contained=no"]` for the target.

```sh
rustup target add aarch64-unknown-linux-musl
# .cargo/config.toml:
#   [target.aarch64-unknown-linux-musl]
#   linker = "zcc.sh"                         # exec zig cc -target aarch64-linux-musl "$@"
#   rustflags = ["-C", "link-self-contained=no"]
cargo build --release --target aarch64-unknown-linux-musl
```

## Integration (done)

`src/vfs/guest.rs` ships this binary per guest arch as
`src/vfs/assets/ailoy-vfs-fwd.{aarch64,x86_64}` (`include_bytes!`), picks by the
host arch (guest arch == host arch under libkrun), writes it, `chmod +x`, and
runs it with `VFS_HOST`/`VFS_TOKEN`. It is the sole forwarder — a binary ships
for every supported arch, so there is no runtime dependency install.
The mount/re-mount lifecycle (`AgentVfs::ensure_mounted`) is unchanged.

**Updating the committed binaries** after changing `src/main.rs`: cross-build
both arches (see above) and copy the outputs to `src/vfs/assets/`:
```sh
cargo build --release --target aarch64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
cp target/aarch64-unknown-linux-musl/release/ailoy-vfs-fwd ../../src/vfs/assets/ailoy-vfs-fwd.aarch64
cp target/x86_64-unknown-linux-musl/release/ailoy-vfs-fwd  ../../src/vfs/assets/ailoy-vfs-fwd.x86_64
```
A CI cross-build step could replace the committed blobs later.
