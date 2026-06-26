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

## Building (static musl, any host — no external toolchain)

The sandbox guest arch matches the host: aarch64 on Apple Silicon, x86_64 on
Intel/AMD Linux. Build for the guest's arch. The only requirement is the musl
target; cross-linking from a non-Linux host uses Rust's **bundled lld**
(`-C linker-flavor=ld.lld`) — no zig, no external linker, no source patches.
(`fuser` 0.17+ fixed its `build.rs` to gate the pure-Rust path on the *target*
OS, so it cross-compiles cleanly to Linux from a macOS build host.)

```sh
rustup target add aarch64-unknown-linux-musl   # or x86_64-unknown-linux-musl
RUSTFLAGS="-C linker-flavor=ld.lld" \
  cargo build --release --target aarch64-unknown-linux-musl
# -> target/aarch64-unknown-linux-musl/release/ailoy-vfs-fwd  (~600 KB static ELF)
```

On a Linux build host the `RUSTFLAGS` is optional (the native linker handles
musl), but it is harmless and keeps the recipe host-agnostic.

## Integration (done — built from source, no committed binary)

ailoy's top-level `build.rs` compiles this crate for the target arch (the guest
arch under libkrun) with the recipe above and writes the ELF to `OUT_DIR`;
`src/vfs/guest.rs` embeds it via
`include_bytes!(concat!(env!("OUT_DIR"), "/ailoy-vfs-fwd"))`, writes it into the
guest, `chmod +x`, and runs it with `VFS_HOST`/`VFS_TOKEN`. It is the sole
forwarder — no runtime dependency install. The mount/re-mount lifecycle
(`AgentVfs::ensure_mounted`) is unchanged.

There is **no committed binary** and **no release step**: changing `src/main.rs`
here is picked up automatically on the next `cargo build` of ailoy (build.rs has
`rerun-if-changed` on the forwarder sources). The build is gated on the
`vfs` + `sandbox` features; other builds get an empty stub.
