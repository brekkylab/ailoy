//! Build the in-guest VFS forwarder from source (no committed binary).
//!
//! The sandbox VFS path mounts a small static FUSE forwarder inside the guest
//! (`tools/vfs-forwarder`). Rather than commit a prebuilt per-arch binary, we
//! compile it here for the guest arch (== the crate's target arch under libkrun
//! same-arch virtualization) as a static `…-unknown-linux-musl` ELF and embed it
//! via `include_bytes!(concat!(env!("OUT_DIR"), "/ailoy-vfs-fwd"))`.
//!
//! Cross-linking from a non-Linux build host (e.g. macOS dev) needs no external
//! toolchain: Rust's bundled `lld` (`-C linker-flavor=ld.lld`) links the ELF.
//! The only requirement is the musl target: `rustup target add <arch>-unknown-linux-musl`.
//!
//! The forwarder is only used by the sandbox guest, so it is built only when both
//! the `vfs` and `sandbox` features are enabled. Otherwise an empty stub is
//! written so the `include_bytes!` in `src/vfs/guest.rs` always resolves (the
//! bytes are never used without a sandbox runenv).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let embedded = out_dir.join("ailoy-vfs-fwd");

    let want_forwarder = std::env::var_os("CARGO_FEATURE_VFS").is_some()
        && std::env::var_os("CARGO_FEATURE_SANDBOX").is_some();
    if !want_forwarder {
        // Empty stub: include_bytes! resolves but the bytes are never used.
        std::fs::write(&embedded, b"").expect("write forwarder stub");
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let fwd_dir = manifest_dir.join("tools/vfs-forwarder");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH");
    let triple = format!("{arch}-unknown-linux-musl");

    // Rebuild the embedded binary only when the forwarder sources change.
    for f in ["src/main.rs", "Cargo.toml", "Cargo.lock"] {
        println!("cargo:rerun-if-changed={}", fwd_dir.join(f).display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(&cargo)
        .current_dir(&fwd_dir)
        .args(["build", "--release", "--target", &triple])
        // Bundled lld cross-links the ELF from any host with no external linker.
        .env("RUSTFLAGS", "-C linker-flavor=ld.lld")
        // Do not inherit the parent build's rustflags / target dir; keep the
        // forwarder build self-contained in its own `target/`.
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_DIR")
        .status()
        .expect("failed to spawn cargo for the vfs forwarder build");

    if !status.success() {
        panic!(
            "failed to build the in-guest vfs forwarder for {triple}.\n\
             Ensure the musl target is installed:\n    rustup target add {triple}\n\
             (the forwarder cross-links with Rust's bundled lld; no external toolchain is needed)."
        );
    }

    let built = fwd_dir
        .join("target")
        .join(&triple)
        .join("release/ailoy-vfs-fwd");
    std::fs::copy(&built, &embedded).unwrap_or_else(|e| {
        panic!("vfs forwarder binary missing at {}: {e}", built.display())
    });
}
