//! Build two helper binaries into `OUT_DIR` so the sandbox runenv can
//! `include_bytes!` them without committing binary assets. Only runs with the
//! `sandbox` feature.
//!
//! 1. The guest init (`crates/ailoy-guest-init`), cross-compiled to Linux/musl —
//!    needs the guest-arch musl target (`rustup target add
//!    {aarch64,x86_64}-unknown-linux-musl`). Runs inside the guest.
//! 2. The host VM boot helper (`crates/ailoy-krun-boot`), built release+LTO for
//!    the host — a small signed copy of *this* is what `Sandbox::exec` invokes to
//!    boot the microVM (instead of re-invoking the fat consumer binary).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_SANDBOX").is_none() {
        return;
    }
    build_guest_init();
    build_krun_boot();
}

/// Build the host-side boot helper (`crates/ailoy-krun-boot`) release+LTO for the
/// host arch and drop it at `OUT_DIR/ailoy-krun-boot`. Its own workspace keeps it
/// out of ailoy's normal build; we invoke a nested cargo with an isolated target
/// dir so the outer build's target-dir lock and flags don't interfere.
fn build_krun_boot() {
    let crate_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("crates/ailoy-krun-boot");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("krun-boot");

    for f in ["src/main.rs", "Cargo.toml"] {
        println!("cargo:rerun-if-changed={}", crate_dir.join(f).display());
    }

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        .current_dir(&crate_dir)
        .args(["build", "--release"])
        .env("CARGO_TARGET_DIR", &target_dir)
        // Keep the parent build's flags/target/wrappers out of this build.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .status()
        .expect("run cargo for ailoy-krun-boot");
    assert!(status.success(), "ailoy-krun-boot build failed");

    let bin = target_dir.join("release/ailoy-krun-boot");
    let dest = out_dir.join("ailoy-krun-boot");
    std::fs::copy(&bin, &dest)
        .unwrap_or_else(|e| panic!("copy krun-boot {} -> {}: {e}", bin.display(), dest.display()));
}

/// Cross-compile the guest init (`crates/ailoy-guest-init`) to a Linux/musl
/// static binary and drop it at `OUT_DIR/ailoy-guest-init`.
fn build_guest_init() {

    // A krun guest runs the host's architecture.
    let target = match env::consts::ARCH {
        "aarch64" => "aarch64-unknown-linux-musl",
        "x86_64" => "x86_64-unknown-linux-musl",
        other => panic!("ailoy sandbox: no guest-init musl target for host arch {other}"),
    };

    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("crates/ailoy-guest-init");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("guest-init");

    for f in ["src/main.rs", "Cargo.toml", ".cargo/config.toml"] {
        println!("cargo:rerun-if-changed={}", crate_dir.join(f).display());
    }

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        // Run from the guest crate so its `.cargo/config.toml` (the lld linker
        // for the musl target) is discovered.
        .current_dir(&crate_dir)
        .args(["build", "--release", "--target", target])
        .env("CARGO_TARGET_DIR", &target_dir)
        // Keep the parent build's flags/target/wrappers out of the guest build.
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_BUILD_TARGET")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .status()
        .expect("run cargo for ailoy-guest-init");
    assert!(status.success(), "ailoy-guest-init build failed");

    let bin = target_dir.join(target).join("release/ailoy-guest-init");
    let dest = out_dir.join("ailoy-guest-init");
    std::fs::copy(&bin, &dest).unwrap_or_else(|e| {
        panic!("copy guest-init {} -> {}: {e}", bin.display(), dest.display())
    });
}
