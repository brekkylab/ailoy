//! Cross-compile the guest init (`crates/ailoy-guest-init`) to a Linux/musl
//! static binary and drop it in `OUT_DIR`, so the sandbox runenv can
//! `include_bytes!` it without a committed binary asset. Only runs with the
//! `sandbox` feature; needs the guest-arch musl target installed
//! (`rustup target add {aarch64,x86_64}-unknown-linux-musl`).

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_SANDBOX").is_none() {
        return;
    }

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
