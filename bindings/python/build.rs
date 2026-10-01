//! Delay-loads `dokan2.dll` on a Windows MSVC target, and gives the extension the `LC_RPATH`
//! libfuse-t is found by on a macOS target.
//!
//! A `rustc-link-arg` applies only to the package that printed it, so this cdylib emits both
//! itself. With the delay-load the import works on a host without Dokany and only a mount fails,
//! saying what to install; without the rpath the import fails with
//! `Library not loaded: @rpath/libfuse-t.dylib`.

fn main() {
    // `ailoy` links Dokany and libfuse-t through virtx's default features whatever this crate's
    // `mount` is, so only the target decides.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo::rustc-link-arg=/DELAYLOAD:dokan2.dll");
        println!("cargo::rustc-link-lib=delayimp");
    }

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let fuse_t = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("fuse-t")
        .expect(
            "virtx's `mount` feature on macOS needs FUSE-T installed: brew install --cask fuse-t",
        );
    for path in &fuse_t.link_paths {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}
