//! Delay-loads `dokan2.dll` on a Windows MSVC target, and gives the extension the `LC_RPATH`
//! libfuse-t is found by on a macOS target.
//!
//! cortex needs both from the binary that links it, and a `rustc-link-arg` applies only to the
//! package that prints it, so this cdylib asks itself. With the delay-load the import works on a
//! host without Dokany and only a mount fails, saying what to install. Without the rpath the
//! import fails in `dlopen` with `Library not loaded: @rpath/libfuse-t.dylib`.
//!
//! `ailoy` depends on cortex with its default features, so Dokany and libfuse-t are linked
//! whether or not this crate's `mount` is on; the checks are on the target alone.

fn main() {
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
            "cortex's `mount` feature on macOS needs FUSE-T installed: brew install --cask fuse-t",
        );
    for path in &fuse_t.link_paths {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}
