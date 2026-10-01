//! napi's link setup, plus the `LC_RPATH` that finds libfuse-t on a macOS target.
//!
//! A `rustc-link-arg` applies only to the package that printed it, so this cdylib emits the
//! rpath itself; without it `require` fails with `Library not loaded: @rpath/libfuse-t.dylib`.

fn main() {
    napi_build::setup();

    // `ailoy` links libfuse-t through virtx's default features whatever this crate's `mount`
    // is, so only the target OS decides.
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
