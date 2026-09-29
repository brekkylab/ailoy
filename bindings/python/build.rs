//! Gives the extension the `LC_RPATH` libfuse-t is found by, when the `mount` feature is on
//! for a macOS target.
//!
//! cortex's own `build.rs` emits the same rpath, but a `rustc-link-arg` applies only to the
//! targets of the package that printed it — so a dependent that is itself linked, as this
//! cdylib is, has to ask again. Without it the import fails in `dlopen` with
//! `Library not loaded: @rpath/libfuse-t.dylib`.
//!
//! `ailoy` depends on cortex with its default features, so libfuse-t is linked whether or not
//! this crate's `mount` is on; the check is on the target alone.

fn main() {
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
