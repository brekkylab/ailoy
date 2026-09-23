fn main() {
    emit_fuse_t_rpath();
    tauri_build::build()
}

/// Adds the `LC_RPATH` that `@rpath/libfuse-t.dylib` needs.
///
/// This binary links FUSE-T through `cortex`, and cortex's build script already emits this
/// exact flag — but `cargo::rustc-link-arg` from a *dependency* never reaches the final
/// link: it applies only to the targets of the package that emits it, and cortex builds a
/// rlib, which has no link step. So the bundle came out with no `LC_RPATH` at all and died
/// in dyld before `main` — no window, no log line, and none of the dialogs `lib.rs` has,
/// because none of that code ever ran.
///
/// A shell can paper over it with `DYLD_FALLBACK_LIBRARY_PATH`, which is how this went
/// unnoticed. Finder cannot: LaunchServices sets no such variable, and macOS strips every
/// `DYLD_*` on exec of a system binary anyway.
///
/// Build scripts are compiled for the host, so this `cfg` reads the machine doing the
/// building — which for a macOS-only app is the same thing.
#[cfg(target_os = "macos")]
fn emit_fuse_t_rpath() {
    // `cargo_metadata(false)`: cortex owns the `-l`/`-L` for this library. All that is
    // wanted here is the search path, restated as a runtime one.
    let fuse_t = pkg_config::Config::new()
        .cargo_metadata(false)
        .probe("fuse-t")
        .expect("the desktop app needs FUSE-T installed: brew install --cask fuse-t");
    for path in &fuse_t.link_paths {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
}

#[cfg(not(target_os = "macos"))]
fn emit_fuse_t_rpath() {}
