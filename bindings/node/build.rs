//! napi's link setup, and delay-loading `dokan2.dll` on a Windows MSVC target.
//!
//! cortex's own `build.rs` asks for the delay-load, but a `rustc-link-arg` applies only to
//! the targets of the package that printed it — so a dependent that is itself linked, as
//! this cdylib is, has to ask again. Without it `require` fails in the loader on a host
//! without Dokany, including for the callers that never mount; with it the DLL is loaded
//! by the first mount, which `cortex::fs::mount_support` checks for first.
//!
//! `ailoy` depends on cortex with its default features, so Dokany is linked whether or not
//! this crate's `mount` is on; the check is on the target alone.
//!
//! macOS needs nothing here: the FUSE-T shim opens libfuse-t itself, at run time.

fn main() {
    napi_build::setup();

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo::rustc-link-arg=/DELAYLOAD:dokan2.dll");
        println!("cargo::rustc-link-lib=delayimp");
    }
}
