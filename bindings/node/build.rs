//! napi's link setup, plus `dokan2.dll` delay-loaded on a Windows MSVC target.
//!
//! A `rustc-link-arg` applies only to the package that printed it, so this cdylib emits it
//! itself. With the delay-load `require` works on a host without Dokany and only a mount fails,
//! saying what to install. macOS needs nothing: virtx opens libfuse-t itself, at run time.

fn main() {
    napi_build::setup();

    // `ailoy` links Dokany through virtx's default features whatever this crate's `mount` is,
    // so only the target decides.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo::rustc-link-arg=/DELAYLOAD:dokan2.dll");
        println!("cargo::rustc-link-lib=delayimp");
    }
}
