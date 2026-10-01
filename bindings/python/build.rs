//! Delay-loads `dokan2.dll` on a Windows MSVC target.
//!
//! A `rustc-link-arg` applies only to the package that printed it, so this cdylib asks for the
//! delay-load itself; without it the import fails with `DLL load failed` on a host without
//! Dokany, even for callers that never mount. macOS needs nothing: cortex opens libfuse-t
//! itself, at run time.

fn main() {
    // `ailoy` links Dokany through cortex's default features whatever this crate's `mount` is,
    // so only the target decides.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo::rustc-link-arg=/DELAYLOAD:dokan2.dll");
        println!("cargo::rustc-link-lib=delayimp");
    }
}
