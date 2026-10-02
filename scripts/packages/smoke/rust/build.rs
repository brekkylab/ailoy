//! What a binary that takes ailoy needs on Windows, as virtx's README says of a binary that
//! takes virtx with `mount` -- which ailoy does: the delay-load of `dokan2.dll`, which no
//! dependency's `build.rs` can ask for on a dependent's behalf. Without it this binary would
//! not start on a host without Dokany.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows")
        && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    {
        println!("cargo::rustc-link-arg=/DELAYLOAD:dokan2.dll");
        println!("cargo::rustc-link-lib=delayimp");
    }
}
