//! Stages the model catalog snapshot for `include_str!`, whether or not there is one.
//!
//! `assets/models.json` is not tracked: `npm run catalog` (part of `tauri:build`) fetches it
//! from models.dev right before a release is built, so what ships is as fresh as the build.
//! Without it — a clone, a dev build, CI — the crate still compiles, embedding nothing, and
//! the app fills its catalog from the network at start. See `catalog.rs`.
//!
//! Never fetches here. A build script that needs the network breaks offline and sandboxed
//! builds, runs again on every rust-analyzer pass, and still would not refresh when cargo
//! decided it had nothing to rerun.

use std::{env, fs, path::PathBuf};

fn main() {
    // The directory rather than the file: a path that does not exist makes cargo rerun the
    // script — and rebuild the crate — on every build, and a new file in a watched
    // directory is a change cargo sees.
    println!("cargo:rerun-if-changed=assets");
    let out =
        PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("models.json");
    let snapshot = match fs::read("assets/models.json") {
        Ok(bytes) => bytes,
        Err(_) => {
            if env::var("PROFILE").as_deref() == Ok("release") {
                println!(
                    "cargo:warning=no model catalog to embed: run `npm run catalog` in apps/desktop first, or the app starts with an empty model list until it reaches models.dev"
                );
            }
            Vec::new()
        }
    };
    fs::write(&out, snapshot).expect("OUT_DIR is writable");
}
