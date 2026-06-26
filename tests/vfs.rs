//! VFS integration tests (live; require credentials + microsandbox/macFUSE).
//!
//! Split across `tests/vfs/*.rs`; this file is the single test binary that ties
//! them together. Run explicitly, e.g.:
//!
//! ```sh
//! set -a; . .env; set +a
//! cargo test --features "vfs sandbox" --test vfs -- --ignored --nocapture
//! ```
//!
//! (A crate-root file resolves `mod foo;` to `tests/foo.rs`, so the submodules
//! are pulled in from the `vfs/` subdir explicitly via `#[path]`.)
#![cfg(all(feature = "vfs", feature = "sandbox"))]

#[path = "vfs/common.rs"]
mod common;

#[path = "vfs/bench.rs"]
mod bench;
#[path = "vfs/forwarder.rs"]
mod forwarder;
#[path = "vfs/host.rs"]
mod host;
#[path = "vfs/providers.rs"]
mod providers;
#[path = "vfs/sandbox.rs"]
mod sandbox;
