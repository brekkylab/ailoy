//! The pool command this app attaches.
//!
//! # Why a library
//!
//! Two binaries have to reach the **same `Executable`**. `headhunter` registers it with
//! the console and hands it to the agent; `headhunting` calls the very same thing on this
//! host. That is what makes "checking a command by hand" mean the same as what the agent
//! will meet.
//!
//! The rest of the app (`prompt`, `trace`) is not here. It serves only this app's run, so
//! it sits next to `main.rs`.

pub mod executable;
