//! Ailoy Desktop's session engine: everything the window does, without the window.

pub mod assembler;
pub mod catalog;
pub mod config;
pub mod console;
pub mod engine;
pub mod error;
pub mod events;
pub mod prompt;
pub mod providers;
pub mod run;
pub mod store;
pub mod types;
pub mod usage;
pub mod workspace;

pub use catalog::Catalog;
pub use config::EngineConfig;
pub use engine::Engine;
pub use error::{EngineError, Result};
pub use events::RunEvent;
pub use store::Store;
pub use types::*;

/// The text decoder, for the ignored probe in `tests/encoding_probe.rs`.
///
/// Exposed rather than duplicated: a probe that reimplemented the decision would be
/// checking its own copy of it.
pub fn decode_for_tests(buf: Vec<u8>, truncated: bool) -> Option<(String, &'static str)> {
    workspace::fsops::as_text(buf, truncated)
}
