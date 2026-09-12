//! Ailoy Desktop's session engine: everything the window does, without the window.

pub mod assembler;
pub mod catalog;
pub mod config;
pub mod console;
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
pub use error::{EngineError, Result};
pub use events::RunEvent;
pub use store::Store;
pub use types::*;
