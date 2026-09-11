//! Ailoy Desktop's session engine: everything the window does, without the window.

pub mod catalog;
pub mod config;
pub mod error;
pub mod prompt;
pub mod providers;
pub mod store;
pub mod types;

pub use catalog::Catalog;
pub use config::EngineConfig;
pub use error::{EngineError, Result};
pub use store::Store;
pub use types::*;
