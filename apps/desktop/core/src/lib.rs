//! Ailoy Desktop's session engine: everything the window does, without the window.

pub mod config;
pub mod error;
pub mod types;

pub use config::EngineConfig;
pub use error::{EngineError, Result};
pub use types::*;
