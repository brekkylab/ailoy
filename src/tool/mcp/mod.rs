mod common;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm32;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{MCPClient, MCPTool};
#[cfg(target_arch = "wasm32")]
pub use wasm32::{MCPClient, MCPTool};
