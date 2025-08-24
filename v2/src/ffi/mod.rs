#[cfg(any(target_family = "unix", target_family = "windows"))]
mod cxx_bridge;
#[cfg(target_family = "wasm")]
pub mod js_bridge;
#[cfg(feature = "node")]
mod node;
#[cfg(feature = "python")]
pub mod py;
#[cfg(target_family = "wasm")]
pub mod web;

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use cxx_bridge::*;
#[cfg(target_family = "wasm")]
pub use js_bridge::*;
