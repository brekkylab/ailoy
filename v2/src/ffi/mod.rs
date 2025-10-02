#[cfg(any(target_family = "unix", target_family = "windows"))]
pub mod cxx_bridge;
#[cfg(target_family = "wasm")]
pub mod js_bridge;
#[cfg(feature = "nodejs")]
pub(crate) mod node;
#[cfg(feature = "python")]
pub(crate) mod py;
#[cfg(target_family = "wasm")]
pub(crate) mod web;

#[cfg(any(target_family = "unix", target_family = "windows"))]
mod dlpack_wrap;
mod faiss_wrap;
pub use faiss_wrap::*;
