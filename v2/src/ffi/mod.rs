#[cfg(any(target_family = "unix", target_family = "windows"))]
mod cxx_bridge;
#[cfg(feature = "node")]
mod node;
#[cfg(feature = "python")]
pub mod py;

#[cfg(any(target_family = "unix", target_family = "windows"))]
mod dlpack_wrap;
#[cfg(any(target_family = "unix", target_family = "windows"))]
mod faiss_wrap;

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use cxx_bridge::*;

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use faiss_wrap::*;
