#[cfg(any(target_family = "unix", target_family = "windows"))]
mod cxx_bridge;
#[cfg(feature = "node")]
mod node;
#[cfg(feature = "python")]
pub mod py;

pub mod dlpack_wrap;
pub mod faiss_wrap;
pub mod util;

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use cxx_bridge::*;
