#[cfg(any(target_family = "unix", target_family = "windows"))]
mod cxx_bridge;

#[cfg(any(target_family = "unix", target_family = "windows"))]
pub use cxx_bridge::*;
