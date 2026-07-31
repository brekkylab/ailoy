// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

// Re-export cortex so callers construct volumes (`ailoy::cortex::VolumeSpec`,
// `ailoy::cortex::S3Config`, …) without depending on cortex directly. New cortex
// volume kinds surface here automatically — ailoy needs no per-volume wrapper.
#[cfg(any(feature = "sandbox", feature = "local-fuse"))]
pub use cortex;

pub mod agent;
pub mod datatype;
pub mod lang_model;
pub(crate) mod macros;
pub mod message;
pub mod runenv;
pub mod skill;
pub mod tool;
pub(crate) mod util;
