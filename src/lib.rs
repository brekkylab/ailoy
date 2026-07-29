// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

// Re-export cortex so downstreams (e.g. agent-k) can register its virtio-fs
// backends in the sandbox process — `ailoy::cortex::msb::register_s3_backend()`
// — without depending on cortex directly.
#[cfg(feature = "sandbox")]
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
