// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod datatype;
pub mod lang_model;
pub(crate) mod macros;
pub mod message;
pub mod runenv;
pub mod skill;
pub mod tool;
pub(crate) mod util;
#[cfg(feature = "vfs")]
pub mod vfs;
