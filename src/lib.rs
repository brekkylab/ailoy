// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

// Raw-msb_krun ephemeral sandbox that mounts a cortex VFS. cortex owns the
// filesystem; ailoy owns booting the VM and mounting it in.
#[cfg(feature = "krun")]
pub mod krun;

pub mod agent;
pub mod datatype;
pub mod lang_model;
pub(crate) mod macros;
pub mod message;
pub mod runenv;
pub mod skill;
pub mod tool;
pub(crate) mod util;
