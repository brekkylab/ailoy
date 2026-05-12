// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod datatype;
pub mod lang_model;
pub(crate) mod lang_model_impl;
pub(crate) mod macros;
pub mod message;
pub mod runenv;
pub mod tool;
pub mod tool_impl;
pub(crate) mod util;
