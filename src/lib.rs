// Allow proc-macros to use `ailoy::` prefix when invoked from within this crate.
extern crate self as ailoy;

pub mod agent;
pub mod datatype;
pub mod lang_model;
#[cfg(feature = "rt")]
mod lang_model_impl;
pub(crate) mod macros;
pub mod message;
pub mod shell;
pub mod tool;
#[cfg(feature = "rt")]
pub(crate) mod tool_impl;
