//! Agents that drive a language model through tool-augmented turns, in a virtx console.
//!
//! The agent runs in this process, on `ailoy-runner`; the data it is made of and says is
//! `ailoy-core`'s, shown here at the same paths.

pub use ailoy_runner::{
    agent, browser, console, datatype, message, model, to_value, tool, tool_func,
};
