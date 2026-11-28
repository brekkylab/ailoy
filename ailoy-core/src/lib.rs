//! What names an agent and what it says, as data: the spec ([`agent`]), the messages of a
//! conversation ([`message`]), a tool's description ([`tool`]) and a model's options
//! ([`model`]). All of it is serde data that goes anywhere an agent runs; what builds a model
//! from its name, and runs the agent, is the `ailoy-runner` crate. [`tool`] also holds what makes and
//! runs a tool: its provider's trait and `ToolFunc`. An [`AgentProvider`](agent::AgentProvider)
//! holds the tools a spec names, and the agents kept by name, each with its sessions.
//!
//! The data is always built. `Client` and `AppClient`, which ask a server for what it keeps and
//! runs, are the `client` feature; reaching a server in this process is `embed`.

pub mod agent;
#[cfg(feature = "client")]
mod client;
pub mod datatype;
pub mod message;
pub mod model;
pub mod tool;

#[cfg(feature = "client")]
pub use client::{AppClient, Client};
#[cfg(feature = "client")]
pub use http_over_x::{Auth, Error};

/// For [`to_value!`], whose expansion names `IndexMap` in the caller's crate, which need not
/// depend on `indexmap` itself.
#[doc(hidden)]
pub use indexmap as __indexmap;
