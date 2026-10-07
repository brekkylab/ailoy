//! Node bindings for ailoy, built with napi-rs.
//!
//! The shape is ailoy's own, spelled in JavaScript: an [`Agent`](ailoy::agent::Agent) is built
//! by an [`AgentBuilder`](ailoy::agent::AgentBuilder) and awaited, a turn is iterated with
//! `for await`, and the registries are the process-wide ones the crate keeps. Names are
//! camelCased and nothing else changes, so ailoy's Rust documentation applies as is.
//!
//! Data (messages, specs, tool descriptions, a turn's outputs) crosses as plain objects in its
//! serde form rather than a class per type, since that form is what ailoy stores and sends.
//!
//! The console an agent runs in is `@brekkylab/virtx`'s `ConsoleClient`, which an agent drives
//! through protocol frames (see [`console`]): two addons cannot see into each other's types,
//! and this one carries none of virtx's classes.

mod agent;
mod console;
mod convert;
mod error;
mod registry;
mod tool;
