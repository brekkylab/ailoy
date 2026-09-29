//! Node bindings for ailoy, built with napi-rs.
//!
//! The shape is ailoy's own, spelled in JavaScript: an [`Agent`](ailoy::agent::Agent) is built
//! by an [`AgentBuilder`](ailoy::agent::AgentBuilder) and awaited, a turn is iterated with
//! `for await`, and the registries an agent resolves its model and tools from are the
//! process-wide ones the crate keeps. Names are camelCased and nothing else changes — a caller
//! reading ailoy's Rust documentation should find the same names doing the same things.
//!
//! What crosses the boundary as data — messages, specs, tool descriptions, a turn's outputs —
//! crosses as the objects its serde form already is, rather than as a class per type: that
//! form is what ailoy stores and sends, so it is the one a caller already has.
//!
//! cortex comes along whole. Its classes are linked into this addon from `cortex-node` rather
//! than loaded from cortex's own, because an agent has to take a `ConsoleClient` apart to share its
//! session, and two addons cannot see into each other's — see [`agent`] for the sharing.

mod agent;
mod convert;
mod error;
mod registry;
mod tool;
