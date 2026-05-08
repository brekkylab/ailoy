//! Tool abstractions and their relationships.
//!
//! A [`ToolProvider`] is a name-keyed registry of [`ToolProviderElem`] entries —
//! each describes a tool source (function, MCP server, or A2A agent). An
//! [`crate::agent::AgentSpec`] lists the [`ToolDesc`]s it wants exposed to the
//! model; at agent construction the provider is asked to resolve each desc to
//! a concrete [`ToolFunc`] that drives the tool's runtime behaviour.
//!
//! ```text
//! ToolProvider (name → ToolProviderElem)
//!     │
//!     │  ToolProvider::provide(&[ToolDesc])
//!     ▼
//! HashMap<String, ToolFunc>   ← bound to the agent's spec
//! ```
//!
//! ## Lifecycle
//!
//! 1. **`ToolProvider` is created** — populated with entries via
//!    [`ToolProvider::insert_func`], [`ToolProvider::insert_func_factory`],
//!    [`ToolProvider::insert_builtin`], [`ToolProvider::insert_a2a`], or the
//!    MCP variants.
//! 2. **`Agent` is instantiated from an `AgentSpec`** — [`ToolProvider::provide`]
//!    walks `spec.tools`, looks up each [`ToolDesc`] by name, and builds the
//!    matching [`ToolFunc`] (a fresh one per call for factory-style entries).
//! 3. **`ToolFunc` drives execution** — when the model issues a tool call, the
//!    agent invokes the resolved [`ToolFunc`] to produce the result stream.

mod desc;
mod func;
mod provider;

pub use desc::*;
pub use func::*;
pub use provider::*;
