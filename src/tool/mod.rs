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
//! 1. **`ToolProvider` is created** — [`ToolProvider::new`] starts pre-loaded
//!    with every built-in tool ([`ToolProvider::empty`] opts out); additional
//!    entries are added via [`ToolProvider::insert_func`],
//!    [`ToolProvider::insert_func_factory`], [`ToolProvider::insert_a2a`], or
//!    the MCP variants.
//! 2. **`Agent` is instantiated from an `AgentSpec`** — [`ToolProvider::provide`]
//!    walks `spec.tools`, looks up each [`ToolDesc`] by name, and builds the
//!    matching [`ToolFunc`] (a fresh one per call for factory-style entries).
//! 3. **`ToolFunc` drives execution** — when the model issues a tool call, the
//!    agent invokes the resolved [`ToolFunc`] to produce the result stream.

mod desc;
mod func;
pub(crate) mod r#impl;
mod provider;

/// The console a tool runs commands in — `cortex`'s own type, re-exported rather
/// than wrapped.
///
/// Re-exported here for two reasons. It is the path [`tool_func!`](crate::tool_func)
/// writes into its expansion, and a `#[macro_export]`ed macro can only name types
/// through `$crate` — spelling `::cortex::..` there would make every crate that
/// writes a tool depend on `cortex` under that exact name. And it is where a tool
/// author looks: the console is part of the tool-writing interface, so it should be
/// reachable from the module that defines the rest of it.
///
/// Nothing is added to it. A tool calls [`Console::exec`], [`Console::read`] and
/// [`Console::write`] as cortex defines them — argv in, bytes out, milliseconds, and
/// a timeout that arrives as an error rather than a flag. ailoy ships no convenience
/// layer over that on purpose: a helper here would become a second interface that
/// tool authors have to learn and that has to be kept in step with cortex's.
pub use cortex::console::Console;
pub use desc::*;
pub use func::*;
pub use r#impl::WebSearchEngineKind;
pub use provider::*;
// pub use r#impl::builtin::{};
