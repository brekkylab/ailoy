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
//!    [`ToolProvider::insert_mcp`].
//! 2. **`Agent` is instantiated from an `AgentSpec`** — [`ToolProvider::provide`]
//!    walks `spec.tools`, looks up each [`ToolDesc`] by name, and builds the
//!    matching [`ToolFunc`] (a fresh one per call for factory-style entries).
//!
//! The two remote sources do not fit that order, because what they contribute
//! to step 1 is only knowable by asking them and step 2 is a `fn` with nothing
//! to await on. So both are contacted during step 1 instead, and the registering
//! call hands back the [`ToolDesc`]s to put in the spec — by step 2 they are
//! ordinary name-keyed entries like any other:
//!
//! * **MCP** — [`register_mcp_stdio`] / [`register_mcp_streamable_http`] run
//!   `initialize` and `tools/list`, and fan one server out into one entry per
//!   tool, named `{prefix}__{remote name}`.
//! * **A2A** — [`register_a2a`] fetches the agent card for its description. One
//!   agent is one tool; calling it needs only the URL, so
//!   [`ToolProvider::insert_a2a`] alone is enough when the desc is already known.
//! 3. **`ToolFunc` drives execution** — when the model issues a tool call, the
//!    agent invokes the resolved [`ToolFunc`] to produce the result stream.

mod desc;
mod func;
pub(crate) mod r#impl;
mod provider;

pub use desc::*;
pub use func::*;
pub use r#impl::*;
pub use provider::*;
