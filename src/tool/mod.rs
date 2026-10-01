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
//! 1. **Registration** — entries are inserted into a [`ToolProvider`], which
//!    starts with every built-in tool.
//! 2. **Resolution** — [`ToolProvider::provide`] builds a [`ToolFunc`] for each
//!    [`ToolDesc`] in `spec.tools` when an `Agent` is instantiated.
//! 3. **Execution** — the agent invokes the resolved [`ToolFunc`] for each tool
//!    call the model issues.
//!
//! Remote sources (MCP, A2A) are contacted during step 1, since what they offer
//! is knowable only by asking and step 2 has nothing to await on. Their
//! registering calls ([`register_mcp_stdio`], [`register_mcp_streamable_http`],
//! [`register_a2a`]) hand back the [`ToolDesc`]s to put in the spec, so by
//! step 2 they are ordinary name-keyed entries.

mod desc;
mod func;
pub(crate) mod r#impl;
mod provider;

pub use desc::*;
pub use func::*;
pub use r#impl::*;
pub use provider::*;
