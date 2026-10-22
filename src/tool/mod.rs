//! Tool abstractions and their relationships.
//!
//! A [`ToolProvider`] says how one tool is built (a function or an MCP server's
//! tool), and the process-wide registry keys one by tool name. An
//! [`crate::agent::AgentSpec`] lists the [`ToolDesc`]s it wants exposed to the
//! model; at agent construction each desc is built into a concrete [`ToolFunc`]
//! through the provider set for its name.
//!
//! ```text
//! registry (tool name → ToolProvider)
//!     │
//!     │  create_tool(&ToolDesc)
//!     ▼
//! ToolFunc   ← bound to the agent's spec
//! ```
//!
//! ## Lifecycle
//!
//! 1. **Registration** — [`set_tool_provider`] sets a provider under a name; the
//!    registry starts with every built-in tool.
//! 2. **Resolution** — [`create_tool`] builds a [`ToolFunc`] for each
//!    [`ToolDesc`] in `spec.tools` when an `Agent` is instantiated.
//! 3. **Execution** — the agent invokes the resolved [`ToolFunc`] for each tool
//!    call the model issues.
//!
//! Remote sources (MCP) are contacted during step 1, since what they offer
//! is knowable only by asking and step 2 has nothing to await on. Their
//! registering calls ([`register_mcp_stdio`], [`register_mcp_streamable_http`])
//! hand back the [`ToolDesc`]s to put in the spec, so by step 2 they are
//! ordinary name-keyed entries.

mod desc;
mod func;
pub(crate) mod r#impl;
mod provider;

pub use desc::*;
pub use func::*;
pub use r#impl::*;
pub use provider::*;
