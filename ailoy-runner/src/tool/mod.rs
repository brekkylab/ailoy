//! The tools an agent calls.
//!
//! A [`ToolProvider`] describes and makes tools: the built-in ones, a function of this process
//! (`(ToolDesc, ToolFunc)`), or an MCP server's ([`MCPServer`]). The process-wide registry keeps
//! each tool's description and provider by its name; an
//! [`AgentSpec`](crate::agent::AgentSpec) lists the names, and each is made into a
//! [`ToolFunc`] when an `Agent` is built.
//!
//! ```text
//! registry (tool name → ToolDesc, ToolProvider)
//!     │
//!     │  create_tool(name)
//!     ▼
//! ToolDesc, ToolFunc   ← shown to the agent's model, called by the agent
//! ```
//!
//! [`set_tool_provider`] provides tools in place of those of their names; the registry starts
//! with every built-in tool. An MCP server is asked what it offers as it is registered
//! ([`register_mcp_stdio`], [`register_mcp_streamable_http`]), which hand back the names to put
//! in the spec.

mod desc;
mod func;
pub(crate) mod r#impl;
pub(crate) mod provider;

pub use ailoy_core::tool::{
    MAX_TOOL_NAME_LEN, ToolDesc, ToolDescBuilder, ToolFunc, ToolProvider, sanitize_tool_name,
};
pub(crate) use desc::warn_if_tool_name_too_long;
pub use r#impl::*;
pub use provider::*;

/// What a sandbox tool runs a command with, and reads and writes a file with.
pub use crate::console::{ExecOutput, Sandbox, exec, read, resolve, write};
