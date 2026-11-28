//! The tools an agent calls.
//!
//! A [`ToolSet`] describes and makes tools: a group of this crate's ([`Console`], [`Web`],
//! [`Browser`]), a function of this process (`(ToolDesc, ToolFunc)`), or an MCP server's
//! ([`MCPServer`]). A [`ToolProvider`] keeps each tool's description and set by its name; an
//! [`AgentSpec`](crate::agent::AgentSpec) lists the names, and each is made into a [`ToolFunc`]
//! when an `AgentRuntime` is built.
//!
//! ```text
//! ToolProvider (tool name → ToolDesc, ToolSet)
//!     │
//!     │  create(name)
//!     ▼
//! ToolDesc, ToolFunc   ← shown to the agent's model, called by the agent
//! ```
//!
//! An agent made in this process makes its tools through the `tools` of
//! [`default_provider`](crate::agent::default_provider).

mod func;
pub(crate) mod r#impl;

pub use ailoy_core::tool::{
    MAX_TOOL_NAME_LEN, ToolDesc, ToolDescBuilder, ToolFunc, ToolProvider, ToolSet,
    sanitize_tool_name,
};
pub use r#impl::*;

/// What a sandbox tool runs a command with, and reads and writes a file with.
pub use crate::console::{ExecOutput, Sandbox, exec, read, resolve, write};
