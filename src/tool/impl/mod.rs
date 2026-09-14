// mod a2a;
mod builtins;
mod mcp;

// pub(crate) use a2a::{get_a2a_tool_desc, get_a2a_tool_func};
pub use builtins::WebSearchEngineKind;
pub(crate) use builtins::*;
pub use mcp::{MCP_NAME_SEPARATOR, MCPConnection, MCPToolEntry};
pub(crate) use mcp::{mcp_tool_desc, prefixed_tool_name};
