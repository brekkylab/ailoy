mod builtins;
mod mcp;

pub(crate) use builtins::*;
pub use builtins::{WebSearchProvider, get_web_search_provider, set_web_search_provider};
pub use mcp::{MCP_NAME_SEPARATOR, MCPConnection, MCPToolEntry};
pub(crate) use mcp::{mcp_tool_desc, prefixed_tool_name};
