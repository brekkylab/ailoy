pub(crate) mod base;
pub(crate) mod builtin;
pub(crate) mod function;
pub(crate) mod mcp;

pub use base::{Tool, ToolBehavior};
pub use builtin::{
    BuiltinToolKind, create_terminal_tool, create_web_fetch_tool, create_web_search_duckduckgo_tool,
};
pub use function::{FunctionTool, ToolFunc, ToolFuncResult};
pub use mcp::{MCPClient, MCPTool};
