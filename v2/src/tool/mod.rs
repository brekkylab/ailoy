mod builtin;
mod mcp;

use std::pin::Pin;

use crate::value::{Part, ToolCall, ToolDescription};

pub use builtin::*;
pub use mcp::*;

pub trait Tool: Clone {
    fn get_description(&self) -> ToolDescription;

    fn run(
        self,
        toll_call: ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<Part, String>> + Send + Sync>>;
}

#[derive(Clone, Debug)]
pub enum AnyTool {
    Builtin(builtin::BuiltinTool),
    MCP(mcp::MCPTool),
}

impl Tool for AnyTool {
    fn get_description(&self) -> ToolDescription {
        match self {
            AnyTool::Builtin(t) => t.get_description(),
            AnyTool::MCP(t) => t.get_description(),
        }
    }

    fn run(
        self,
        toll_call: ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<Part, String>> + Send + Sync>> {
        match self {
            AnyTool::Builtin(t) => t.run(toll_call),
            AnyTool::MCP(t) => t.run(toll_call),
        }
    }
}
