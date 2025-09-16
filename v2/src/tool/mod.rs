mod builtin;
mod mcp;

use std::fmt::Debug;

use ailoy_macros::multi_platform_async_trait;
pub use builtin::*;
pub use mcp::*;

use crate::value::{Part, ToolCallArg, ToolDesc};

#[multi_platform_async_trait]
pub trait Tool: Debug + 'static {
    fn get_description(&self) -> ToolDesc;

    async fn run(&self, args: ToolCallArg) -> Result<Vec<Part>, String>;
}
