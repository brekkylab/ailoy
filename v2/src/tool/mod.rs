mod builtin;
mod mcp;

use std::sync::Arc;

use crate::value::{Part, ToolCall, ToolDescription};

pub use builtin::*;
use futures::future::BoxFuture;
pub use mcp::*;

pub trait Tool: Send + Sync + 'static {
    fn get_description(&self) -> ToolDescription;

    fn run(self: Arc<Self>, tc: ToolCall) -> BoxFuture<'static, Result<Vec<Part>, String>>;
}
