mod builtin;
mod mcp;

use std::sync::Arc;

use crate::value::{Part, ToolCall, ToolDesc};

pub use builtin::*;
use futures::future::BoxFuture;
pub use mcp::*;

pub trait Tool: Send + Sync + 'static {
    fn get_description(&self) -> ToolDesc;

    fn run(self: Arc<Self>, tc: ToolCall) -> BoxFuture<'static, Result<Part, String>>;
}
