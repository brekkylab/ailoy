mod builtin;
mod mcp;

use std::fmt::Debug;
use std::sync::Arc;

use crate::value::{Part, ToolCallArg, ToolDesc};

pub use builtin::*;
pub use mcp::*;

#[cfg(not(target_arch = "wasm32"))]
pub trait Tool: Send + Sync + Debug + 'static {
    fn get_description(&self) -> ToolDesc;

    fn run(
        self: Arc<Self>,
        args: ToolCallArg,
    ) -> futures::future::BoxFuture<'static, Result<Vec<Part>, String>>;
}

#[cfg(target_arch = "wasm32")]
pub trait Tool: Debug + 'static {
    fn get_description(&self) -> ToolDesc;

    fn run(
        self: Arc<Self>,
        args: ToolCallArg,
    ) -> futures::future::LocalBoxFuture<'static, Result<Vec<Part>, String>>;
}
