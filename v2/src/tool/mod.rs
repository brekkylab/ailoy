mod builtin;
mod mcp;

use std::fmt::Debug;
use std::sync::Arc;

use crate::{
    utils::BoxFuture,
    value::{Part, ToolCallArg, ToolDesc},
};

pub use builtin::*;
pub use mcp::*;

pub trait Tool: Debug + 'static {
    fn get_description(&self) -> ToolDesc;

    fn run(self: Arc<Self>, args: ToolCallArg) -> BoxFuture<'static, Result<Vec<Part>, String>>;
}
