pub mod builtin;
pub mod mcp;

use std::fmt::Debug;

use ailoy_macros::multi_platform_async_trait;
pub use builtin::*;
use downcast_rs::{Downcast, impl_downcast};
pub use mcp::*;

use crate::{
    utils::{MaybeSend, MaybeSync},
    value::{ToolDesc, Value},
};

#[multi_platform_async_trait]
pub trait Tool: Debug + Downcast + MaybeSend + MaybeSync + 'static {
    fn get_description(&self) -> ToolDesc;

    async fn run(&self, args: Value) -> Result<Value, String>;
}

impl_downcast!(Tool);
