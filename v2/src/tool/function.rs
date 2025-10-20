use std::{
    fmt::{self, Debug, Formatter},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};

use crate::{
    tool::ToolBehavior,
    value::{ToolDesc, Value},
};

#[maybe_send_sync]
pub type ToolFunc = dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value, anyhow::Error>>>>;

#[derive(Clone)]
pub struct FunctionTool {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl Debug for FunctionTool {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionTool")
            .field("desc", &self.desc)
            .field("f", &"(Function)")
            .finish()
    }
}

impl FunctionTool {
    pub fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        FunctionTool { desc, f }
    }
}

#[multi_platform_async_trait]
impl ToolBehavior for FunctionTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        self.f.clone()(args).await
    }
}
