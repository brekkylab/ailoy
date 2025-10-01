pub mod builtin;
pub mod mcp;

use std::{any::TypeId, fmt::Debug, sync::Arc};

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
pub use builtin::*;
use downcast_rs::{Downcast, impl_downcast};
pub use mcp::*;

use crate::value::{ToolDesc, Value};

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait Tool: Debug + Downcast {
    fn get_description(&self) -> ToolDesc;

    async fn run(&self, args: Value) -> Result<Value, String>;
}

impl_downcast!(Tool);

#[derive(Clone)]
pub struct ArcTool {
    pub inner: Arc<dyn Tool>,
    pub type_id: TypeId,
}

impl ArcTool {
    pub fn new<T: Tool + 'static>(tool: T) -> Self {
        Self {
            inner: Arc::new(tool),
            type_id: TypeId::of::<T>(),
        }
    }

    pub fn new_from_arc<T: Tool + 'static>(tool: Arc<T>) -> Self {
        Self {
            inner: tool,
            type_id: TypeId::of::<T>(),
        }
    }

    pub fn new_from_arc_any(tool: Arc<dyn Tool>) -> Self {
        Self {
            inner: tool.clone(),
            type_id: tool.type_id(),
        }
    }

    pub fn type_of<T: Tool + 'static>(&self) -> bool {
        TypeId::of::<T>() == self.type_id
    }

    pub fn downcast<T: Tool + 'static>(&self) -> Result<Arc<T>, String> {
        if !self.type_of::<T>() {
            return Err(format!(
                "TypeId of provided type is not same as {:?}",
                self.type_id
            ));
        }
        let arc_ptr = Arc::into_raw(self.inner.clone());
        let arc_model = unsafe { Arc::from_raw(arc_ptr as *const T) };
        Ok(arc_model)
    }
}
