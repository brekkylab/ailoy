use std::sync::Arc;

use crate::{message::ToolDesc, tool::ToolFunc};

#[derive(Clone)]
pub struct ToolRuntime {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl ToolRuntime {
    pub(crate) fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self { desc, f }
    }

    pub fn get_desc(&self) -> &ToolDesc {
        &self.desc
    }

    pub fn get_func(&self) -> Arc<ToolFunc> {
        self.f.clone()
    }
}
