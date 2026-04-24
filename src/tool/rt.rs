use std::sync::Arc;

use futures::stream::BoxStream;

use crate::{
    datatype::Value,
    message::{MessageOutput, ToolDesc},
    tool::{ToolContext, ToolFunc},
};

#[derive(Clone)]
pub struct Tool {
    desc: ToolDesc,
    f: Arc<ToolFunc>,
}

impl Tool {
    pub(crate) fn new(desc: ToolDesc, f: Arc<ToolFunc>) -> Self {
        Self { desc, f }
    }

    pub fn get_desc(&self) -> &ToolDesc {
        &self.desc
    }

    pub fn get_func(&self) -> Arc<ToolFunc> {
        self.f.clone()
    }

    /// Execute this tool and return a stream of [`MessageOutput`] items.
    ///
    /// The agent iterates the full stream to forward intermediate sub-agent
    /// outputs and collect the final tool result.  For simple tools that emit
    /// exactly one item, callers can just call `.next().await`.
    pub fn call(&self, args: Value, ctx: ToolContext) -> BoxStream<'static, MessageOutput> {
        self.f.call(args, ctx)
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use futures::StreamExt as _;

    use crate::{
        datatype::Value,
        message::Message,
        tool::{Tool, ToolContext},
    };

    impl Tool {
        pub async fn call_next(&self, args: Value, ctx: ToolContext) -> Message {
            self.call(args, ctx).next().await.unwrap().message
        }
    }
}
