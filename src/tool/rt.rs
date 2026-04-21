use std::sync::Arc;

use futures::stream::BoxStream;

use super::func::ToolContext;
use crate::{
    message::{MessageOutput, ToolDesc},
    tool::ToolFunc,
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
    pub fn call(
        &self,
        args: &crate::message::Part,
        ctx: ToolContext,
    ) -> anyhow::Result<BoxStream<'static, MessageOutput>> {
        self.f.call(args.clone(), ctx)
    }
}

#[cfg(all(test, feature = "sandbox"))]
pub(crate) mod test_helpers {
    use std::sync::Arc;

    use futures::StreamExt as _;

    use crate::{
        message::{Message, Part},
        sandbox::Sandbox,
        tool::{Tool, ToolContext},
    };

    pub async fn call_with_sandbox(tool: &Tool, args: Part, sandbox: Arc<Sandbox>) -> Message {
        tool.call(
            &args,
            ToolContext {
                sandbox: Some(sandbox),
            },
        )
        .unwrap()
        .next()
        .await
        .unwrap()
        .message
    }
}
