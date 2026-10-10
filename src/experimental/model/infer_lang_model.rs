use futures::{future::BoxFuture, stream::BoxStream};

use crate::{
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

/// A language model that can be run on a conversation.
pub trait InferLangModel: Send + Sync {
    // fn from_desc(desc: String) -> anyhow::Result<Self>
    // where
    //     Self: Sized;

    /// Run the model to completion and return the whole output.
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>>;

    /// Run the model and yield its output as deltas while it is generated.
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>>;
}
