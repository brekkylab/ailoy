use futures::{future::BoxFuture, stream::BoxStream};

use crate::{
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

/// Sampling, reasoning and output-format options differ by vendor, so each implementor
/// carries its own rather than taking them here.
pub trait LangModelInference: Send + Sync {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>>;

    fn infer<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>>;
}
