use futures::{future::BoxFuture, stream::BoxStream};

pub use ailoy_core::model::{LangModelOptions, ThinkingEffort};

use crate::{
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

/// A request the model's API refused as sent: a 4xx saying the request itself is wrong,
/// not that the server is busy or the caller unknown. Sending it again as is gains nothing;
/// what is in it has to change.
#[derive(Clone, Debug)]
pub struct RequestRefused {
    pub status: u16,
    pub message: String,
}

impl std::fmt::Display for RequestRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestRefused {}

/// The error for a request the API answered with `status`: a [`RequestRefused`] when the
/// status says the request itself was wrong, else a plain error with the same message.
pub(crate) fn api_error(status: u16, message: String) -> anyhow::Error {
    if matches!(status, 400 | 413 | 422) {
        RequestRefused { status, message }.into()
    } else {
        anyhow::anyhow!(message)
    }
}

/// A language model that can be run on a conversation.
pub trait InferLangModel: Send + Sync {
    /// Run the model to completion and return the whole output.
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>>;

    /// Run the model and yield its output as deltas while it is generated.
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>>;
}
