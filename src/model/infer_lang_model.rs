use futures::{future::BoxFuture, stream::BoxStream};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{Message, MessageDeltaOutput, MessageOutput, TokenUsage},
    tool::ToolDesc,
};

/// How much a model thinks before it answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingEffort {
    Low,
    Medium,
    High,
}

/// Settings every [`InferLangModel`] can apply, whichever way it runs the model.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LangModelOptions {
    /// How much the model thinks; `None` keeps the model's or backend's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,

    /// JSON schema the output must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
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

    /// How many tokens of input the model read for an output that reported `usage`, cached or
    /// not, which a [`ContextManager`](crate::agent::ContextManager) checks against its limit.
    ///
    /// The default is `usage.input_tokens`, the whole input for a provider that counts what it
    /// read from or wrote to a prompt cache inside it. A backend whose provider reports those
    /// apart adds them.
    fn total_input_tokens(&self, usage: &TokenUsage) -> u64 {
        usage.input_tokens
    }
}
