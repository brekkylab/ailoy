mod provider;

use futures::{future::BoxFuture, stream::BoxStream};
pub use provider::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{Message, MessageDeltaOutput, MessageOutput},
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

/// Settings every model can apply, whichever way it is run.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LangModelOptions {
    /// How much the model thinks; `None` keeps the model's or backend's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,

    /// JSON schema the output must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
}

/// Which end provides the key an agent's model is called with. The model is called where the
/// agent runs either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ModelKeyProvider {
    /// The server, with the key it was set up with.
    #[default]
    Server,
    /// The client, which sends its key with each run.
    Client,
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
