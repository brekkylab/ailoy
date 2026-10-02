mod anthropic;
pub(crate) mod bedrock;
mod chat_completion;
mod gemini;
mod openai;

pub use anthropic::{AnthropicMarshal, AnthropicUnmarshal};
pub use bedrock::{BedrockMarshal, BedrockRegion, BedrockUnmarshal};
pub use chat_completion::{ChatCompletionMarshal, ChatCompletionUnmarshal};
pub use gemini::{GeminiMarshal, GeminiUnmarshal};
pub use openai::{OpenAIMarshal, OpenAIUnmarshal};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{MessageDeltaOutput, Unmarshal},
};

/// Wire protocol used when calling a language model API.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum LangModelAPISchema {
    /// OpenAI-compatible `/v1/chat/completions` format
    #[default]
    ChatCompletion,

    /// Anthropic Messages API format
    Anthropic,

    /// Amazon Bedrock Converse API format, one body for every model family
    /// Bedrock serves. The provider `url` is the runtime base
    /// (`https://bedrock-runtime.<region>.amazonaws.com`); the model id and
    /// action are appended per request. Authenticated with a Bedrock API key
    /// as a bearer token (no SigV4); streams as a binary event stream.
    Bedrock,

    /// Google Gemini API format
    Gemini,

    /// OpenAI Responses API format
    #[serde(rename = "openai")]
    OpenAI,
}

/// Classifies a 429 body as permanent quota exhaustion (don't retry) vs a
/// transient rate limit. Defaults to transient.
pub trait QuotaClassifier {
    fn is_permanent_quota_error(&self, _body: &str) -> bool {
        false
    }
}

/// Object-safe view of a provider's [`Unmarshal<MessageDeltaOutput>`] and [`QuotaClassifier`],
/// so a [`LangModelAPISchema`] maps to one `dyn` impl; `Unmarshal: Default` can't be a `dyn` supertrait.
pub trait ProviderApi: QuotaClassifier {
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput>;
    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>>;
}

impl<T> ProviderApi for T
where
    T: QuotaClassifier + Unmarshal<MessageDeltaOutput>,
{
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        T::default().unmarshal(val)
    }

    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        <T as Unmarshal<MessageDeltaOutput>>::unmarshal_event(self, data)
    }
}

/// Maps a wire schema to its provider implementation.
pub fn provider_api(schema: &LangModelAPISchema) -> Box<dyn ProviderApi + Send + Sync> {
    match schema {
        LangModelAPISchema::Anthropic => Box::new(AnthropicUnmarshal),
        LangModelAPISchema::Bedrock => Box::new(BedrockUnmarshal),
        LangModelAPISchema::ChatCompletion => Box::new(ChatCompletionUnmarshal),
        LangModelAPISchema::Gemini => Box::new(GeminiUnmarshal),
        LangModelAPISchema::OpenAI => Box::new(OpenAIUnmarshal),
    }
}
