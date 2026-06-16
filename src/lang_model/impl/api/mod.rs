mod anthropic;
mod chat_completion;
mod gemini;
mod openai;

pub use anthropic::{AnthropicMarshal, AnthropicUnmarshal};
pub use chat_completion::{ChatCompletionMarshal, ChatCompletionUnmarshal};
pub use gemini::{GeminiMarshal, GeminiUnmarshal};
pub use openai::{OpenAIMarshal, OpenAIUnmarshal};

use crate::{
    datatype::Value,
    lang_model::LangModelAPISchema,
    message::{MessageDeltaOutput, Unmarshal},
};

/// Classifies a 429 body as permanent quota exhaustion (don't retry) vs a
/// transient rate limit. Defaults to transient.
pub trait QuotaClassifier {
    fn is_permanent_quota_error(&self, _body: &str) -> bool {
        false
    }
}

/// Provider-specific response handling, dispatched dynamically so callers map a
/// [`LangModelAPISchema`] to its implementation once and reuse it for both
/// unmarshaling and 429 classification.
pub trait ProviderApi: QuotaClassifier {
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput>;
}

impl<T> ProviderApi for T
where
    T: QuotaClassifier + Unmarshal<MessageDeltaOutput>,
{
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        T::default().unmarshal(val)
    }
}

/// Maps a wire schema to its provider implementation.
pub fn provider_api(schema: &LangModelAPISchema) -> Box<dyn ProviderApi + Send> {
    match schema {
        LangModelAPISchema::Anthropic => Box::new(AnthropicUnmarshal),
        LangModelAPISchema::ChatCompletion => Box::new(ChatCompletionUnmarshal),
        LangModelAPISchema::Gemini => Box::new(GeminiUnmarshal),
        LangModelAPISchema::OpenAI => Box::new(OpenAIUnmarshal),
    }
}
