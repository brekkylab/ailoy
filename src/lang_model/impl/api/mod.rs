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
/// [`LangModelAPISchema`] to its implementation once and reuse it for whole-
/// response unmarshaling, per-event (SSE) unmarshaling, and 429 classification.
///
/// Both parsers live on the provider's [`Unmarshal<MessageDeltaOutput>`] impl
/// (`unmarshal` for a whole response, `unmarshal_event` for one SSE event); this
/// trait just re-exposes them for dynamic dispatch, since `Unmarshal: Default`
/// isn't object-safe and so can't be a supertrait of a `dyn` type.
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
        LangModelAPISchema::ChatCompletion => Box::new(ChatCompletionUnmarshal),
        LangModelAPISchema::Gemini => Box::new(GeminiUnmarshal),
        LangModelAPISchema::OpenAI => Box::new(OpenAIUnmarshal),
    }
}
