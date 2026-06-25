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

/// Parses a single streaming (SSE) event payload into a [`MessageDeltaOutput`]
/// delta.
///
/// This is distinct from [`Unmarshal<MessageDeltaOutput>`], which parses a whole
/// non-streaming response: the on-the-wire shape of a stream event differs from
/// the final response for every provider (e.g. Anthropic `content_block_delta`
/// events vs. a complete `content` array; ChatCompletion `choices[].delta` vs.
/// `choices[].message`), so streaming needs its own parser.
///
/// `data` is the raw `data:` field of one SSE event. Control events that carry
/// no delta (e.g. OpenAI's `[DONE]` sentinel, Anthropic `ping`) return
/// `Ok(None)`. The default implementation reports that streaming is unsupported,
/// so a provider opts in by overriding this method.
pub trait StreamUnmarshal {
    fn unmarshal_event(&self, _data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        Err(anyhow::anyhow!(
            "streaming (SSE) is not supported for this provider"
        ))
    }
}

/// Provider-specific response handling, dispatched dynamically so callers map a
/// [`LangModelAPISchema`] to its implementation once and reuse it for both
/// unmarshaling and 429 classification.
pub trait ProviderApi: QuotaClassifier + StreamUnmarshal {
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput>;
}

impl<T> ProviderApi for T
where
    T: QuotaClassifier + StreamUnmarshal + Unmarshal<MessageDeltaOutput>,
{
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        T::default().unmarshal(val)
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
