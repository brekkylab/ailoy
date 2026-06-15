mod anthropic;
mod chat_completion;
mod gemini;
mod openai;

pub use anthropic::{AnthropicMarshal, AnthropicUnmarshal};
pub use chat_completion::{ChatCompletionMarshal, ChatCompletionUnmarshal};
pub use gemini::{GeminiMarshal, GeminiUnmarshal};
pub use openai::{OpenAIMarshal, OpenAIUnmarshal};

/// Classifies a 429 body as permanent quota exhaustion (don't retry) vs a
/// transient rate limit. Defaults to transient.
pub trait QuotaClassifier {
    fn is_permanent_quota_error(&self, _body: &str) -> bool {
        false
    }
}
