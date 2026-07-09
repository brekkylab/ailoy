mod anthropic;
mod chat_completion;
mod gemini;
mod openai;

pub use anthropic::{AnthropicMarshal, AnthropicUnmarshal};
pub use chat_completion::{ChatCompletionMarshal, ChatCompletionUnmarshal};
pub use gemini::{GeminiMarshal, GeminiUnmarshal};
pub use openai::{OpenAIMarshal, OpenAIUnmarshal};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    lang_model::LangModelRequest,
    message::{Marshal, Marshaled, MessageDeltaOutput, Unmarshal},
};

/// Wire protocol used when calling a language model API.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LangModelAPISchema {
    /// OpenAI-compatible `/v1/chat/completions` format
    ChatCompletion,

    /// Anthropic Messages API format
    Anthropic,

    /// Google Gemini API format
    Gemini,

    /// OpenAI Responses API format
    #[serde(rename = "openai")]
    OpenAI,
}

impl Default for LangModelAPISchema {
    fn default() -> Self {
        LangModelAPISchema::ChatCompletion
    }
}

impl Marshal<LangModelRequest<'_>> for LangModelAPISchema {
    fn marshal(&self, req: &LangModelRequest) -> Value {
        match self {
            LangModelAPISchema::Anthropic => {
                Value::from(Marshaled::<LangModelRequest, AnthropicMarshal>::new(&req))
            }
            LangModelAPISchema::ChatCompletion => Value::from(Marshaled::<
                LangModelRequest,
                ChatCompletionMarshal,
            >::new(&req)),
            LangModelAPISchema::Gemini => {
                Value::from(Marshaled::<LangModelRequest, GeminiMarshal>::new(&req))
            }
            LangModelAPISchema::OpenAI => {
                Value::from(Marshaled::<LangModelRequest, OpenAIMarshal>::new(&req))
            }
        }
    }
}

impl Unmarshal<MessageDeltaOutput> for LangModelAPISchema {
    fn unmarshal(&self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        match self {
            LangModelAPISchema::Anthropic => AnthropicUnmarshal.unmarshal(val),
            LangModelAPISchema::ChatCompletion => ChatCompletionUnmarshal.unmarshal(val),
            LangModelAPISchema::Gemini => GeminiUnmarshal.unmarshal(val),
            LangModelAPISchema::OpenAI => OpenAIUnmarshal.unmarshal(val),
        }
    }
}

impl LangModelAPISchema {
    /// Classify a `429` response body as permanent quota/credit exhaustion
    /// (which never recovers, so must not be retried) vs a transient rate
    /// limit (retry with backoff). Defaults to transient.
    ///
    /// * OpenAI (both `openai` and OpenAI-compatible `chat_completion`) report
    ///   permanent exhaustion via `insufficient_quota`. Since `chat_completion`
    ///   is shared by arbitrary providers we cannot assume a provider-specific
    ///   signal there, so only the first-party `OpenAI` schema is classified.
    /// * Gemini reports `RESOURCE_EXHAUSTED`; a `RetryInfo` detail marks the
    ///   transient case.
    /// * Anthropic reports billing exhaustion as `402`, outside this `429`
    ///   path, so its `429`s are always transient.
    pub fn is_permanent_quota_error(&self, body: &str) -> bool {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };
        match self {
            LangModelAPISchema::OpenAI => {
                let error = &json["error"];
                error["type"] == "insufficient_quota" || error["code"] == "insufficient_quota"
            }
            LangModelAPISchema::Gemini => {
                let error = &json["error"];
                // RESOURCE_EXHAUSTED covers both; a RetryInfo detail marks the transient case.
                error["status"] == "RESOURCE_EXHAUSTED"
                    && !error["details"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .any(|d| {
                            d["@type"]
                                .as_str()
                                .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
                        })
            }
            LangModelAPISchema::Anthropic | LangModelAPISchema::ChatCompletion => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_permanent_quota_error() {
        let schema = LangModelAPISchema::OpenAI;
        let quota = r#"{"error":{"type":"insufficient_quota","code":"insufficient_quota"}}"#;
        let rate = r#"{"error":{"type":"rate_limit_exceeded","code":"rate_limit_exceeded"}}"#;
        assert!(schema.is_permanent_quota_error(quota));
        assert!(!schema.is_permanent_quota_error(rate));
        // Unparseable body is treated as transient (don't suppress retries).
        assert!(!schema.is_permanent_quota_error("not json"));
    }

    #[test]
    fn gemini_permanent_quota_error() {
        let schema = LangModelAPISchema::Gemini;
        let quota = r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}"#;
        let rate = r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34s"}]}}"#;
        assert!(schema.is_permanent_quota_error(quota));
        assert!(!schema.is_permanent_quota_error(rate));
        assert!(!schema.is_permanent_quota_error("not json"));
    }

    #[test]
    fn other_schemas_treat_429_as_transient() {
        assert!(!LangModelAPISchema::Anthropic.is_permanent_quota_error(
            r#"{"error":{"type":"insufficient_quota"}}"#
        ));
        assert!(!LangModelAPISchema::ChatCompletion.is_permanent_quota_error(
            r#"{"error":{"type":"insufficient_quota"}}"#
        ));
    }
}
