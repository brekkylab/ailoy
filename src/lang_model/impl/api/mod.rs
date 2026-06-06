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
