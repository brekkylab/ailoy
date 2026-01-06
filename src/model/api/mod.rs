pub(crate) mod anthropic;
pub(crate) mod chat_completion;
pub(crate) mod gemini;
pub(crate) mod openai;
pub(crate) mod stream;

pub use stream::StreamAPILangModel;

use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use crate::model::language_model::ThinkEffort;

#[derive(Clone, Debug, PartialEq)]
struct RequestConfig {
    pub model: Option<String>,

    pub system_message: Option<String>,

    pub stream: bool,

    pub think_effort: ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, EnumString, Display)]
#[cfg_attr(feature = "python", derive(ailoy_macros::PyStringEnum))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum APISpecification {
    ChatCompletion,
    OpenAI,
    Gemini,
    Claude,

    // these variants exist as alias of an existing variant.
    Responses, // alias of OpenAI
    Grok,      // alias of ChatCompletion with custom url
}

impl APISpecification {
    pub fn default_url(&self) -> &'static str {
        match self {
            APISpecification::ChatCompletion => "https://api.openai.com/v1/chat/completions",
            APISpecification::OpenAI | APISpecification::Responses => {
                "https://api.openai.com/v1/responses"
            }
            APISpecification::Gemini => "https://generativelanguage.googleapis.com/v1beta/models",
            APISpecification::Claude => "https://api.anthropic.com/v1/messages",
            APISpecification::Grok => "https://api.x.ai/v1/chat/completions",
        }
    }
}
