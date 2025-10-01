pub mod anthropic;
pub mod chat_completion;
pub mod gemini;
pub mod openai;
mod stream;
// pub mod xai;

pub(super) use stream::*;

#[derive(Clone, Debug, PartialEq)]
struct RequestConfig {
    pub model: Option<String>,

    pub system_message: Option<String>,

    pub stream: bool,

    pub think_effort: crate::model::ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone)]
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
