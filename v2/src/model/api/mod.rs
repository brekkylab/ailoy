use crate::model::ThinkEffort;

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

    pub think_effort: ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone)]
pub enum APIProvider {
    OpenAI,
    Google,
    Anthropic,
    XAI,
}

#[derive(Debug, Clone)]
pub struct APIUsage {
    pub provider: APIProvider,
    pub model: String,
    pub api_key: String,
}

impl APIUsage {
    fn new(provider: APIProvider, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        APIUsage {
            provider,
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    pub fn default_url(&self) -> &'static str {
        match self.provider {
            APIProvider::OpenAI => "https://api.openai.com/v1/responses",
            APIProvider::Google => "https://generativelanguage.googleapis.com/v1beta/models",
            APIProvider::Anthropic => "https://api.anthropic.com/v1/messages",
            APIProvider::XAI => "https://api.x.ai/v1/chat/completions",
        }
    }
}
