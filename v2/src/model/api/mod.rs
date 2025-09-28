pub mod anthropic;
pub mod chat_completion;
pub mod gemini;
pub mod openai;
pub mod sse;
// pub mod xai;

#[derive(Debug, Clone)]
pub enum APIProvider {
    OpenAI,
    Google,
    Anthropic,
    XAI,
}

#[derive(Debug, Clone)]
pub struct APIModel {
    pub provider: APIProvider,
    pub model: String,
    pub api_key: String,
}

impl APIModel {
    fn new(provider: APIProvider, model: impl Into<String>, api_key: impl Into<String>) -> Self {
        APIModel {
            provider,
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    pub fn endpoint(&self) -> &'static str {
        match self.provider {
            APIProvider::OpenAI => "https://api.openai.com/v1/responses",
            APIProvider::Google => "https://generativelanguage.googleapis.com/v1beta/models",
            APIProvider::Anthropic => "https://api.anthropic.com/v1/messages",
            APIProvider::XAI => "https://api.x.ai/v1/chat/completions",
        }
    }
}
