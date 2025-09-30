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
