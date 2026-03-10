pub(crate) mod anthropic;
pub(crate) mod chat_completion;
pub(crate) mod gemini;
pub(crate) mod openai;
#[cfg(test)]
pub(crate) mod test_helpers;

use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use crate::lang_model::language_model::ThinkEffort;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RequestConfig {
    pub model: Option<String>,
    pub system_message: Option<String>,
    pub stream: bool,
    pub think_effort: ThinkEffort,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, EnumString, Display)]
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

    pub fn env_var(&self) -> &'static str {
        match self {
            APISpecification::Claude => "ANTHROPIC_API_KEY",
            APISpecification::OpenAI | APISpecification::Responses => "OPENAI_API_KEY",
            APISpecification::Gemini => "GEMINI_API_KEY",
            APISpecification::ChatCompletion => "OPENAI_API_KEY",
            APISpecification::Grok => "XAI_API_KEY",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerEvent {
    pub event: String,
    pub data: String,
}

pub(super) fn drain_next_event(buf: &mut Vec<u8>) -> Option<ServerEvent> {
    // Try \n\n delimiter
    if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
        let event_data = buf[..pos].to_vec();
        buf.drain(..pos + 2);
        return Some(parse_server_event(event_data));
    }
    // Try \r\n\r\n delimiter (also valid per SSE spec)
    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
        let event_data = buf[..pos].to_vec();
        buf.drain(..pos + 4);
        return Some(parse_server_event(event_data));
    }
    None
}

fn parse_server_event(raw: Vec<u8>) -> ServerEvent {
    let text = String::from_utf8_lossy(&raw);
    let mut event: String = String::new();
    let mut data_lines: Vec<String> = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim().to_string());
        }
    }
    ServerEvent {
        event,
        data: if data_lines.is_empty() {
            String::new()
        } else {
            data_lines.join("\n")
        },
    }
}
