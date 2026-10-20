use crate::model::RequestRefused;

mod bedrock_api;
mod claude_api;
mod claude_cli;
mod codex_cli;
mod deepseek_api;
mod gemini_api;
mod gemini_cli;
mod glm_api;
mod gpt_api;
mod kimi_api;
mod openrouter_api;
mod qwen_api;
mod vllm;

pub use bedrock_api::*;
pub use claude_api::*;
pub use claude_cli::*;
pub use codex_cli::*;
pub use deepseek_api::*;
pub use gemini_api::*;
pub use gemini_cli::*;
pub use glm_api::*;
pub use gpt_api::*;
pub use kimi_api::*;
pub use openrouter_api::*;
pub use qwen_api::*;
pub use vllm::*;

/// The error for a request the API answered with `status`: a [`RequestRefused`] when the
/// status says the request itself was wrong, else a plain error with the same message.
pub(crate) fn api_error(status: reqwest::StatusCode, message: String) -> anyhow::Error {
    if RequestRefused::is_refusal(status.as_u16()) {
        RequestRefused {
            status: status.as_u16(),
            message,
        }
        .into()
    } else {
        anyhow::anyhow!(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_refused_request_is_told_apart() {
        let refused = api_error(reqwest::StatusCode::BAD_REQUEST, "too big".into());
        assert!(refused.downcast_ref::<RequestRefused>().is_some());
        assert_eq!(refused.to_string(), "too big");
        let down = api_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "down".into());
        assert!(down.downcast_ref::<RequestRefused>().is_none());
    }
}
