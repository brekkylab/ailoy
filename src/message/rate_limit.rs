use serde::{Deserialize, Serialize};

/// One rate-limit window as a provider reports it in response headers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimitWindow {
    pub limit: Option<u64>,
    pub remaining: Option<u64>,
    /// When the window is fully replenished, as Unix epoch milliseconds.
    pub reset_at_ms: Option<u64>,
}

impl RateLimitWindow {
    fn is_empty(&self) -> bool {
        self.limit.is_none() && self.remaining.is_none() && self.reset_at_ms.is_none()
    }
}

/// Rate-limit headroom read off one response. Anthropic reports requests, unified tokens,
/// input tokens and output tokens; OpenAI-shaped APIs report requests and tokens; Gemini
/// and Bedrock report nothing, and are `None` upstream rather than an empty value here.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimitInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requests: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<RateLimitWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<RateLimitWindow>,
}

impl RateLimitInfo {
    pub fn is_empty(&self) -> bool {
        [
            &self.requests,
            &self.tokens,
            &self.input_tokens,
            &self.output_tokens,
        ]
        .iter()
        .all(|w| w.as_ref().is_none_or(|w| w.is_empty()))
    }
}
