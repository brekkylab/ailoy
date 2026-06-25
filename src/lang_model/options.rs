use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LangModelOptions {
    /// Maximum number of tokens the model may generate in a single response.
    /// When `None`, provider-specific defaults apply (e.g. Anthropic defaults
    /// to 8192). Set explicitly to cap output length per call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Sampling temperature passed to the language model on every call.
    /// `None` leaves the provider default in place. Not every provider
    /// supports the same range; values outside the provider's accepted
    /// range will surface as API errors.
    ///
    /// In practice `temperature` is the only sampling knob most callers ever
    /// need to touch — [`top_p`](Self::top_p) and [`top_k`](Self::top_k) are
    /// rarely used and are exposed mainly for parity with provider APIs.
    /// Prefer adjusting `temperature` alone unless you have a specific reason
    /// to combine it with nucleus or top-k sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus (top-p) sampling parameter passed to the language model on every
    /// call. Rarely needed in practice — see [`temperature`](Self::temperature).
    /// Honoured by all supported providers (Anthropic, Gemini, OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling parameter passed to the language model on every call.
    /// Rarely needed in practice — see [`temperature`](Self::temperature).
    /// Only honoured by providers that support it (e.g. Anthropic, Gemini);
    /// silently ignored by providers that do not (e.g. OpenAI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,

    /// Constrains the model's output to a JSON schema validated at construction time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<super::ResponseFormat>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl LangModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}
