use serde::{Deserialize, Serialize};
use url::Url;

#[cfg(feature = "rt")]
pub use rt::*;

/// Describes the logical properties of a language model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LangModelDesc {
    /// Model identifier (e.g. "gpt-4", "claude-3-opus", etc.)
    pub model: String,

    /// System message used when invoking the model.
    pub system_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LangModelAPISpec {
    ChatCompletion,
    Anthropic,
}

/// Specifies how a language model is executed.
///
/// This describes the runtime provider required to actually run the model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LangModelProvider {
    #[serde(rename = "api")]
    API {
        spec: LangModelAPISpec,

        url: Url,

        api_key: Option<String>,
    },

    #[serde(rename = "local")]
    Local {},
}

#[cfg(feature = "rt")]
mod rt {
    pub use super::*;

    /// Runtime
    pub struct LangModel {
        desc: LangModelDesc,
        provider: LangModelProvider,
    }
}
