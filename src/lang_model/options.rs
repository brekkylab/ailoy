use std::borrow::Cow;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::datatype::Value;

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LangModelOptions {
    /// Output token cap per response; `None` uses the provider default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Sampling temperature; `None` keeps the provider default, and values outside the
    /// provider's range surface as API errors. Usually the only sampling knob worth setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    /// Nucleus (top-p) sampling; rarely needed, see [`temperature`](Self::temperature).
    /// Dropped for OpenAI reasoning models and for Anthropic/Bedrock while thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Top-k sampling; rarely needed, see [`temperature`](Self::temperature).
    /// Silently ignored by providers that do not support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,

    /// Constrains the model's output to a JSON schema validated at construction time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Turns on the model's thinking at the given effort, and asks the provider to return
    /// it, which lands in [`Message::thinking`](crate::message::Message::thinking). `None`
    /// sends nothing and leaves the provider default in place: some models think anyway,
    /// most do not. Claude drops `temperature`, `top_p` and `top_k` while thinking; a model
    /// that cannot think fails with an API error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningEffort>,
}

impl LangModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// How much the model thinks before it answers.  See
/// [`LangModelOptions::reasoning`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// The name every provider that takes an effort level uses for it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    /// A thinking-token budget for providers that take one instead of a level.
    pub(crate) fn budget_tokens(self) -> u64 {
        match self {
            Self::Low => 2048,
            Self::Medium => 8192,
            Self::High => 24576,
        }
    }
}

impl std::str::FromStr for ReasoningEffort {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => anyhow::bail!("reasoning effort must be low, medium or high, not {other:?}"),
        }
    }
}

/// JSON schema the response must match, stored provider-agnostic; each marshal adapts
/// it to its API.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "schema", rename_all = "snake_case")]
pub enum ResponseFormat {
    JsonSchema(Value),
}

impl ResponseFormat {
    /// Validates `schema` against JSON Schema Draft 7, failing if it is structurally
    /// invalid (e.g. `"type": 123`).
    pub fn json_schema(schema: Value) -> anyhow::Result<Self> {
        let serde_schema: serde_json::Value = schema.clone().into();
        jsonschema::validator_for(&serde_schema)
            .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?;
        Ok(Self::JsonSchema(schema))
    }
}

impl schemars::JsonSchema for ResponseFormat {
    fn schema_name() -> Cow<'static, str> {
        "ResponseFormat".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "object",
            "required": ["type"]
        })
    }
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;

    use super::ResponseFormat;

    #[test]
    fn json_schema_preserves_tagged_object_contract() {
        let schema =
            <ResponseFormat as JsonSchema>::json_schema(&mut schemars::SchemaGenerator::default());

        assert_eq!(
            serde_json::to_value(schema).unwrap(),
            serde_json::json!({
                "type": "object",
                "required": ["type"]
            })
        );
    }
}
