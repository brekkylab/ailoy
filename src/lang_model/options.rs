use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::datatype::Value;

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
    pub response_format: Option<ResponseFormat>,
}

impl LangModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Constrains the model's response to a specific JSON format.
///
/// Constructed via [`ResponseFormat::json_schema`], which validates the schema
/// against JSON Schema Draft 7 before storing it, then normalises it to satisfy
/// provider-specific requirements.  The stored schema is provider-agnostic;
/// each marshal converts it to the wire format expected by its API.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "schema", rename_all = "snake_case")]
pub enum ResponseFormat {
    JsonSchema(Value),
}

impl ResponseFormat {
    /// Validate `schema` against JSON Schema Draft 7.  Returns `Err` if the
    /// schema is structurally invalid (e.g. `"type": 123`).  The stored schema
    /// is the user's original; provider-specific transformations happen in each
    /// marshal via [`ResponseSchemaMarshal::marshal_response_schema`].
    pub fn json_schema(schema: Value) -> anyhow::Result<Self> {
        let serde_schema: serde_json::Value = schema.clone().into();
        jsonschema::validator_for(&serde_schema)
            .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?;
        Ok(Self::JsonSchema(schema))
    }
}

impl schemars::JsonSchema for ResponseFormat {
    fn schema_name() -> String {
        "ResponseFormat".into()
    }

    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::{InstanceType, ObjectValidation, SchemaObject, SingleOrVec};
        SchemaObject {
            instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Object))),
            object: Some(Box::new(ObjectValidation {
                required: ["type".to_owned()].into(),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}
