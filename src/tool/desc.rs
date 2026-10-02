use std::fmt;

use serde::{Deserialize, Serialize};

use crate::datatype::Value;

/// A tool as described to a language model: name, description, and JSON Schema
/// for its parameters and (optional) return value, in the convention OpenAI,
/// Anthropic and Hugging Face `transformers` share. Build with [`ToolDescBuilder`].
///
/// # Example
/// ```rust
/// # use ailoy::tool::ToolDescBuilder;
/// # use ailoy::to_value;
///
/// let desc = ToolDescBuilder::new("temperature")
///     .description("Get the current temperature for a given city")
///     .parameters(to_value!({
///         "type": "object",
///         "properties": {
///             "location": {
///                 "type": "string",
///                 "description": "The city name"
///             },
///             "unit": {
///                 "type": "string",
///                 "description": "Temperature unit (default: Celsius)",
///                 "enum": ["Celsius", "Fahrenheit"]
///             }
///         },
///         "required": ["location"]
///     }))
///     .returns(to_value!({
///         "type": "number"
///     }))
///     .build();
///
/// assert_eq!(desc.name, "temperature");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ToolDesc {
    /// The unique name of the tool or function.
    pub name: String,

    /// A natural-language description of what the tool does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// A [`Value`] describing the JSON Schema of the expected parameters.
    /// Typically an object schema such as `{ "type": "object", "properties": ... }`.
    pub parameters: Value,

    /// An optional [`Value`] that defines the return value schema.  
    /// If omitted, the tool is assumed to return free-form text or JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<Value>,
}

impl ToolDesc {
    pub fn new(
        name: String,
        description: Option<String>,
        parameters: Value,
        returns: Option<Value>,
    ) -> Self {
        ToolDesc {
            name,
            description,
            parameters,
            returns,
        }
    }
}

impl fmt::Display for ToolDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "ToolDesc {}", s)
    }
}

/// A builder for constructing [`ToolDesc`] objects.
///
/// Provides a fluent, chainable API for creating a tool description safely and clearly.
/// If no `parameters` are provided, it defaults to [`Value::Null`].
///
/// # Example
/// ```rust
/// # use ailoy::tool::ToolDescBuilder;
/// # use ailoy::datatype::Value;
///
/// let tool = ToolDescBuilder::new("weather")
///     .description("Fetch current weather information")
///     .parameters(Value::Null)
///     .returns(Value::Null)
///     .build();
///
/// assert_eq!(tool.name, "weather");
/// ```
#[derive(Clone, Debug)]
pub struct ToolDescBuilder {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub returns: Option<Value>,
}

impl ToolDescBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: None,
            returns: None,
        }
    }

    pub fn description(self, desc: impl Into<String>) -> Self {
        Self {
            name: self.name,
            description: Some(desc.into()),
            parameters: self.parameters,
            returns: self.returns,
        }
    }

    pub fn parameters(self, param: impl Into<Value>) -> Self {
        Self {
            name: self.name,
            description: self.description,
            parameters: Some(param.into()),
            returns: self.returns,
        }
    }

    pub fn returns(self, ret: impl Into<Value>) -> Self {
        Self {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
            returns: Some(ret.into()),
        }
    }

    pub fn build(self) -> ToolDesc {
        ToolDesc {
            name: self.name,
            description: self.description,
            parameters: match self.parameters {
                Some(p) => p,
                None => Value::Null,
            },
            returns: self.returns,
        }
    }
}

/// Longest tool name accepted by OpenAI, the strictest model API on this limit.
pub const MAX_TOOL_NAME_LEN: usize = 64;

/// Replace every char outside `[A-Za-z0-9_-]` (the function-name set OpenAI,
/// Anthropic and Gemini accept) with `_`, so a name from an MCP server, an A2A
/// card, or a caller's label can't get the request rejected.
///
/// Sanitizes rather than refuses: callers keep the wire name separately.
pub fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Warn when `name` is longer than every model API will accept.
///
/// A warning, not an error: the target provider is unknown here.
pub(crate) fn warn_if_tool_name_too_long(name: &str) {
    if name.len() > MAX_TOOL_NAME_LEN {
        log::warn!(
            "tool name '{name}' is {} characters; some model APIs reject names over {MAX_TOOL_NAME_LEN}",
            name.len()
        );
    }
}
