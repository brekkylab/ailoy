use std::fmt;

use serde::{Deserialize, Serialize};

use crate::datatype::Value;

/// Describes a **tool** (or function) that a language model can invoke.
///
/// `ToolDesc` defines the schema, behavior, and input/output specification of a callable
/// external function, allowing an LLM to understand how to use it.
///
/// The primary role of this struct is to describe to the LLM what a *tool* does,
/// how it can be invoked, and what input (`parameters`) and output (`returns`) schemas it expects.
///
/// The format follows the same **schema conventions** used by Hugging Face’s
/// `transformers` library, as well as APIs such as *OpenAI* and *Anthropic*.
/// The `parameters` and `returns` fields are typically defined using **JSON Schema**.
///
/// We provide a builder [`ToolDescBuilder`] helper for convenient and fluent construction.
/// Please refer to [`ToolDescBuilder`].
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

/// Longest tool name the stricter of the two model APIs (OpenAI) accepts.
pub const MAX_TOOL_NAME_LEN: usize = 64;

/// Map anything a model API would refuse in a tool name onto `_`.
///
/// OpenAI, Anthropic and Gemini all constrain a function name to
/// `[A-Za-z0-9_-]`, and [`ToolDesc::name`] is handed to them verbatim — so a
/// name that came from somewhere else (an MCP server's tool list, an A2A agent
/// card, a caller's label) has to be brought into that set before a model ever
/// sees it, or the request itself is rejected.
///
/// Sanitising rather than refusing: whatever the name is on the wire is kept
/// separately by the caller that needs it, and one stray character should not
/// cost the caller the whole tool.
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
/// A warning and not an error: which provider this tool will be sent to is not
/// known here, and only the strictest of them draws the line at
/// [`MAX_TOOL_NAME_LEN`].
pub(crate) fn warn_if_tool_name_too_long(name: &str) {
    if name.len() > MAX_TOOL_NAME_LEN {
        log::warn!(
            "tool name '{name}' is {} characters; some model APIs reject names over {MAX_TOOL_NAME_LEN}",
            name.len()
        );
    }
}
