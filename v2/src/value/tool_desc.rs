use indexmap::IndexMap;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self},
    ser::SerializeStruct as _,
};

/// A recursive, reduced JSON-Schema–style node that defines the shape of a tool’s
/// parameters and return types. This enum mirrors the subset of schema fields most
/// LLM chat templates (e.g., Hugging Face `apply_chat_template`) expect for tool
/// descriptions.
///
/// # Variants and JSON shape
/// - `String { description, enum }`
///   - Serializes as: `{"type":"string", "description"?: string, "enum"?: [string, ...]}`
///   - `enum` constrains the allowed literal values.
/// - `Number { description }`
///   - Serializes as: `{"type":"number", "description"?: string}`
/// - `Boolean { description }`
///   - Serializes as: `{"type":"boolean", "description"?: string}`
/// - `Object { properties, required }`
///   - Serializes as: `{"type":"object", "properties": {key: <schema>, ...}, "required": [key, ...]}`
///   - `properties` uses `IndexMap` to preserve field order when emitting JSON—useful for
///     models that are sensitive to presentation.
///   - By convention, every `required` key should exist in `properties`.
/// - `Array { items }`
///   - Serializes as: `{"type":"array", "items"?: <schema>}`
///   - If `items` is `None`, any item shape is accepted.
/// - `Null {}`
///   - Serializes as: `{"type":"null"}`
///
/// # Notes
/// - This is a *reduced* dialect, not a full JSON Schema implementation. Only the
///   fields listed above are supported by the serializer/deserializer.
/// - The type is recursive: `Object.properties` and `Array.items` hold nested
///   `ToolDescArg` nodes.
/// - Use builder helpers (`new_string`, `with_desc`, `with_enum`, `with_properties`,
///   `with_items`, …) to construct nodes fluently while keeping the JSON output tidy.
///
/// # Examples
/// ```json
/// {"type":"string","description":"user id","enum":["guest","member","admin"]}
/// {"type":"number"}
/// {"type":"boolean","description":"whether to include archived" }
/// {
///   "type":"object",
///   "properties": {
///     "id":   {"type":"string"},
///     "tags": {"type":"array","items":{"type":"string"}}
///   },
///   "required": ["id"]
/// }
/// ```
///
/// See also: [`ToolDesc`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolDescArg {
    String {
        description: Option<String>,
        r#enum: Option<Vec<String>>,
    },
    Number {
        description: Option<String>,
    },
    Boolean {
        description: Option<String>,
    },
    Object {
        properties: IndexMap<String, Box<ToolDescArg>>,
        required: Vec<String>,
    },
    Array {
        items: Option<Box<ToolDescArg>>,
    },
    Null {},
}

impl ToolDescArg {
    pub fn new_string() -> ToolDescArg {
        ToolDescArg::String {
            description: None,
            r#enum: None,
        }
    }

    pub fn new_number() -> Self {
        Self::Number { description: None }
    }

    pub fn new_boolean() -> Self {
        Self::Boolean { description: None }
    }

    pub fn new_object() -> Self {
        Self::Object {
            properties: IndexMap::new(),
            required: Vec::new(),
        }
    }

    pub fn new_array() -> Self {
        Self::Array { items: None }
    }

    pub fn new_null() -> Self {
        Self::Null {}
    }

    pub fn with_desc(self, description: impl Into<String>) -> Self {
        match self {
            Self::String {
                description: _,
                r#enum,
            } => Self::String {
                description: Some(description.into()),
                r#enum,
            },
            Self::Number { description: _ } => Self::Number {
                description: Some(description.into()),
            },
            Self::Boolean { description: _ } => Self::Boolean {
                description: Some(description.into()),
            },
            _ => self,
        }
    }

    pub fn with_enum(self, choices: impl IntoIterator<Item = impl Into<String>>) -> Self {
        match self {
            Self::String {
                description,
                r#enum: _,
            } => Self::String {
                description,
                r#enum: Some(choices.into_iter().map(|v| v.into()).collect()),
            },
            _ => self,
        }
    }

    pub fn with_properties(
        self,
        properties: impl IntoIterator<Item = (impl Into<String>, impl Into<ToolDescArg>)>,
        required: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        match self {
            Self::Object {
                properties: _,
                required: _,
            } => Self::Object {
                properties: properties
                    .into_iter()
                    .map(|(k, v)| (k.into(), Box::new(v.into())))
                    .collect(),
                required: required.into_iter().map(|v| v.into()).collect(),
            },
            _ => self,
        }
    }

    pub fn with_items(self, items: impl Into<ToolDescArg>) -> Self {
        match self {
            Self::Array { items: _ } => Self::Array {
                items: Some(Box::new(items.into())),
            },
            _ => self,
        }
    }
}

impl Serialize for ToolDescArg {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ToolDescArg", 2)?;
        match self {
            ToolDescArg::String {
                description,
                r#enum,
            } => {
                state.serialize_field("type", "string")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
                if let Some(r#enum) = r#enum {
                    state.serialize_field("enum", r#enum)?;
                }
            }
            ToolDescArg::Number { description } => {
                state.serialize_field("type", "number")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
            }
            ToolDescArg::Boolean { description } => {
                state.serialize_field("type", "boolean")?;
                if let Some(description) = description {
                    state.serialize_field("description", description)?;
                }
            }
            ToolDescArg::Object {
                properties,
                required,
            } => {
                state.serialize_field("type", "object")?;
                state.serialize_field("properties", properties)?;
                state.serialize_field("required", required)?;
            }
            ToolDescArg::Array { items } => {
                state.serialize_field("type", "array")?;
                if let Some(items) = items {
                    state.serialize_field("items", items)?;
                }
            }
            ToolDescArg::Null {} => {
                state.serialize_field("type", "null")?;
            }
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for ToolDescArg {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Raw {
            #[serde(rename = "type")]
            r#type: String,
            description: Option<String>,
            properties: Option<IndexMap<String, Box<ToolDescArg>>>,
            required: Option<Vec<String>>,
            r#enum: Option<Vec<String>>,
            items: Option<Box<ToolDescArg>>,
        }

        let raw = Raw::deserialize(deserializer)?;

        match raw.r#type.as_str() {
            "string" => Ok(ToolDescArg::String {
                description: raw.description,
                r#enum: raw.r#enum,
            }),
            "number" => Ok(ToolDescArg::Number {
                description: raw.description,
            }),
            "boolean" => Ok(ToolDescArg::Boolean {
                description: raw.description,
            }),
            "object" => Ok(ToolDescArg::Object {
                properties: raw.properties.unwrap_or_default(),
                required: raw.required.unwrap_or_default(),
            }),
            "array" => Ok(ToolDescArg::Array { items: raw.items }),
            "null" => Ok(ToolDescArg::Null {}),
            other => Err(de::Error::custom(format!(r#"unsupported "type": {other}"#))),
        }
    }
}

/// Describes a single callable tool for an LLM: its human-readable name/description,
/// the JSON-shaped parameter schema, and the JSON-shaped return schema. Tool
/// descriptions are typically embedded in a chat template’s system/header section
/// so the model knows *what it can call* and *how to format arguments/returns*.
///
/// This struct pairs with `ToolDescArg`, a reduced, recursive
/// JSON-Schema–style node used to express `parameters` and `return`.
///
/// # Fields
/// - `name`
///   Human-friendly identifier the model can reference when calling the tool.
/// - `description`
///   Short, imperative explanation of what the tool does and when to use it.
/// - `parameters`
///   Schema describing the request payload. In practice this is **usually**
///   `ToolDescArg::Object` with `properties` (argument names) and
///   `required` (argument names that must be present).
/// - `return`
///   Schema describing the tool’s JSON response. Can be any `ToolDescArg`,
///   commonly an `Object`.
///
/// # Serialization format
/// This type serializes/deserializes with a fixed outer wrapper so that the JSON
/// matches common chat-template expectations (e.g., Hugging Face
/// `apply_chat_template`):
///
/// ```json
/// {
///   "type": "function",
///   "function": {
///     "name": "tool_name",
///     "description": "What the tool does",
///     "parameters": { /* ToolDescArg */ },
///     "return": { /* ToolDescArg */ }
///   }
/// }
/// ```
///
/// Notes:
/// - The Rust field `r#return` is emitted/parsed as the JSON key `"return"`.
/// - Ordering of `Object.properties` is preserved via `IndexMap`, which can help
///   readability and (for some models) prompt stability.
///
/// # Conventions & tips
/// - Keep `name` concise and stable; avoid spaces that could be misparsed.
/// - Write `description` as guidance for the model (“Use this to … when …”).
/// - Prefer explicit, narrow schemas. For example, use `String { enum: [...] }`
///   to constrain literals; mark truly required args in `required`.
/// - For `parameters`, use `Object` with well-named properties; avoid deeply
///   nested or overly generic shapes unless necessary.
/// - For `return`, document the *machine-readable* JSON shape your tool produces.
///   (LLMs tend to follow the declared schema more reliably than prose.)
///
/// # Example
/// Building a tool that fetches weather by city and returns a number:
///
/// ```rust
/// use indexmap::IndexMap;
///
/// let params = ToolDescArg::new_object().with_properties(
///     [("location", ToolDescArg::new_string()
///         .with_desc("The city name"))],
///     ["location"],
/// );
///
/// let ret = ToolDescArg::new_number()
///     .with_desc("Current temperature in Celsius");
///
/// let tool = ToolDesc {
///     name: "weather".into(),
///     description: "Get the current temperature for a city".into(),
///     parameters: params,
///     r#return: ret,
/// };
///
/// let json = serde_json::to_string_pretty(&tool).unwrap();
/// // Yields:
/// // {
/// //   "type": "function",
/// //   "function": {
/// //     "name": "weather",
/// //     "description": "Get the current temperature for a city",
/// //     "parameters": {
/// //       "type": "object",
/// //       "properties": {
/// //         "location": { "type": "string", "description": "The city name" }
/// //       },
/// //       "required": ["location"]
/// //     },
/// //     "return": { "type": "number", "description": "Current temperature in Celsius" }
/// //   }
/// // }
/// ```
///
/// # Compatibility
/// The emitted JSON shape matches the commonly used “function tool” format and
/// is suitable for chat templates that accept a list of tools with schemas (e.g.,
/// Hugging Face Transformers’ `apply_chat_template`). The schema nodes themselves
/// intentionally support a reduced subset of JSON Schema for predictability.
///
/// For more background on the expected schema shape, see:
/// https://huggingface.co/docs/transformers/v4.53.3/en/chat_extras#schema
///
/// See also: [`ToolDescArg`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDesc {
    name: String,

    description: String,

    parameters: ToolDescArg, // In most cases, it's `ToolDescArg::Object`

    r#return: Option<ToolDescArg>,
}

impl ToolDesc {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: impl Into<ToolDescArg>,
        r#return: Option<impl Into<ToolDescArg>>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: parameters.into(),
            r#return: r#return.and_then(|v| Some(v.into())),
        }
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }
}

impl Serialize for ToolDesc {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Borrowing inner view to control field order inside "function"
        #[derive(Serialize)]
        struct Inner<'a> {
            name: &'a str,
            description: &'a str,
            parameters: &'a ToolDescArg,
            #[serde(rename = "return")]
            r#return: Option<&'a ToolDescArg>,
        }

        let mut st = serializer.serialize_struct("ToolDesc", 2)?;
        st.serialize_field("type", "function")?;
        st.serialize_field(
            "function",
            &Inner {
                name: &self.name,
                description: &self.description,
                parameters: &self.parameters,
                r#return: self.r#return.as_ref(),
            },
        )?;
        st.end()
    }
}

impl<'de> Deserialize<'de> for ToolDesc {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct InnerOwned {
            name: String,
            description: String,
            parameters: ToolDescArg,
            #[serde(rename = "return")]
            r#return: Option<ToolDescArg>,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wrapper {
            #[serde(rename = "type")]
            r#type: String,
            function: InnerOwned,
        }

        let w = Wrapper::deserialize(deserializer)?;

        if w.r#type != "function" {
            return Err(de::Error::custom(format!(
                r#"expected `"type":"function"`, got "{}""#,
                w.r#type
            )));
        }

        Ok(ToolDesc {
            name: w.function.name,
            description: w.function.description,
            parameters: w.function.parameters,
            r#return: w.function.r#return,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn simple_tool_description_serde() {
        let original = ToolDesc::new(
            "temperature",
            "Get current temperature",
            ToolDescArg::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescArg::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescArg::new_string()
                            .with_desc("Default: Celcius")
                            .with_enum(["Celcius", "Fernheit"]),
                    ),
                ],
                ["location"],
            ),
            Some(ToolDescArg::new_number()),
        );
        let serialized = {
            let expected = r#"{"type":"function","function":{"name":"temperature","description":"Get current temperature","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city name"},"unit":{"type":"string","description":"Default: Celcius","enum":["Celcius","Fernheit"]}},"required":["location"]},"return":{"type":"number"}}}"#;
            let actual = serde_json::to_string(&original).unwrap();
            assert_eq!(expected, actual);
            actual
        };
        let recovered: ToolDesc = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, recovered);
    }
}
