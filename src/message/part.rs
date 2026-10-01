use std::fmt;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::datatype::{Bytes, Value};

/// Represents a function call contained within a message part.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PartFunction {
    /// The name of the function
    pub name: String,

    /// The arguments of the function, usually represented as a JSON object.
    pub arguments: Value,
}

/// Image data in a [`Part`]: embedded bytes or a URL.
///
/// # Example
/// ```rust
/// # use ailoy::message::Part;
/// let part = Part::image_url("https://example.com/image.png".to_string()).unwrap();
/// assert!(part.is_image());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PartImage {
    Embedded { mime_type: String, data: Bytes },
    Url { url: String },
}

/// One content unit of a message (text, function call, value, image); a
/// `(text, image, text)` message is three parts. A part carries no intent such
/// as reasoning or tool call; that comes from its place in the
/// [`Message`](crate::message::Message).
///
/// # Example
///
/// ## Rust
/// ```rust
/// # use ailoy::message::Part;
/// let part = Part::text("Hello, world!");
/// assert!(part.is_text());
/// ```
///
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Part {
    /// Plain utf-8 encoded text.
    Text { text: String },

    /// A tool call requested by the model; `id` pairs it with its result.
    Function { id: String, function: PartFunction },

    /// Holds a structured data value, typically considered as a JSON structure.
    Value { value: Value },

    /// An image, embedded or by URL.
    Image { image: PartImage },
}

impl Part {
    pub fn text(v: impl Into<String>) -> Self {
        Self::Text { text: v.into() }
    }

    pub fn function(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<Value>,
    ) -> Self {
        Self::Function {
            id: id.into(),
            function: PartFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }

    pub fn image_embedded(mime_type: impl Into<String>, data: Bytes) -> anyhow::Result<Self> {
        Ok(Part::Image {
            image: PartImage::Embedded {
                mime_type: mime_type.into(),
                data,
            },
        })
    }

    pub fn image_url(url: String) -> anyhow::Result<Self> {
        let url = Url::parse(&url).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Part::Image {
            image: PartImage::Url { url: url.into() },
        })
    }

    pub fn value(value: impl Into<Value>) -> Self {
        Part::Value {
            value: value.into(),
        }
    }

    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Self::Function { .. })
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value { .. })
    }

    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image { .. })
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(&str, &str, &Value)> {
        match self {
            Self::Function {
                id,
                function: PartFunction { name, arguments },
            } => Some((id.as_str(), name.as_str(), arguments)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(&mut String, &mut String, &mut Value)> {
        match self {
            Self::Function {
                id,
                function: PartFunction { name, arguments },
            } => Some((id, name, arguments)),
            _ => None,
        }
    }

    pub fn as_value(&self) -> Option<&Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }

    pub fn as_value_mut(&mut self) -> Option<&mut Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "Part {}", s)
    }
}
