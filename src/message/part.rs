use std::fmt;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::datatype::{Bytes, Value};

/// Represents a function call contained within a message part.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartFunction {
    /// The name of the function
    pub name: String,

    /// The arguments of the function, usually represented as a JSON object.
    pub arguments: Value,
}

/// Represents the image data contained in a [`Part`].
///
/// `PartImage` provides structured access to image data.
/// Currently, it only implments "binary" types.
///
/// # Example
/// ```rust
/// let part = Part::image_binary(640, 480, "rgb", (0..640*480*3).map(|i| (i % 255) as u8)).unwrap();
///
/// if let Some(img) = part.as_image() {
///     assert_eq!(img.height(), 640);
///     assert_eq!(img.width(), 480);
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum PartImage {
    Embedded { mime_type: String, data: Bytes },
    Url { url: String },
}

/// Represents a semantically meaningful content unit exchanged between the model and the user.
///
/// Conceptually, each `Part` encapsulates a piece of **data** that contributes
/// to a chat message — such as text, a function invocation, or an image.  
///
/// For example, a single message consisting of a sequence like  
/// `(text..., image, text...)` is represented as a `Message` containing
/// an array of three `Part` elements.
///
/// Note that a `Part` does **not** carry "intent", such as "reasoning" or "tool call".
/// These higher-level semantics are determined by the context of a [`Message`].
///
/// # Example
///
/// ## Rust
/// ```rust
/// let part = Part::text("Hello, world!");
/// assert!(part.is_text());
/// ```
///
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Part {
    /// Plain utf-8 encoded text.
    Text { text: String },

    /// Represents a structured function call to an external tool.
    ///
    /// Many language models (LLMs) use a **function calling** mechanism to extend their capabilities.
    /// When an LLM decides to use external *tools*, it produces a structured output called a `function`.
    /// A function conventionally consists of two fields: a `name`, and an `arguments` field formatted as JSON.
    /// This is conceptually similar to making an HTTP POST request, where the request body carries a single JSON object.
    ///
    /// This struct models that convention, representing a function invocation request
    /// from an LLM to an external tool or API.
    ///
    /// # Examples
    /// ```rust
    /// let f = PartFunction {
    ///     name: "translate".to_string(),
    ///     arguments: Value::from_json(r#"{"source": "hello", "lang": "cn"}"#).unwrap(),
    /// };
    /// ```
    Function { id: String, function: PartFunction },

    /// Holds a structured data value, typically considered as a JSON structure.
    Value { value: Value },

    /// Contains an image payload or reference used within a message part.
    /// The image may be provided as raw binary data or an encoded format (e.g., PNG, JPEG),
    /// or as a reference via a URL. Optional metadata can be included alongside the image.
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
        match self {
            Self::Text { .. } => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::Function { .. } => true,
            _ => false,
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            Self::Value { .. } => true,
            _ => false,
        }
    }

    pub fn is_image(&self) -> bool {
        match self {
            Self::Image { .. } => true,
            _ => false,
        }
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
