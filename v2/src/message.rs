use std::{collections::HashMap, fmt};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap as _,
};
use serde_json::Value;
use url::Url;

/// Represents a unit of message content.
///
/// A `Part` can be one of several types, such as plain text, JSON data, or an image.
#[derive(Clone, Debug)]
pub enum Part {
    Text(String),
    Json(Value),
    ImageURL(Url),
    ImageBase64(String),
}

impl Part {
    /// Constructor for text part
    pub fn from_text<T: Into<String>>(text: T) -> Part {
        Part::Text(text.into())
    }

    pub fn is_text(&self) -> bool {
        match self {
            Part::Text(_) => true,
            _ => false,
        }
    }

    pub fn get_text(&self) -> Option<&str> {
        match self {
            Part::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Constructor for JSON part
    pub fn from_json<T: Into<Value>>(json: T) -> Part {
        Part::Json(json.into())
    }

    pub fn is_json(&self) -> bool {
        match self {
            Part::Json(_) => true,
            _ => false,
        }
    }

    pub fn get_json(&self) -> Option<&Value> {
        match self {
            Part::Json(v) => Some(v),
            _ => None,
        }
    }

    /// Constructor for image URL part
    pub fn from_image_url<T: Into<Url>>(url: T) -> Part {
        Part::ImageURL(url.into())
    }

    /// Constructor for image base64 part
    pub fn from_image_base64<T: Into<String>>(encoded: T) -> Part {
        Part::ImageBase64(encoded.into())
    }
}

// Serialization logic for Part
impl Serialize for Part {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        match self {
            Part::Text(text) => {
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
            }
            Part::Json(value) => {
                map.serialize_entry("type", "json")?;
                map.serialize_entry("json", value)?;
            }
            Part::ImageURL(url) => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("url", url.as_str())?;
            }
            Part::ImageBase64(encoded) => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("base64", encoded)?;
            }
        };
        map.end()
    }
}

// Deserialization logic for Part
impl<'de> Deserialize<'de> for Part {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(PartVisitor)
    }
}

struct PartVisitor;

impl<'de> Visitor<'de> for PartVisitor {
    type Value = Part;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut ty: Option<String> = None;
        let mut key: Option<String> = None;
        let mut value: Option<String> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else {
                if key.is_some() {
                    return Err(de::Error::custom("multiple part fields found"));
                }
                key = Some(k);
                value = Some(map.next_value()?);
            }
        }

        let ty = ty.ok_or_else(|| de::Error::missing_field("type"))?;
        let key = key.ok_or_else(|| de::Error::custom("missing part key"))?;
        let value = value.ok_or_else(|| de::Error::custom("missing part value"))?;

        if ty == "text" && key == "text" {
            Ok(Part::Text(value))
        } else if ty == "json" && key == "json" {
            match serde_json::from_str(&value) {
                Ok(value) => Ok(Part::Json(value)),
                Err(err) => Err(de::Error::custom(format!(
                    "Invalid Json part: {} {}",
                    value,
                    err.to_string(),
                ))),
            }
        } else if ty == "image" && key == "url" {
            match url::Url::parse(&value) {
                Ok(value) => Ok(Part::ImageURL(value)),
                Err(err) => Err(de::Error::custom(format!(
                    "Invalid URL: {} {}",
                    value,
                    err.to_string()
                ))),
            }
        } else if ty == "image" && key == "base64" {
            Ok(Part::ImageBase64(value))
        } else {
            Err(de::Error::custom("Invalid type"))
        }
    }
}

/// A data structure to represent which part of the message is being updated.
#[derive(Clone, Debug)]
pub enum MessageDelta {
    Content(Part),
    Reasoning(Part),
    ToolCall(Part),
}

impl MessageDelta {
    pub fn content(part: Part) -> MessageDelta {
        MessageDelta::Content(part)
    }

    pub fn reasoning(part: Part) -> MessageDelta {
        MessageDelta::Reasoning(part)
    }

    pub fn tool_call(part: Part) -> MessageDelta {
        MessageDelta::ToolCall(part)
    }

    pub fn get_part(&self) -> &Part {
        match self {
            MessageDelta::Content(part) => part,
            MessageDelta::Reasoning(part) => part,
            MessageDelta::ToolCall(part) => part,
        }
    }

    pub fn take_part(self) -> Part {
        match self {
            MessageDelta::Content(part) => part,
            MessageDelta::Reasoning(part) => part,
            MessageDelta::ToolCall(part) => part,
        }
    }
}

/// Role of the message sender
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Represents a full message with multiple sections.
///
/// Its serialization and deserialization logic is compatible with HuggingFace-style messages (also known as OpenAI-compatible format).
/// The only exception is that, since this structure supports multi-modal content, the `Part` logic may differ from text-only implementations.
#[derive(Clone, Debug)]
pub struct Message {
    role: Role,
    content: Vec<Part>,
    reasoning: Vec<Part>,
    tool_calls: Vec<Part>,
}

impl Message {
    /// Create an empty message
    pub fn new(role: Role) -> Message {
        Message {
            role,
            content: Vec::new(),
            reasoning: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a message with initial content
    pub fn with_content(role: Role, content: Part) -> Message {
        Message {
            role,
            content: vec![content],
            reasoning: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Append content to the content vector
    pub fn push_content(&mut self, content: Part) -> () {
        self.content.push(content);
    }

    /// Append reasoning part
    pub fn push_reasoning(&mut self, reasoning: Part) -> () {
        self.reasoning.push(reasoning);
    }

    /// Append tool call part
    pub fn push_tool_calls(&mut self, tool_calls: Part) -> () {
        self.tool_calls.push(tool_calls);
    }

    pub fn role(&self) -> &Role {
        &self.role
    }

    pub fn content(&self) -> &Vec<Part> {
        &self.content
    }

    pub fn reasoning(&self) -> &Vec<Part> {
        &self.reasoning
    }

    pub fn tool_calls(&self) -> &Vec<Part> {
        &self.tool_calls
    }
}

// Serialization logic for Message
impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        if !self.reasoning.is_empty() {
            map.serialize_entry("reasoning", &self.reasoning)?;
        }
        if !self.content.is_empty() {
            map.serialize_entry("content", &self.content)?;
        }
        if !self.tool_calls.is_empty() {
            map.serialize_entry("tool_calls", &self.tool_calls)?;
        }
        map.end()
    }
}

/// Deserialization logic for Message
impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> Visitor<'de> for MessageVisitor {
    type Value = Message;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Message, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut role: Option<Role> = None;
        let mut content: Vec<Part> = Vec::new();
        let mut reasoning: Vec<Part> = Vec::new();
        let mut tool_calls: Vec<Part> = Vec::new();

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else if k == "content" {
                content = map.next_value()?;
            } else if k == "reasoning" {
                reasoning = map.next_value()?;
            } else if k == "tool_calls" {
                tool_calls = map.next_value()?;
            } else {
                return Err(de::Error::unknown_field(
                    &k,
                    &["content", "reasoning", "tool_calls"],
                ));
            }
        }
        let role = role.ok_or_else(|| de::Error::missing_field("role"))?;

        Ok(Message {
            role,
            content,
            reasoning,
            tool_calls,
        })
    }
}

#[derive(Debug)]
pub struct MessageAggregator {
    last_delta: Option<MessageDelta>,
    aggregated: Message,
}

impl MessageAggregator {
    pub fn new(role: Role) -> Self {
        MessageAggregator {
            last_delta: None,
            aggregated: Message::new(role),
        }
    }

    fn finalize_last_delta(&mut self) {
        if let Some(last_delta) = self.last_delta.take() {
            match last_delta {
                MessageDelta::Content(part) => {
                    self.aggregated.push_content(part);
                }
                MessageDelta::Reasoning(part) => {
                    self.aggregated.push_reasoning(part);
                }
                MessageDelta::ToolCall(part) => {
                    self.aggregated.push_tool_calls(part);
                }
            }
        };
    }

    pub fn update(&mut self, delta: MessageDelta) {
        match (&mut self.last_delta, delta) {
            (
                Some(MessageDelta::Content(Part::Text(last_text))),
                MessageDelta::Content(Part::Text(text)),
            ) => {
                // Text concat
                last_text.push_str(&text);
            }
            (
                Some(MessageDelta::Reasoning(Part::Text(last_text))),
                MessageDelta::Reasoning(Part::Text(text)),
            ) => {
                // Text concat
                last_text.push_str(&text);
            }
            (None, delta) => {
                self.last_delta = Some(delta);
            }
            (Some(_), delta) => {
                // Current last_delta -> registred into `self.aggregated`
                self.finalize_last_delta();
                // Current delta -> last delta
                self.last_delta = Some(delta);
            }
        };
    }

    pub fn finalize(mut self) -> Message {
        self.finalize_last_delta();
        self.aggregated
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolDescriptionPropertyType {
    String,
    Number,
    Boolean,
    Object,
    Array,
    Null,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDescriptionProperty {
    r#type: ToolDescriptionPropertyType,
    description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDescriptionParameters {
    r#type: String, // Fixed to ["object"]
    properties: HashMap<String, ToolDescriptionProperty>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDescription {
    name: String,
    description: String,
    parameters: ToolDescriptionParameters,
    required: Vec<String>,
    r#return: ToolDescriptionProperty,
}

impl ToolDescription {
    pub fn new() -> Self {
        todo!()
    }
}
