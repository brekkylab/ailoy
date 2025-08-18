use std::fmt::{self, Display};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap,
};
use serde_json::json;
use strum::{Display, EnumString};
use url::Url;

/// Represents a single, typed unit of message content (multi-modal).
///
/// A `Part` models one element inside a message’s `content` (and related fields like
/// `reasoning` / `tool_calls`). It is designed to be **OpenAI-compatible** when serialized,
/// using the modern “array-of-parts” shape such as:
/// `{ "type": "text", "text": "..." }` or `{ "type": "image", "url": "..." }`.
///
/// # Variants
/// - [`Part::Text`]: Plain UTF-8 text.
/// - [`Part::Function`]: Raw JSON string for a tool/function call payload, with an optional
///   `tool_call_id` to correlate results.
/// - [`Part::ImageURL`]: A web-accessible image (HTTP(S) URL).
/// - [`Part::ImageBase64`]: An inline, base64-encoded image payload.
///
/// # OpenAI-compatible mapping (typical)
/// These are the common wire shapes produced/consumed when targeting OpenAI-style APIs:
///
/// - `Text(s)`
///   ```json
///   { "type": "text", "text": "hello" }
///   ```
///
/// - `ImageURL(url)`
///   ```json
///   { "type": "image", "url": "https://example.com/cat.png" }
///   ```
///
/// - `ImageBase64(data)`
///   ```json
///   { "type": "image", "data": "<base64-bytes>" }
///   ```
///
/// - `Function { id, function }`
///   The `function` field holds the **raw JSON** for the tool/function call (often the
///   `arguments` string in OpenAI tool calls). The `id` corresponds to `tool_call_id`
///   and is used to link the tool’s eventual result back to this call. Serialization of
///   this variant is handled by higher-level message (de)serializers that place it under
///   `tool_calls` as appropriate.
///
/// # Notes on `Function`
/// - **As-is storage:** While streaming, `function` may be incomplete or invalid JSON.
///   This is expected. Parse only after the stream has finalized/aggregated.
/// - **No validation:** The variant does not validate or mutate the JSON string.
///   Converting into a strongly-typed [`crate::value::ToolCall`] will fail if the JSON
///   is malformed or incomplete.
/// - **Correlation:** When present, `id` should be echoed back as `tool_call_id` when
///   returning the tool’s output, matching OpenAI’s linking behavior.
///
/// # Invariants & behavior
/// - Insertion order is preserved by the container (e.g., `Message.content`).
/// - Image parts do not fetch or validate URLs/bytes at construction time.
/// - The enum is transport-agnostic; OpenAI compatibility is achieved by the
///   surrounding (de)serializer layer.
///
/// # Examples
/// Building a multi-modal user message:
/// ```rust
/// # use url::Url;
/// # use crate::value::{Message, Part, Role};
/// let mut msg = Message::new(Role::User);
/// msg.push_content(Part::Text("What does this sign say?".into()));
/// msg.push_content(Part::ImageURL(Url::parse("https://example.com/sign.jpg").unwrap()));
/// ```
///
/// Collecting tool-call arguments during streaming:
/// ```rust
/// # use crate::value::Part;
/// let mut buf = String::new();
/// // Append chunks as they arrive:
/// buf.push_str("{\"location\":\"Dubai\"");
/// buf.push_str(",\"unit\":\"Celsius\"}");
/// let part = Part::Function { id: None, function: buf }; // parse after finalization
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    /// Plain UTF-8 text.
    Text(String),

    /// Tool/function call payload captured as a raw JSON string.
    ///
    /// `id` corresponds to `tool_call_id` (when available) for correlating tool results.
    /// `function` holds the unmodified JSON; it may be incomplete while streaming.
    Function {
        /// Optional `tool_call_id` used to correlate tool results.
        id: Option<String>,
        /// Raw JSON for the function call payload (often the `arguments` string).
        function: String,
    },

    /// Web-addressable image (HTTP(S) URL).
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "url": "<...>" }`.
    ImageURL(Url),

    /// Inline, base64-encoded image bitmap bytes.
    ///
    /// Typically serialized as:
    /// `{ "type": "image", "data": "<base64>" }`.
    ImageData(String),

    /// Base64-encoded audio bytes.
    ///
    /// Typically serialized as:
    /// `{ "type": "audio", "data": "<base64>", "format": "<format>" }`.
    Audio {
        /// Base64-encoded string
        data: String,
        /// "mp3" or "wav"
        format: String,
    },
}

impl Part {
    /// Constructor for text part
    pub fn new_text<T: Into<String>>(text: T) -> Part {
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

    pub fn get_text_owned(&self) -> Option<String> {
        match self {
            Part::Text(v) => Some(v.to_owned()),
            _ => None,
        }
    }

    /// Constructor for Function part
    pub fn new_function<T: Into<String>>(json: T) -> Part {
        Part::Function {
            id: None,
            function: json.into(),
        }
    }

    pub fn new_function_with_id<T: Into<String>>(id: impl Into<String>, json: T) -> Part {
        Part::Function {
            id: Some(id.into()),
            function: json.into(),
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Part::Function { .. } => true,
            _ => false,
        }
    }

    pub fn get_function_id(&self) -> Option<String> {
        match self {
            Part::Function { id, .. } => id.to_owned(),
            _ => None,
        }
    }

    pub fn get_function(&self) -> Option<&String> {
        match self {
            Part::Function { function, .. } => Some(function),
            _ => None,
        }
    }

    pub fn get_function_owned(&self) -> Option<String> {
        match self {
            Part::Function { function, .. } => Some(function.to_owned()),
            _ => None,
        }
    }

    /// Constructor for image URL part
    pub fn new_image_url<T: Into<Url>>(url: T) -> Part {
        Part::ImageURL(url.into())
    }

    /// Constructor for image base64 part
    pub fn new_image_data<T: Into<String>>(encoded: T) -> Part {
        Part::ImageData(encoded.into())
    }

    /// Constructor for audio base64 part
    pub fn new_audio_data<T: Into<String>>(encoded: T, format: String) -> Part {
        Part::Audio {
            data: encoded.into(),
            format,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Part::Text(v) => v.is_empty(),
            Part::Function { function: v, .. } => v.is_empty(),
            Part::ImageURL(_) => false,
            Part::ImageData(v) => v.is_empty(),
            Part::Audio { data: v, .. } => v.is_empty(),
        }
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
            Part::Function { id, function } => {
                // Try to embed as real JSON if valid; otherwise fall back to raw text.
                let function: serde_json::Value = serde_json::from_str(function).map_err(|e| {
                    <S::Error as serde::ser::Error>::custom(format!(
                        "invalid JSON in Part::Json: {e}"
                    ))
                })?;
                map.serialize_entry("type", "function")?;
                if id.is_some() {
                    map.serialize_entry("id", &id.to_owned().unwrap())?;
                }
                map.serialize_entry("function", &function)?;
            }
            Part::ImageURL(url) => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("url", url.as_str())?;
            }
            Part::ImageData(encoded) => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("data", encoded)?;
            }
            Part::Audio { data, format } => {
                map.serialize_entry("type", "audio")?;
                map.serialize_entry(
                    "audio",
                    &json!({
                        "data": data,
                        "format": format,
                    }),
                )?;
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
        let mut value: Option<serde_json::Value> = None;
        let mut id: Option<String> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "type" {
                if ty.is_some() {
                    return Err(de::Error::duplicate_field("type"));
                }
                ty = Some(map.next_value()?);
            } else if k == "id" {
                if id.is_some() {
                    return Err(de::Error::duplicate_field("id"));
                }
                id = Some(map.next_value()?);
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
            Ok(Part::Text(value.to_string()))
        } else if ty == "function" && key == "function" {
            match serde_json::from_str(&value.to_string()) {
                Ok(value) => Ok(Part::Function {
                    id,
                    function: value,
                }),
                Err(err) => Err(de::Error::custom(format!(
                    "Invalid Json part: {} {}",
                    value,
                    err.to_string(),
                ))),
            }
        } else if ty == "image" && key == "url" {
            match url::Url::parse(&value.to_string()) {
                Ok(value) => Ok(Part::ImageURL(value)),
                Err(err) => Err(de::Error::custom(format!(
                    "Invalid URL: {} {}",
                    value,
                    err.to_string()
                ))),
            }
        } else if ty == "image" && key == "data" {
            Ok(Part::ImageData(value.to_string()))
        } else if ty == "audio" && key == "audio" {
            Ok(Part::Audio {
                data: value["data"].to_string(),
                format: value["format"].to_string(),
            })
        } else {
            Err(de::Error::custom("Invalid type"))
        }
    }
}

impl Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Part::Text(text) => f.write_str(&format!("Part(type=text, text=\"{}\")", text)),
            Part::Function { id, function } => match id {
                Some(id) => f.write_str(&format!(
                    "Part(type=function, id=\"{}\", function=\"{}\")",
                    id, function
                )),
                None => f.write_str(&format!("Part(type=function, function=\"{}\")", function)),
            },
            Part::ImageURL(url) => {
                f.write_str(&format!("Part(type=image, url=\"{}\")", url.to_string()))
            }
            Part::ImageData(data) => {
                f.write_str(&format!("Part(type=image, data=({} bytes))", data.len()))
            }
            Part::Audio { data, format } => f.write_str(&format!(
                "Part(type=audio, foramt={}, data=({} bytes))",
                format,
                data.len()
            )),
        }
    }
}

/// A single streaming update to a message.
///
/// `MessageDelta` represents one incremental piece emitted while a model is
/// generating a response (or while a tool is streaming its output). Each delta
/// identifies **which field** of the message is being extended and the **role**
/// responsible for that piece, along with the concrete [`Part`] being added.
///
/// This mirrors how many chat APIs stream “chunks” of content. You typically
/// feed these deltas into an aggregator (e.g., `MessageAggregator`) to build
/// a complete [`Message`] incrementally.
///
/// # Variants
/// - [`MessageDelta::Content`]: Appends a [`Part`] to the message’s `content`.
/// - [`MessageDelta::Reasoning`]: Appends a [`Part`] to the (optional) `reasoning`
///   field. This is usually **not** user-visible and is intended for internal
///   traces when supported by the model / API.
/// - [`MessageDelta::ToolCall`]: Appends a [`Part`] (often a `Function` part)
///   into the message’s tool-calls area, used to accumulate tool call arguments
///   during streaming.
///
/// # Invariants & behavior
/// - Deltas do not reorder data; your aggregator should preserve arrival order.
/// - The `role` carried by each delta indicates the author of the appended part
///   (commonly [`Role::Assistant`] for model output, [`Role::Tool`] for tool output,
///   and [`Role::User`] for user-streamed uploads).
///
/// # Examples
/// Stream assistant text and a tool call:
/// ```rust
/// # use crate::value::{MessageDelta, Part, Role};
/// let d1 = MessageDelta::new_assistant_content(Part::Text("Looking up weather…".into()));
/// let d2 = MessageDelta::new_assistant_tool_call(Part::Function {
///     id: Some("call_1".into()),
///     function: r#"{"name":"get_weather","arguments":{"city":"Seoul"}}"#.into(),
/// });
/// // Feed `d1`, `d2`, ... into your aggregator.
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MessageDelta {
    /// Append a [`Part`] to the message's `content` array for the given `role`.
    Content(Role, Part),

    /// Append a [`Part`] to the message's `reasoning` array for the given `role`.
    ///
    /// This is typically internal/system-facing metadata (if supported by the API)
    /// and not shown to end users.
    Reasoning(Role, Part),

    /// Append a [`Part`] to the message's `tool_calls` area for the given `role`.
    ///
    /// Commonly used while streaming tool/function call arguments from the assistant.
    ToolCall(Role, Part),
}

impl MessageDelta {
    /// Convenience constructor for a user-authored content delta.
    pub fn new_user_content(part: Part) -> MessageDelta {
        MessageDelta::Content(Role::User, part)
    }

    /// Convenience constructor for an assistant-authored content delta.
    pub fn new_assistant_content(part: Part) -> MessageDelta {
        MessageDelta::Content(Role::Assistant, part)
    }

    /// Convenience constructor for an assistant-authored reasoning delta.
    pub fn new_assistant_reasoning(part: Part) -> MessageDelta {
        MessageDelta::Reasoning(Role::Assistant, part)
    }

    /// Convenience constructor for an assistant-authored tool-call delta.
    pub fn new_assistant_tool_call(part: Part) -> MessageDelta {
        MessageDelta::ToolCall(Role::Assistant, part)
    }

    /// Convenience constructor for a tool-authored content delta.
    pub fn new_tool_content(part: Part) -> MessageDelta {
        MessageDelta::Content(Role::Tool, part)
    }

    /// Returns the `role` that authored this delta.
    pub fn get_role(&self) -> &Role {
        match self {
            MessageDelta::Content(role, _) => role,
            MessageDelta::Reasoning(role, _) => role,
            MessageDelta::ToolCall(role, _) => role,
        }
    }

    /// Returns the [`Part`] contained in this delta.
    pub fn get_part(&self) -> &Part {
        match self {
            MessageDelta::Content(_, part) => part,
            MessageDelta::Reasoning(_, part) => part,
            MessageDelta::ToolCall(_, part) => part,
        }
    }

    /// Consumes the delta, yielding its `(Role, Part)` pair.
    pub fn take(self) -> (Role, Part) {
        match self {
            MessageDelta::Content(role, part) => (role, part),
            MessageDelta::Reasoning(role, part) => (role, part),
            MessageDelta::ToolCall(role, part) => (role, part),
        }
    }
}

impl Serialize for MessageDelta {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(2))?;
        match self {
            MessageDelta::Content(role, part) => {
                map.serialize_entry("role", role)?;
                map.serialize_entry("content", part)?;
            }
            MessageDelta::Reasoning(role, part) => {
                map.serialize_entry("role", role)?;
                map.serialize_entry("reasoning", part)?;
            }
            MessageDelta::ToolCall(role, part) => {
                map.serialize_entry("role", role)?;
                map.serialize_entry("tool_calls", part)?;
            }
        };
        map.end()
    }
}

// Deserialization logic for Part
impl<'de> Deserialize<'de> for MessageDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageDeltaVisitor)
    }
}

struct MessageDeltaVisitor;

impl<'de> Visitor<'de> for MessageDeltaVisitor {
    type Value = MessageDelta;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut role: Option<Role> = None;
        let mut content: Option<Part> = None;
        let mut reasoning: Option<Part> = None;
        let mut tool_call: Option<Part> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else if k == "content" {
                if content.is_some() {
                    return Err(de::Error::duplicate_field("content"));
                }
                content = Some(map.next_value()?);
            } else if k == "reasoning" {
                if reasoning.is_some() {
                    return Err(de::Error::duplicate_field("reasoning"));
                }
                reasoning = Some(map.next_value()?);
            } else if k == "tool_call" {
                if tool_call.is_some() {
                    return Err(de::Error::duplicate_field("tool_call"));
                }
                tool_call = Some(map.next_value()?);
            } else {
                return Err(de::Error::unknown_field(
                    &k,
                    &["content", "reasoning", "tool_call"],
                ));
            }
        }

        let role = role.ok_or_else(|| de::Error::missing_field("role"))?;
        if content.is_some() {
            Ok(MessageDelta::Content(role, content.unwrap()))
        } else if reasoning.is_some() {
            Ok(MessageDelta::Reasoning(role, reasoning.unwrap()))
        } else if tool_call.is_some() {
            Ok(MessageDelta::ToolCall(role, tool_call.unwrap()))
        } else {
            Err(de::Error::missing_field("content"))
        }
    }
}

impl Display for MessageDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageDelta::Content(role, part) => {
                f.write_str(&format!("MessageDelta(role={}, content={})", role, part))?;
            }
            MessageDelta::Reasoning(role, part) => {
                f.write_str(&format!("MessageDelta(role={}, reasoning={})", role, part))?;
            }
            MessageDelta::ToolCall(role, part) => {
                f.write_str(&format!("MessageDelta(role={}, tool_call={})", role, part))?;
            }
        }
        Ok(())
    }
}

/// The author of a message (or streaming delta) in a chat.
///
/// This aligns with common chat schemas (e.g., OpenAI-style), and is serialized
/// in lowercase (`"system"`, `"user"`, `"assistant"`, `"tool"`).
///
/// # Variants
/// - [`Role::System`]: System or policy instructions that guide the assistant.
/// - [`Role::User`]: End-user inputs and uploads.
/// - [`Role::Assistant`]: Model-generated outputs, including tool-call requests.
/// - [`Role::Tool`]: Outputs produced by external tools/functions, typically in
///   response to an assistant tool call (and often correlated via `tool_call_id`).
#[derive(Clone, Debug, Display, Serialize, Deserialize, PartialEq, Eq, EnumString)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "snake_case")]
pub enum Role {
    /// System instructions and constraints provided to the assistant.
    System,
    /// Content authored by the end user.
    User,
    /// Content authored by the assistant/model.
    Assistant,
    /// Content authored by a tool/function, usually as a result of a tool call.
    Tool,
}

/// Represents a complete chat message composed of multiple parts (multi-modal).
///
/// # OpenAI-compatible shape
/// Serialization/deserialization follows the OpenAI chat/response message conventions:
/// - `role` matches OpenAI roles (e.g., `"system"`, `"user"`, `"assistant"`, `"tool"`).
/// - `content` is an array of typed parts (e.g., `{ "type": "text", "text": "..." }`,
///   images, etc.), rather than a single string. This aligns with the modern “array-of-parts”
///   format used by OpenAI’s Chat/Responses APIs.
/// - `tool_calls` (when present) is compatible with OpenAI function/tool calling:
///   each element mirrors OpenAI’s `tool_calls[]` item. In this implementation
///   each `tool_calls` entry is stored as a `Part` (commonly `Part::Json`) containing the
///   raw JSON payload (`{"type":"function","id":"...","function":{"name":"...","arguments":"..."}}`).
/// - `reasoning` is optional and is used to carry model reasoning parts when available
///   from reasoning-capable models. Treat it as **auxiliary** content; if your target API
///   does not accept a `reasoning` field, omit/strip it before sending.
///
/// Because this type supports multi-modal content, exact `Part` variants (text, image, JSON, …)
/// may differ from text-only implementations, while staying wire-compatible with OpenAI.
///
/// # Fields
/// - `role`: Author of the message.
/// - `content`: Primary, user-visible content as a list of parts (text, images, etc.).
/// - `reasoning`: Optional reasoning parts (usually text). Not intended for end users.
/// - `tool_calls`: Tool/function call requests emitted by the assistant, each stored as a part
///   (typically a JSON part containing an OpenAI-shaped `tool_calls[]` item).
///
/// # Invariants & recommendations
/// - Order is preserved: parts appear in the order they were appended.
/// - `reasoning` and `tool_calls` should be present only on assistant messages.
/// - If you target strict OpenAI endpoints that don’t accept `reasoning`, drop that field
///   during serialization.
/// - Parsing/validation of `tool_calls` JSON is the caller’s responsibility.
///
/// # Examples
///
/// ## User message with text + image
/// ```json
/// {
///   "role": "user",
///   "content": [
///     { "type": "text", "text": "What does this sign say?" },
///     { "type": "input_image", "image_url": { "url": "https://example.com/sign.jpg" } }
///   ]
/// }
/// ```
///
/// ## Assistant message requesting a tool call
/// ```json
/// {
///   "role": "assistant",
///   "content": [ { "type": "text", "text": "I'll look that up." } ],
///   "tool_calls": [
///     {
///       "type": "function",
///       "id": "call_abc123",
///       "function": {
///         "name": "foo",
///         "arguments": "{\"location\":\"Dubai\",\"unit\":\"Celsius\"}"
///       }
///     }
///   ]
/// }
/// ```
///
/// ## Assistant message with optional reasoning parts
/// ```json
/// {
///   "role": "assistant",
///   "content": [ { "type": "text", "text": "The current temperature is 42 °C." } ],
///   "reasoning": [ { "type": "text", "text": "(model reasoning tokens, if exposed)" } ]
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
    pub reasoning: Vec<Part>,
    pub tool_calls: Vec<Part>,
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

impl Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("Message(role={}", self.role))?;
        if self.content.len() > 0 {
            let contents_str = self
                .content
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            f.write_str(&format!(", content=[{}]", &contents_str))?;
        }
        if self.reasoning.len() > 0 {
            let reasoning_str = self
                .reasoning
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            f.write_str(&format!(", reasoning=[{}]", &reasoning_str))?;
        }
        if self.tool_calls.len() > 0 {
            let tool_calls_str = self
                .tool_calls
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            f.write_str(&format!(",tool_calls=[{}]", &tool_calls_str))?;
        }
        f.write_str(")")?;
        Ok(())
    }
}

/// Incrementally assembles one or more [`Message`]s from a stream of [`MessageDelta`]s.
///
/// # What this does
/// `MessageAggregator` is a small state machine with a single-item buffer:
/// - It **coalesces adjacent chunks of the same kind** to avoid fragmentation
///   (e.g., text → text string-append; tool-call JSON → JSON string-append).
/// - It **flushes** the buffered chunk into the current [`Message`] whenever a new,
///   non-mergeable delta arrives.
/// - When the **role changes** (e.g., `assistant` → `tool`), it closes the current
///   message and returns it from [`update`], starting a new one for the new role.
///
/// This is handy for token-by-token or chunked streaming from LLM APIs, where many
/// tiny deltas arrive back-to-back.
///
/// # Merge rules (contiguous-only)
/// - `Content(Text) + Content(Text)` → append text
/// - `Reasoning(Text) + Reasoning(Text)` → append text
/// - `ToolCall(Function { id: _, function })`
///   + `ToolCall(Function { id: None, function })` → append `function`
///
/// Any other case is not merged; instead, the buffer is flushed to the current message.
///
/// # Lifecycle
/// 1. Construct with [`new`].
/// 2. Feed deltas in order with [`update`].  
///    - Returns `Some(Message)` **only** when a role boundary is crossed,
///      finalizing and yielding the previous role’s message.
///    - Returns `None` otherwise.
/// 3. Call [`finalize`] to flush and retrieve the last in-progress message (if any).
///
/// # Guarantees & Notes
/// - **Order-preserving**: incorporation order matches arrival order.
/// - **Zero parsing**: tool-call JSON is treated as raw text; parse post-aggregation.
/// - **Cheap steady-state**: merges use `String::push_str`.
/// - **Single-threaded**: not synchronized; use on one task/thread at a time.
/// - **Panics**: none expected.
///
/// # Example
/// ```rust
/// # use crate::value::{MessageAggregator, MessageDelta, Part, Role, Message};
/// let mut agg = MessageAggregator::new();
///
/// // assistant streams content
/// assert!(agg.update(MessageDelta::new_assistant_content(Part::Text("Hel".into()))).is_none());
/// assert!(agg.update(MessageDelta::new_assistant_content(Part::Text("lo".into()))).is_none());
///
/// // role switches to tool: prior assistant message is returned
/// let m1 = agg.update(MessageDelta::new_tool_content(Part::Text("ok".into()))).unwrap();
/// // ... use m1
///
/// // finalize remaining (tool) message
/// let m2 = agg.finalize().unwrap();
/// // ... use m2
/// ```
#[derive(Debug)]
pub struct MessageAggregator {
    /// Last unflushed delta; candidate for coalescing.
    last_delta: Option<MessageDelta>,
    /// The in-progress message for the current role.
    last_message: Option<Message>,
}

impl MessageAggregator {
    /// Creates a fresh aggregator with no buffered state.
    pub fn new() -> Self {
        MessageAggregator {
            last_delta: None,
            last_message: None,
        }
    }

    /// Flushes the buffered delta (if any and non-empty) into `last_message`,
    /// creating the message if it does not yet exist.
    fn flush_delta_into_message(&mut self) {
        // Drop empty buffered delta, if present.
        let should_drop = self
            .last_delta
            .as_ref()
            .map(|d| d.get_part().is_empty())
            .unwrap_or(false);
        if should_drop {
            self.last_delta.take();
            return;
        }

        if let Some(last_delta) = self.last_delta.take() {
            // Ensure we have a target message for this role.
            let mut msg = match self.last_message.take() {
                Some(m) => m,
                None => Message::new(last_delta.get_role().to_owned()),
            };

            match last_delta {
                MessageDelta::Content(_, part) => msg.content.push(part),
                MessageDelta::Reasoning(_, part) => msg.reasoning.push(part),
                MessageDelta::ToolCall(_, part) => msg.tool_calls.push(part),
            }

            self.last_message = Some(msg);
        }
    }

    /// Feed one streaming delta.
    ///
    /// Returns `Some(Message)` only when the **role changes**, which closes and yields
    /// the previously aggregated message. Otherwise returns `None`.
    pub fn update(&mut self, delta: MessageDelta) -> Option<Message> {
        // If the incoming role differs, close out the current message.
        if self
            .last_delta
            .as_ref()
            .map(|d| d.get_role() != delta.get_role())
            .unwrap_or(false)
        {
            self.flush_delta_into_message();
            self.last_delta = Some(delta);
            return self.last_message.take();
        }

        // Try in-place coalescing; otherwise flush and replace buffer.
        match (&mut self.last_delta, delta) {
            // Content(Text) + Content(Text)
            (
                Some(MessageDelta::Content(_, Part::Text(acc))),
                MessageDelta::Content(_, Part::Text(new)),
            ) => {
                acc.push_str(&new);
            }

            // Reasoning(Text) + Reasoning(Text)
            (
                Some(MessageDelta::Reasoning(_, Part::Text(acc))),
                MessageDelta::Reasoning(_, Part::Text(new)),
            ) => {
                acc.push_str(&new);
            }

            // ToolCall(Function{..}) + ToolCall(Function{..})
            // with same ID
            (
                Some(MessageDelta::ToolCall(
                    _,
                    Part::Function {
                        id: last_id,
                        function: last_function,
                        ..
                    },
                )),
                MessageDelta::ToolCall(_, Part::Function { id, function }),
            ) if !(last_id.is_some()
                && id.is_some()
                && last_id.clone().unwrap() != id.clone().unwrap()) =>
            {
                last_function.push_str(&function);
            }

            // No prior buffer → start buffering.
            (None, delta) => {
                self.last_delta = Some(delta);
            }

            // Not mergeable → flush buffer into message, then buffer the new delta.
            (Some(_), delta) => {
                self.flush_delta_into_message();
                self.last_delta = Some(delta);
            }
        }
        self.last_message.take()
    }

    /// Finalizes the aggregator, flushing any buffered content and returning the
    /// last in-progress message, if present.
    pub fn finalize(mut self) -> Option<Message> {
        self.flush_delta_into_message();
        self.last_message
    }
}
