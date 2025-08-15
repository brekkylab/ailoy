use std::fmt::{self, Display};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap,
};
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

    pub fn get_text_mut(&mut self) -> Option<&mut String> {
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

    pub fn get_function_mut(&mut self) -> Option<&mut String> {
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

    pub fn is_empty(&self) -> bool {
        match self {
            Part::Text(v) => v.is_empty(),
            Part::Function { function: v, .. } => v.is_empty(),
            Part::ImageURL(_) => false,
            Part::ImageData(v) => v.is_empty(),
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
            Ok(Part::Text(value))
        } else if ty == "function" && key == "function" {
            match serde_json::from_str(&value) {
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
            match url::Url::parse(&value) {
                Ok(value) => Ok(Part::ImageURL(value)),
                Err(err) => Err(de::Error::custom(format!(
                    "Invalid URL: {} {}",
                    value,
                    err.to_string()
                ))),
            }
        } else if ty == "image" && key == "data" {
            Ok(Part::ImageData(value))
        } else {
            Err(de::Error::custom("Invalid type"))
        }
    }
}

impl Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Part::Text(text) => {
                f.write_fmt(format_args!("Part(type=\"text\", text=\"{}\")", text))?
            }
            Part::Function { id, function } => f.write_fmt(format_args!(
                "Part(type=\"function\", id=\"{}\", function=\"{}\")",
                if let Some(id) = id { id } else { "None" },
                function
            ))?,
            Part::ImageURL(url) => f.write_fmt(format_args!(
                "Part(type=\"image\", url=\"{}\")",
                url.to_string()
            ))?,
            Part::ImageData(data) => f.write_fmt(format_args!(
                "Part(type=\"image\", data=({} bytes))",
                data.len()
            ))?,
        };
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

fn text_or_part_vector<'de, D>(de: D) -> Result<Vec<Part>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum Either {
        Str(String),
        Parts(Vec<Part>),
    }

    match Either::deserialize(de)? {
        Either::Str(s) => {
            if s.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![Part::new_text(s)])
            }
        }
        Either::Parts(v) => Ok(v),
    }
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, deserialize_with = "text_or_part_vector")]
    pub content: Vec<Part>,
    #[serde(default, deserialize_with = "text_or_part_vector")]
    pub reasoning: Vec<Part>,
    #[serde(default)]
    pub tool_calls: Vec<Part>,
}

impl Message {
    /// Create an empty message
    pub fn new(role: Role) -> Message {
        Message {
            role,
            reasoning: Vec::new(),
            content: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    /// Create a message with initial content
    pub fn with_content(role: Role, content: Part) -> Message {
        Message {
            role,
            reasoning: Vec::new(),
            content: vec![content],
            tool_calls: Vec::new(),
        }
    }
}

impl Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut to_write = vec![format!("role={}", &self.role)];
        if self.reasoning.len() > 0 {
            to_write.push(format!(
                "reasoning=[{}]",
                &self
                    .reasoning
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if self.content.len() > 0 {
            to_write.push(format!(
                "content=[{}]",
                &self
                    .content
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if self.tool_calls.len() > 0 {
            to_write.push(format!(
                "tool_calls=[{}]",
                &self
                    .tool_calls
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        f.write_str(&format!("Message({})", to_write.join(", ")))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MessageDelta {
    pub role: Option<Role>,
    #[serde(default, deserialize_with = "text_or_part_vector")]
    pub reasoning: Vec<Part>,
    #[serde(default, deserialize_with = "text_or_part_vector")]
    pub content: Vec<Part>,
    #[serde(default)]
    pub tool_calls: Vec<Part>,
}

impl MessageDelta {
    pub fn new() -> MessageDelta {
        MessageDelta {
            role: None,
            reasoning: Vec::new(),
            content: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_role(self, role: Role) -> MessageDelta {
        MessageDelta {
            role: Some(role),
            reasoning: self.reasoning,
            content: self.content,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_reasoning(self, reasoning: Vec<Part>) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning,
            content: self.content,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_content(self, content: Vec<Part>) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning: self.reasoning,
            content,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_tool_calls(self, tool_calls: Vec<Part>) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning: self.reasoning,
            content: self.content,
            tool_calls,
        }
    }
}

impl Display for MessageDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut to_write = Vec::<String>::new();
        if let Some(role) = &self.role {
            to_write.push(format!("role={}", role));
        };

        if self.reasoning.len() > 0 {
            to_write.push(format!(
                "reasoning=[{}]",
                &self
                    .reasoning
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if self.content.len() > 0 {
            to_write.push(format!(
                "content=[{}]",
                &self
                    .content
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        if self.tool_calls.len() > 0 {
            to_write.push(format!(
                "tool_calls=[{}]",
                &self
                    .tool_calls
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
        }
        f.write_fmt(format_args!("MessageDelta({})", to_write.join(", ")))?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, EnumString, Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MessageOutput {
    pub delta: MessageDelta,
    pub finish_reason: Option<FinishReason>,
}

impl MessageOutput {
    pub fn new() -> MessageOutput {
        MessageOutput {
            delta: MessageDelta::new(),
            finish_reason: None,
        }
    }

    pub fn with_delta(self, delta: MessageDelta) -> Self {
        MessageOutput {
            delta,
            finish_reason: self.finish_reason,
        }
    }

    pub fn with_finish_reason(self, finish_reason: FinishReason) -> Self {
        MessageOutput {
            delta: self.delta,
            finish_reason: Some(finish_reason),
        }
    }
}

impl Display for MessageOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("MessageOutput(delta="))?;
        self.delta.fmt(f)?;
        if let Some(finish_reason) = &self.finish_reason {
            f.write_str(&format!(", finish_reason="))?;
            finish_reason.fmt(f)?;
        };
        f.write_str(&format!(")"))?;
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
/// assert!(agg.update(MessageDelta::new_assistant_tool_call(Part::Text("Hel".into()))).is_none());
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
    buffer: Option<Message>,
}

impl MessageAggregator {
    /// Creates a fresh aggregator with no buffered state.
    pub fn new() -> Self {
        MessageAggregator { buffer: None }
    }

    /// Feed one streaming delta.
    ///
    /// Returns `Some(Message)`, which closes and yields if message is completed, otherwise `None`.
    pub fn update(&mut self, msg_out: MessageOutput) -> Option<Message> {
        let delta = msg_out.delta;
        if let Some(role) = delta.role {
            if self.buffer.is_some() {
                todo!()
            } else {
                self.buffer = Some(Message::new(role));
            }
        };

        let buffer = self.buffer.as_mut().expect("Role not specified");
        for part in delta.reasoning {
            if buffer.reasoning.is_empty() {
                buffer.reasoning.push(part);
            } else {
                let last_part = buffer.reasoning.last_mut().unwrap();
                if last_part.is_text() && part.is_text() {
                    last_part
                        .get_text_mut()
                        .unwrap()
                        .push_str(part.get_text().unwrap());
                } else {
                    buffer.reasoning.push(part);
                }
            }
        }
        for part in delta.content {
            if buffer.content.is_empty() {
                buffer.content.push(part);
            } else {
                let last_part = buffer.content.last_mut().unwrap();
                if last_part.is_text() && part.is_text() {
                    last_part
                        .get_text_mut()
                        .unwrap()
                        .push_str(part.get_text().unwrap());
                } else {
                    buffer.content.push(part);
                }
            }
        }
        for part in delta.tool_calls {
            if buffer.tool_calls.is_empty() {
                buffer.tool_calls.push(part);
            } else {
                let last_part = buffer.tool_calls.last_mut().unwrap();
                if last_part.is_function() && part.is_function() {
                    last_part
                        .get_function_mut()
                        .unwrap()
                        .push_str(part.get_function().unwrap());
                } else {
                    buffer.tool_calls.push(part);
                }
            }
        }

        if msg_out.finish_reason.is_some() {
            return self.buffer.take();
        } else {
            None
        }
    }
}
