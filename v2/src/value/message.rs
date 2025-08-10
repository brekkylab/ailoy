use std::fmt;

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess, Visitor},
    ser::SerializeMap as _,
};
use url::Url;

/// Represents a single, typed unit of message content (multi-modal).
///
/// A `Part` models one element inside a message’s `content` (or related fields like
/// `reasoning` / `tool_calls`). It is designed to be **OpenAI-compatible** when serialized,
/// using the modern “array-of-parts” shape (e.g., `{ "type": "text", "text": "..." }`,
/// `{ "type": "input_image", ... }`).
///
/// # Variants
/// - [`Part::Text`]: Plain UTF-8 text.
/// - [`Part::Json`]: Opaque JSON text **as-is** (see notes below).
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
///   {
///     "type": "input_image",
///     "image_url": { "url": "https://example.com/cat.png" }
///   }
///   ```
///
/// - `ImageBase64(data)`
///   ```json
///   {
///     "type": "input_image",
///     "image": { "data": "<base64-bytes>", "mime_type": "image/png" }
///   }
///   ```
///   *Note:* This enum stores only the base64 string. The MIME type is provided by
///   the serializer (often `"image/png"`) or by higher-level configuration.
///
/// - `Json(s)`
///   This is intentionally **opaque**. It’s commonly used for intermediary content such as
///   tool/function call arguments or other JSON blobs produced by an LLM stream. The raw string
///   is preserved byte-for-byte. Downstream code may wrap or parse it depending on context
///   (e.g., when placed under a message’s `tool_calls` field).
///
/// # Notes on `Json`
/// - **As-is storage:** The string may be incomplete or invalid JSON while streaming; this is
///   expected. Callers that need structured data should finalize aggregation first and then parse.
/// - **No validation:** The type does not validate or alter the JSON string in any way.
///
/// # Invariants & behavior
/// - Order is preserved by the container (e.g., `Message.content`).
/// - Image parts do not fetch or validate URLs/bytes at construction time.
/// - The enum itself is transport-agnostic; OpenAI-compatibility is achieved by the (de)serializer.
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
/// Collecting raw tool call arguments as JSON during streaming:
/// ```rust
/// # use crate::value::Part;
/// let mut args = String::new();
/// // append chunks as they arrive:
/// args.push_str("{\"location\":\"Dubai\"");
/// args.push_str(",\"unit\":\"Celsius\"}");
/// let json_part = Part::Json(args); // parse after the stream is complete
/// ```
///
/// Embedding a base64 image:
/// ```rust
/// # use crate::value::Part;
/// let b64 = base64::encode(&[/* bytes */]);
/// let img = Part::ImageBase64(b64);
/// // Serializer will include a suitable mime_type (e.g., image/png) if configured.
/// ```
#[derive(Clone, Debug)]
pub enum Part {
    /// Plain UTF-8 text.
    Text(String),

    /// Raw JSON text captured exactly as produced by the model.
    ///
    /// Note that it can contain **incomplete or invalid JSON** while streaming;
    /// this type stores it verbatim. Parse only after finalization when needed.
    Json(String),

    /// Web-addressable image (HTTP(S) URL).
    ///
    /// Typically serialized as:
    /// `{ "type":"input_image", "image_url": { "url": "<...>" } }`.
    ImageURL(Url),

    /// Inline, base64-encoded image bytes.
    ///
    /// Typically serialized as:
    /// `{ "type":"input_image", "image": { "data": "<base64>", "mime_type": "image/png" } }`.
    /// The MIME type is provided by higher-level configuration/serialization.
    ImageBase64(String),
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

    /// Constructor for JSON part
    pub fn new_json<T: Into<String>>(json: T) -> Part {
        Part::Json(json.into())
    }

    pub fn is_json(&self) -> bool {
        match self {
            Part::Json(_) => true,
            _ => false,
        }
    }

    pub fn get_json(&self) -> Option<&String> {
        match self {
            Part::Json(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_json_owned(&self) -> Option<String> {
        match self {
            Part::Json(v) => Some(v.to_owned()),
            _ => None,
        }
    }

    /// Constructor for image URL part
    pub fn new_image_url<T: Into<Url>>(url: T) -> Part {
        Part::ImageURL(url.into())
    }

    /// Constructor for image base64 part
    pub fn new_image_base64<T: Into<String>>(encoded: T) -> Part {
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
/// - `tool_calls` should be present only on assistant messages.
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
///       "id": "call_abc123",
///       "type": "function",
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

/// Incrementally builds a complete [`Message`] from a stream of [`MessageDelta`]s.
///
/// # Overview
/// `MessageAggregator` is a small state machine that:
/// - Buffers the most recent delta (`last_delta`) so adjacent chunks of the **same kind**
///   can be coalesced to reduce fragmentation and allocations.
/// - Flushes the buffered chunk into the target [`Message`] (`aggregated`) whenever the
///   incoming delta can’t be merged (e.g., the variant changes) or when you call [`finalize`].
///
/// Today it merges only these contiguous cases:
/// - `MessageDelta::Content(Part::Text)` + `MessageDelta::Content(Part::Text)` → string append
/// - `MessageDelta::Reasoning(Part::Text)` + `MessageDelta::Reasoning(Part::Text)` → string append
/// - `MessageDelta::ToolCall(Part::Json)` + `MessageDelta::ToolCall(Part::Json)` → string append
///
/// Any other incoming delta will cause the buffered delta to be **finalized** into the
/// aggregated [`Message`] using the appropriate `push_*` method, and the new delta becomes
/// the new buffer.
///
/// This is especially useful when consuming token-by-token or chunked outputs from
/// streaming LLM APIs where many tiny deltas of the same field arrive back-to-back.
///
/// # Lifecycle
/// 1. Create with [`new`], giving the role of the resulting aggregated message.
/// 2. Feed deltas in arrival order via [`update`].
/// 3. Call [`finalize`] **once** to flush the last buffered delta and obtain the
///    completed [`Message`]. `finalize` consumes the aggregator.
///
/// # Guarantees & Notes
/// - **Order-preserving**: deltas are incorporated in the exact order they are seen.
/// - **Zero parsing**: for `ToolCall(Part::Json)`, content is treated as a raw JSON string
///   and concatenated; parsing (if any) is your responsibility after finalization.
/// - **Cheap steady-state**: contiguous text/json chunks append in-place with `String::push_str`.
/// - **Thread safety**: this type is not synchronized; use it on one task/thread at a time.
/// - **Panics**: none expected.
///
/// # Example: simple concatenation
/// ```rust
/// # use crate::value::{MessageAggregator, MessageDelta, Message, Part, Role};
/// let mut agg = MessageAggregator::new(Role::Assistant);
///
/// agg.update(MessageDelta::Content(Part::Text("Hel".into())));
/// agg.update(MessageDelta::Content(Part::Text("lo".into())));
/// agg.update(MessageDelta::Reasoning(Part::Text(" chain-of-thought chunk".into())));
///
/// let msg: Message = agg.finalize();
/// assert_eq!(msg.content_text(), Some("Hello")); // assuming a helper accessor
/// // reasoning and other fields were also appended in arrival order.
/// ```
///
/// # Methods
/// - [`new`]: initialize with the target message role.
/// - [`update`]: feed a single [`MessageDelta`]; may coalesce with the buffered delta or flush it.
/// - [`finalize`]: flush the last buffered delta and return the completed [`Message`].
///
/// [`new`]: Self::new
/// [`update`]: Self::update
/// [`finalize`]: Self::finalize
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
                last_text.push_str(&text);
            }
            (
                Some(MessageDelta::Reasoning(Part::Text(last_text))),
                MessageDelta::Reasoning(Part::Text(text)),
            ) => {
                last_text.push_str(&text);
            }
            (
                Some(MessageDelta::ToolCall(Part::Json(last_text))),
                MessageDelta::ToolCall(Part::Json(text)),
            ) => {
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
