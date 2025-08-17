use std::fmt::{self, Display};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess},
    ser::SerializeMap as _,
};
use strum::{Display, EnumString};

use crate::value::{Part, PartFmt, PartWithFmt, ToolCall};

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

// fn text_or_part_vector<'de, D>(de: D) -> Result<Vec<Part>, D::Error>
// where
//     D: serde::Deserializer<'de>,
// {
//     #[derive(serde::Deserialize)]
//     #[serde(untagged)]
//     enum Either {
//         Null(()),
//         Str(String),
//         Parts(Vec<Part>),
//     }

//     match Either::deserialize(de)? {
//         Either::Null(()) => Ok(vec![]),
//         Either::Str(s) => {
//             if s.is_empty() {
//                 Ok(vec![])
//             } else {
//                 Ok(vec![Part::new_text(s)])
//             }
//         }
//         Either::Parts(v) => Ok(v),
//     }
// }

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
/// - `contents`: Primary, user-visible content as a list of parts (text, images, etc.).
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
///   "contents": [
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
///   "contents": [ { "type": "text", "text": "I'll look that up." } ],
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
///   "contents": [ { "type": "text", "text": "The current temperature is 42 °C." } ],
///   "reasoning": [ { "type": "text", "text": "(model reasoning tokens, if exposed)" } ]
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub reasoning: String,
    pub contents: Vec<Part>,
    pub tool_calls: Vec<Part>,
}

impl Message {
    /// Create an empty message
    pub fn new(role: Role) -> Message {
        Message {
            role,
            reasoning: String::new(),
            contents: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_reasoning(self, reasoning: impl Into<String>) -> Message {
        Message {
            role: self.role,
            reasoning: reasoning.into(),
            contents: self.contents,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_contents(self, contents: impl IntoIterator<Item = Part>) -> Message {
        Message {
            role: self.role,
            reasoning: self.reasoning,
            contents: contents.into_iter().collect(),
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_tool_calls(self, tool_calls: impl IntoIterator<Item = Part>) -> Message {
        Message {
            role: self.role,
            reasoning: self.reasoning,
            contents: self.contents,
            tool_calls: tool_calls.into_iter().collect(),
        }
    }
}

impl Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut to_write = vec![format!("role={}", &self.role)];
        if !self.reasoning.is_empty() {
            to_write.push(format!("reasoning={}", &self.reasoning));
        }
        if self.contents.len() > 0 {
            to_write.push(format!(
                "contents=[{}]",
                &self
                    .contents
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
        f.write_str(&format!("Message {{ {} }}", to_write.join(", ")))?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MessageFmt {
    pub part_fmt: PartFmt,

    /// {"|HERE|": "...", "content": [...], "tool_calls": [...]}
    /// default: "reasoning"
    pub reasoning_field: String,

    /// it the value is true, put `"reasoning": null` markers
    /// if the value is false, no field.
    /// default: false
    pub reasoning_null_marker: bool,

    /// {"reasoning": "...", "|HERE|": [...], "tool_calls": [...]}
    /// default: "content"
    pub contents_field: String,

    /// {"reasoning": "...", "contents": |HERE|, "tool_calls": [...]}
    /// true: ser/de as a string (vector must be a length 1 with text part)
    /// false: ser/de as a vector of parts (usually multimodal)
    /// default: "false"
    pub contents_textonly: bool,

    /// it the value is true, put `"contents": null` markers
    /// if the value is false, no field.
    /// default: false
    pub contents_null_marker: bool,

    /// {"reasoning": "...", "contents": [...], "|HERE|": [...]}
    /// default: "tool_calls"
    pub tool_calls_field: String,

    /// it the value is true, put `"tool_calls": null` markers
    /// if the value is false, no field.
    /// default: false
    pub tool_calls_null_marker: bool,
}

impl MessageFmt {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for MessageFmt {
    fn default() -> Self {
        Self {
            part_fmt: PartFmt::default(),
            reasoning_field: String::from("reasoning"),
            reasoning_null_marker: false,
            contents_field: String::from("content"),
            contents_textonly: false,
            contents_null_marker: false,
            tool_calls_field: String::from("tool_calls"),
            tool_calls_null_marker: false,
        }
    }
}

impl Serialize for Message {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted = MessageWithFmt::new(self, MessageFmt::new());
        formatted.serialize(serializer)
    }
}

#[derive(Debug, Clone)]
pub struct MessageWithFmt<'a>(&'a Message, MessageFmt);

impl<'a> MessageWithFmt<'a> {
    pub fn new(msg: &'a Message, fmt: MessageFmt) -> Self {
        Self(msg, fmt)
    }
}

impl<'a> Serialize for MessageWithFmt<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.0.role)?;

        if !self.0.reasoning.is_empty() {
            map.serialize_entry(&self.1.reasoning_field, &self.0.reasoning)?;
        } else if self.1.reasoning_null_marker {
            map.serialize_entry(&self.1.reasoning_field, &())?;
        }

        if self.1.contents_textonly {
            if !self.0.contents.is_empty() {
                map.serialize_entry(
                    &self.1.contents_field,
                    self.0.contents.get(0).unwrap().get_text().unwrap(),
                )?;
            }
        } else {
            if !self.0.contents.is_empty() {
                map.serialize_entry(
                    &self.1.contents_field,
                    &self
                        .0
                        .contents
                        .iter()
                        .map(|v| PartWithFmt::new(v, self.1.part_fmt.clone()))
                        .collect::<Vec<_>>(),
                )?;
            } else if self.1.contents_null_marker {
                map.serialize_entry(&self.1.contents_field, &())?;
            }
        }

        if !self.0.tool_calls.is_empty() {
            map.serialize_entry(
                &self.1.tool_calls_field,
                &self
                    .0
                    .tool_calls
                    .iter()
                    .map(|v| PartWithFmt::new(v, self.1.part_fmt.clone()))
                    .collect::<Vec<_>>(),
            )?;
        } else if self.1.tool_calls_null_marker {
            map.serialize_entry(&self.1.tool_calls_field, &())?;
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for Message {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> de::Visitor<'de> for MessageVisitor {
    type Value = Message;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut role: Option<Role> = None;
        let mut contents: Vec<Part> = Vec::new();
        let mut reasoning: String = String::new();
        let mut tool_calls: Vec<Part> = Vec::new();

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else if k == "content" {
                contents = map.next_value()?;
            } else if k == "reasoning" || k == "reasoning_content" {
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
            contents,
            reasoning,
            tool_calls,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct MessageDelta {
    pub role: Option<Role>,
    pub reasoning: String,
    pub contents: Vec<Part>,
    pub tool_calls: Vec<Part>,
}

impl MessageDelta {
    pub fn new() -> MessageDelta {
        MessageDelta {
            role: None,
            reasoning: String::new(),
            contents: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_role(self, role: Role) -> MessageDelta {
        MessageDelta {
            role: Some(role),
            reasoning: self.reasoning,
            contents: self.contents,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_reasoning(self, reasoning: String) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning,
            contents: self.contents,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_contents(self, contents: Vec<Part>) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning: self.reasoning,
            contents,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_tool_calls(self, tool_calls: Vec<Part>) -> MessageDelta {
        MessageDelta {
            role: self.role,
            reasoning: self.reasoning,
            contents: self.contents,
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
            to_write.push(format!("reasoning=\"{}\"", &self));
        }
        if self.contents.len() > 0 {
            to_write.push(format!(
                "contents=[{}]",
                &self
                    .contents
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
        f.write_fmt(format_args!("MessageDelta {{ {} }}", to_write.join(", ")))?;
        Ok(())
    }
}

impl Serialize for MessageDelta {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted = MessageDeltaWithFmt::new(self, MessageFmt::default());
        formatted.serialize(serializer)
    }
}

#[derive(Debug, Clone)]
pub struct MessageDeltaWithFmt<'a>(&'a MessageDelta, MessageFmt);

impl<'a> MessageDeltaWithFmt<'a> {
    pub fn new(inner: &'a MessageDelta, fmt: MessageFmt) -> Self {
        Self(inner, fmt)
    }
}

impl<'a> Serialize for MessageDeltaWithFmt<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        if self.0.role.is_some() {
            map.serialize_entry("role", &self.0.role)?;
        }

        if !self.0.reasoning.is_empty() {
            map.serialize_entry(&self.1.reasoning_field, &self.0.reasoning)?;
        } else if self.1.reasoning_null_marker {
            map.serialize_entry(&self.1.reasoning_field, &())?;
        }

        if self.1.contents_textonly {
            todo!()
        } else {
            if !self.0.contents.is_empty() {
                let v = self
                    .0
                    .contents
                    .iter()
                    .map(|v| PartWithFmt::new(v, self.1.part_fmt.clone()))
                    .collect::<Vec<_>>();
                map.serialize_entry(&self.1.contents_field, &v)?;
            } else if self.1.contents_null_marker {
                map.serialize_entry(&self.1.contents_field, &())?;
            }
        }

        if !self.0.tool_calls.is_empty() {
            let v = self
                .0
                .tool_calls
                .iter()
                .map(|v| PartWithFmt::new(v, self.1.part_fmt.clone()))
                .collect::<Vec<_>>();
            map.serialize_entry(&self.1.tool_calls_field, &v)?;
        } else if self.1.tool_calls_null_marker {
            map.serialize_entry(&self.1.tool_calls_field, &())?;
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for MessageDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessagDeltaVisitor)
    }
}

struct MessagDeltaVisitor;

impl<'de> de::Visitor<'de> for MessagDeltaVisitor {
    type Value = MessageDelta;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut role: Option<Role> = None;
        let mut contents: Vec<Part> = Vec::new();
        let mut reasoning: String = String::new();
        let mut tool_calls: Vec<Part> = Vec::new();

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else if k == "content" {
                contents = map.next_value()?;
            } else if k == "reasoning" || k == "reasoning_content" {
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
        Ok(MessageDelta {
            role,
            contents,
            reasoning,
            tool_calls,
        })
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

#[derive(Debug, Clone, Default)]
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

struct MessageOutputWithFmt<'a>(&'a MessageOutput, MessageFmt);

impl<'a> MessageOutputWithFmt<'a> {
    pub fn new(inner: &'a MessageOutput, fmt: MessageFmt) -> Self {
        Self(inner, fmt)
    }
}

impl<'a> Serialize for MessageOutputWithFmt<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry(
            "delta",
            &MessageDeltaWithFmt::new(&self.0.delta, self.1.clone()),
        )?;
        if self.0.finish_reason.is_some() {
            map.serialize_entry("finish_reason", &self.0.finish_reason)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for MessageOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageOutputVisitor)
    }
}

struct MessageOutputVisitor;

impl<'de> de::Visitor<'de> for MessageOutputVisitor {
    type Value = MessageOutput;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut delta: Option<MessageDelta> = None;
        let mut finish_reason: Option<FinishReason> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "delta" {
                if delta.is_some() {
                    return Err(de::Error::duplicate_field("delta"));
                }
                delta = Some(map.next_value()?);
            } else if k == "finish_reason" {
                finish_reason = map.next_value()?;
            } else {
                return Err(de::Error::unknown_field(&k, &["delta", "finish_reason"]));
            }
        }
        let delta = delta.ok_or_else(|| de::Error::missing_field("delta"))?;
        Ok(MessageOutput {
            delta,
            finish_reason,
        })
    }
}

impl Serialize for MessageOutput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let formatted = MessageOutputWithFmt::new(self, MessageFmt::default());
        formatted.serialize(serializer)
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
        if !delta.reasoning.is_empty() {
            if buffer.reasoning.is_empty() {
                buffer.reasoning = delta.reasoning;
            } else {
                buffer.reasoning.push_str(&delta.reasoning);
            }
        }
        for part in delta.contents {
            if buffer.contents.is_empty() {
                buffer.contents.push(part);
            } else {
                let last_part = buffer.contents.last_mut().unwrap();
                if last_part.is_text() && part.is_text() {
                    last_part
                        .get_text_mut()
                        .unwrap()
                        .push_str(part.get_text().unwrap());
                } else {
                    buffer.contents.push(part);
                }
            }
        }
        for part in delta.tool_calls {
            if buffer.tool_calls.is_empty() {
                buffer.tool_calls.push(part);
            } else {
                let last_part = buffer.tool_calls.last_mut().unwrap();
                if last_part.is_text() && part.is_text() {
                    last_part
                        .get_text_mut()
                        .unwrap()
                        .push_str(part.get_text().unwrap());
                } else {
                    buffer.tool_calls.push(part);
                }
            }
        }

        if msg_out.finish_reason.is_some() {
            let mut rv = self.buffer.take().unwrap();
            rv.tool_calls = rv
                .tool_calls
                .into_iter()
                .map(|v| {
                    match v {
                        Part::Text(text) => match serde_json::from_str::<ToolCall>(&text) {
                            Ok(res) => Part::Function {
                                id: None,
                                function: res,
                            },
                            Err(_) => Part::Text(text),
                        },
                        Part::Function { id, function } => Part::Function { id, function },
                        Part::ImageURL(url) => Part::ImageURL(url),
                        Part::ImageData(data) => Part::ImageData(data),
                        Part::AudioURL(url) => Part::AudioURL(url),
                        Part::AudioData(data) => Part::AudioData(data),
                    };
                    Part::Text("()".to_owned())
                })
                .collect();
            return Some(rv);
        } else {
            None
        }
    }
}
