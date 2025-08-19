use std::fmt::{self, Display};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{self, MapAccess},
    ser::SerializeMap as _,
};
use strum::{Display, EnumString};

use crate::value::{Part, PartStyle, StyledPart};

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
    pub role: Option<Role>,
    pub reasoning: String,
    pub contents: Vec<Part>,
    pub tool_calls: Vec<Part>,
}

impl Message {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_role(role: Role) -> Self {
        Message {
            role: Some(role),
            reasoning: String::new(),
            contents: Vec::new(),
            tool_calls: Vec::new(),
        }
    }

    pub fn with_reasoning(self, reasoning: impl Into<String>) -> Self {
        Message {
            role: self.role,
            reasoning: reasoning.into(),
            contents: self.contents,
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_contents(self, contents: impl IntoIterator<Item = Part>) -> Self {
        Message {
            role: self.role,
            reasoning: self.reasoning,
            contents: contents.into_iter().collect(),
            tool_calls: self.tool_calls,
        }
    }

    pub fn with_tool_calls(self, tool_calls: impl IntoIterator<Item = Part>) -> Self {
        Message {
            role: self.role,
            reasoning: self.reasoning,
            contents: self.contents,
            tool_calls: tool_calls.into_iter().collect(),
        }
    }
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: None,
            reasoning: String::new(),
            contents: Vec::new(),
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MessageStyle {
    pub part_style: PartStyle,

    /// {"||HERE||": "...", "content": [...], "tool_calls": [...]}
    /// default: "reasoning"
    pub reasoning_field: String,

    /// {"reasoning": "...", "||HERE||": [...], "tool_calls": [...]}
    /// default: "content"
    pub contents_field: String,

    /// {"reasoning": "...", "content": ||HERE||, "tool_calls": [...]}
    /// true: ser/de as a string (vector must be a length 1 with text part)
    /// false: ser/de as a vector of parts (usually multimodal)
    /// default: false
    pub contents_textonly: bool,

    /// {"reasoning": "...", "content": [...], "||HERE||": [...]}
    /// default: "tool_calls"
    pub tool_calls_field: String,
}

impl MessageStyle {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for MessageStyle {
    fn default() -> Self {
        Self {
            part_style: PartStyle::default(),
            reasoning_field: String::from("reasoning"),
            contents_field: String::from("content"),
            contents_textonly: false,
            tool_calls_field: String::from("tool_calls"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StyledMessage {
    pub data: Message,
    pub style: MessageStyle,
}

impl StyledMessage {
    /// Create an empty message
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_role(role: Role) -> Self {
        Self {
            data: Message {
                role: Some(role),
                reasoning: String::new(),
                contents: Vec::new(),
                tool_calls: Vec::new(),
            },
            style: MessageStyle::new(),
        }
    }

    pub fn with_reasoning(self, reasoning: impl Into<String>) -> Self {
        Self {
            data: Message {
                role: self.data.role,
                reasoning: reasoning.into(),
                contents: self.data.contents,
                tool_calls: self.data.tool_calls,
            },
            style: self.style,
        }
    }

    pub fn with_contents(self, contents: impl IntoIterator<Item = Part>) -> Self {
        Self {
            data: Message {
                role: self.data.role,
                reasoning: self.data.reasoning,
                contents: contents.into_iter().collect(),
                tool_calls: self.data.tool_calls,
            },
            style: self.style,
        }
    }

    pub fn with_tool_calls(self, tool_calls: impl IntoIterator<Item = Part>) -> Self {
        Self {
            data: Message {
                role: self.data.role,
                reasoning: self.data.reasoning,
                contents: self.data.contents,
                tool_calls: tool_calls.into_iter().collect(),
            },
            style: self.style,
        }
    }

    pub fn with_style(self, style: MessageStyle) -> Self {
        Self {
            data: self.data,
            style,
        }
    }
}

impl Default for StyledMessage {
    fn default() -> Self {
        Self {
            data: Message::default(),
            style: MessageStyle::default(),
        }
    }
}

impl Display for StyledMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Message {{")?;
        let mut prefix_comma = false;
        if self.data.role.is_some() {
            f.write_fmt(format_args!(
                "\"role\": {}",
                self.data.role.as_ref().unwrap()
            ))?;
            prefix_comma = true;
        }
        if !self.data.reasoning.is_empty() {
            if prefix_comma {
                f.write_str(", ")?;
            }
            f.write_fmt(format_args!(
                "\"{}\": \"{}\"",
                self.style.reasoning_field,
                self.data.reasoning.replace("\n", "\\n")
            ))?;
            prefix_comma = true;
        }
        if self.data.contents.len() > 0 {
            if prefix_comma {
                f.write_str(", ")?;
            }
            if self.style.contents_textonly {
                f.write_fmt(format_args!(
                    "\"{}\": \"{}\"",
                    self.style.contents_field,
                    self.data.contents[0].as_str().unwrap()
                ))?;
            } else {
                f.write_fmt(format_args!(
                    "\"{}\": [{}",
                    self.style.contents_field,
                    StyledPart {
                        data: self.data.contents[0].clone(),
                        style: self.style.part_style.clone()
                    }
                ))?;
                for data in &self.data.contents[1..] {
                    f.write_fmt(format_args!(
                        ", {}",
                        StyledPart {
                            data: data.clone(),
                            style: self.style.part_style.clone()
                        }
                    ))?;
                }
                f.write_str("]")?;
                prefix_comma = true;
            }
        }
        if self.data.tool_calls.len() > 0 {
            if prefix_comma {
                f.write_str(", ")?;
            }
            f.write_fmt(format_args!(
                "\"{}\": [{}",
                self.style.tool_calls_field,
                StyledPart {
                    data: self.data.tool_calls[0].clone(),
                    style: self.style.part_style.clone()
                }
            ))?;
            for data in &self.data.tool_calls[1..] {
                f.write_fmt(format_args!(
                    ", {}",
                    StyledPart {
                        data: data.clone(),
                        style: self.style.part_style.clone()
                    }
                ))?;
            }
            f.write_str("]")?;
        }
        f.write_str("}}")?;
        Ok(())
    }
}

impl Serialize for StyledMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.data.role)?;

        if !self.data.reasoning.is_empty() {
            map.serialize_entry(&self.style.reasoning_field, &self.data.reasoning)?;
        }

        if self.style.contents_textonly {
            if !self.data.contents.is_empty() {
                map.serialize_entry(
                    &self.style.contents_field,
                    self.data.contents.get(0).unwrap().as_str().unwrap(),
                )?;
            }
        } else {
            if !self.data.contents.is_empty() {
                map.serialize_entry(
                    &self.style.contents_field,
                    &self
                        .data
                        .contents
                        .iter()
                        .map(|v| StyledPart {
                            data: v.clone(),
                            style: self.style.part_style.clone(),
                        })
                        .collect::<Vec<_>>(),
                )?;
            }
        }

        if !self.data.tool_calls.is_empty() {
            map.serialize_entry(
                &self.style.tool_calls_field,
                &self
                    .data
                    .tool_calls
                    .iter()
                    .map(|v| StyledPart {
                        data: v.clone(),
                        style: self.style.part_style.clone(),
                    })
                    .collect::<Vec<_>>(),
            )?;
        }

        map.end()
    }
}

impl<'de> Deserialize<'de> for StyledMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageVisitor)
    }
}

struct MessageVisitor;

impl<'de> de::Visitor<'de> for MessageVisitor {
    type Value = StyledMessage;

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
        let mut style = MessageStyle::new();

        while let Some(k) = map.next_key::<String>()? {
            if k == "role" {
                if role.is_some() {
                    return Err(de::Error::duplicate_field("role"));
                }
                role = Some(map.next_value()?);
            } else if k == "content" {
                #[derive(Deserialize)]
                #[serde(untagged)]
                enum ContentEither {
                    Null,                   // content: null
                    String(String),         // content: "..."
                    Parts(Vec<StyledPart>), // content: [ {...}, {...} ]
                }
                match map.next_value::<ContentEither>()? {
                    ContentEither::Null => {}
                    ContentEither::String(s) => {
                        if !s.is_empty() {
                            contents = vec![Part::new_text(s)];
                        }
                        style.contents_textonly = true;
                    }
                    ContentEither::Parts(mut v) => {
                        contents.reserve(v.len());
                        for part in v.drain(..) {
                            contents.push(part.data);
                            style
                                .part_style
                                .update(part.style)
                                .map_err(|e| de::Error::custom(e))?;
                        }
                        style.contents_textonly = false;
                    }
                }
                style.contents_field = k;
            } else if k == "reasoning" || k == "reasoning_content" {
                reasoning = map.next_value()?;
                style.contents_field = k;
            } else if k == "tool_calls" {
                let mut v: Vec<StyledPart> = map.next_value()?;
                tool_calls.reserve(v.len());
                for part in v.drain(..) {
                    tool_calls.push(part.data);
                    style
                        .part_style
                        .update(part.style)
                        .map_err(|e| de::Error::custom(e))?;
                }
                style.contents_textonly = false;
                style.contents_field = k;
            }
        }
        Ok(StyledMessage {
            data: Message {
                role,
                contents,
                reasoning,
                tool_calls,
            },
            style,
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
    pub delta: Message,
    pub finish_reason: Option<FinishReason>,
}

impl MessageOutput {
    pub fn new() -> Self {
        Self {
            delta: Message::new(),
            finish_reason: None,
        }
    }

    pub fn with_delta(self, delta: Message) -> Self {
        Self {
            delta,
            finish_reason: self.finish_reason,
        }
    }

    pub fn with_finish_reason(self, finish_reason: FinishReason) -> Self {
        Self {
            delta: self.delta,
            finish_reason: Some(finish_reason),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StyledMessageOutput {
    pub data: MessageOutput,
    pub style: MessageStyle,
}

impl Display for StyledMessageOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut to_write: Vec<String> = Vec::new();
        if self.data.delta.role.is_some()
            || !self.data.delta.reasoning.is_empty()
            || !self.data.delta.contents.is_empty()
            || !self.data.delta.tool_calls.is_empty()
        {
            to_write.push(format!(
                "\"delta\": {}",
                StyledMessage {
                    data: self.data.delta.clone(),
                    style: self.style.clone()
                }
            ));
        }
        if let Some(finish_reason) = &self.data.finish_reason {
            to_write.push(format!("\"finish_reason\": \"{}\"", finish_reason));
        };
        f.write_str(&format!("MessageOutput {{{}}}", to_write.join(", ")))?;
        Ok(())
    }
}

impl Serialize for StyledMessageOutput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry(
            "delta",
            &StyledMessage {
                data: self.data.delta.clone(),
                style: self.style.clone(),
            },
        )?;
        if self.data.finish_reason.is_some() {
            map.serialize_entry("finish_reason", &self.data.finish_reason)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for StyledMessageOutput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MessageOutputVisitor)
    }
}

struct MessageOutputVisitor;

impl<'de> de::Visitor<'de> for MessageOutputVisitor {
    type Value = StyledMessageOutput;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str(r#"a map with "type" field and one other content key"#)
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut delta: Option<StyledMessage> = None;
        let mut finish_reason: Option<FinishReason> = None;

        while let Some(k) = map.next_key::<String>()? {
            if k == "delta" {
                if delta.is_some() {
                    return Err(de::Error::duplicate_field("delta"));
                }
                delta = Some(map.next_value()?);
            } else if k == "finish_reason" {
                finish_reason = map.next_value()?;
            }
        }
        let delta = delta.unwrap_or_default();
        Ok(StyledMessageOutput {
            data: MessageOutput {
                delta: delta.data,
                finish_reason,
            },
            style: delta.style,
        })
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
        if self.buffer.is_none() {
            self.buffer = Some(Message::new())
        }
        let buffer = self.buffer.as_mut().unwrap();
        let delta = msg_out.delta;
        if let Some(role) = delta.role {
            buffer.role = Some(role);
        };
        if !delta.reasoning.is_empty() {
            buffer.reasoning.push_str(&delta.reasoning);
        }
        for part in delta.contents {
            let last_part = buffer.contents.last_mut().unwrap();
            if let Some(part_to_push) = last_part.concatenate(part) {
                buffer.contents.push(part_to_push);
            }
        }
        for part in delta.tool_calls {
            let last_part = buffer.tool_calls.last_mut().unwrap();
            if let Some(part_to_push) = last_part.concatenate(part) {
                buffer.tool_calls.push(part_to_push);
            }
        }

        if msg_out.finish_reason.is_some() {
            return Some(self.buffer.take().unwrap());
        } else {
            None
        }
    }
}
