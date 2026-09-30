use std::fmt;

use serde::{Deserialize, Serialize};

use crate::message::Part;

/// The author of a message (or streaming delta) in a chat.
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    strum::Display,
    strum::EnumString,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Role {
    /// System instructions and constraints provided to the assistant.
    System,
    /// Content authored by the end user.
    User,
    /// Content authored by the assistant/model.
    Assistant,
    /// Outputs produced by external tools/functions
    Tool,
}

/// A chat message from a user, model, or tool: content parts, plus optional
/// thinking (with its signature) and tool calls.
///
/// # Example
///
/// ## Rust
/// ```rust
/// # use ailoy::message::{Message, Part, Role};
/// let msg = Message::new(Role::User).with_contents([Part::text("hello")]);
/// assert_eq!(msg.role, Role::User);
/// assert_eq!(msg.contents.len(), 1);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Message {
    /// Author of the message.
    pub role: Role,

    /// Primary parts of the message (e.g., text, image, value, or function).
    pub contents: Vec<Part>,

    /// Internal “thinking” text used by some models before producing final output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    /// Tool-call parts emitted alongside the main contents.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Part>>,

    /// Optional identifier for function calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// Optional signature for the `thinking` field.
    ///
    /// This is only applicable to certain LLM APIs that require a signature as part of the `thinking` payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            contents: Vec::new(),
            id: None,
            thinking: None,
            tool_calls: None,
            signature: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn with_signatured_thinking(
        mut self,
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.thinking = Some(thinking.into());
        self.signature = Some(signature.into());
        self
    }

    pub fn with_contents(mut self, contents: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        self.contents = contents.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn with_tool_calls(
        mut self,
        tool_calls: impl IntoIterator<Item = impl Into<Part>>,
    ) -> Self {
        self.tool_calls = Some(tool_calls.into_iter().map(|v| v.into()).collect());
        self
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "Message {}", s)
    }
}

/// Explains why a language model's streamed generation finished.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FinishReason {
    /// The model stopped naturally (e.g., EOS token or stop sequence).
    Stop {},

    /// Hit the maximum token/length limit.
    Length {},

    /// Stopped because a tool call was produced, waiting for it's execution.
    ToolCall {},

    /// Content was refused/filtered; string provides reason.
    Refusal { reason: String },
}

impl Default for FinishReason {
    fn default() -> Self {
        Self::Stop {}
    }
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "FinishReason {}", s)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MessageOutput {
    pub message: Message,

    pub finish_reason: FinishReason,

    /// Token usage reported by the language model for this response.
    /// Only populated for top-level LM calls; tool results leave this as `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,

    /// Tool-call nesting level relative to the top-level agent turn.
    ///
    /// `None` (tools leave it unset; the runtime assigns it) means `Some(0)`: a
    /// direct LM response or a final tool result in the top-level history.
    /// `Some(n)` is produced `n` tool-call layers deep, e.g. a sub-agent's
    /// intermediate message. The runtime emits a tool stream's intermediate
    /// outputs at `depth + 1` and its final result at 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,

    /// Name of the agent (from its [`AgentCard`]) that produced this message.
    ///
    /// `None` when the emitting agent has no `AgentCard` (e.g. a top-level
    /// agent without a card) or when the output originates from a non-agent
    /// tool.  In nested subagent chains the innermost producer's name is
    /// preserved: once set, outer agents never overwrite it.
    ///
    /// [`AgentCard`]: crate::agent::AgentCard
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_agent: Option<String>,
}

impl fmt::Display for MessageOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "MessageOutput {}", s)
    }
}
