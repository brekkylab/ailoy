//! Reading messages into a wire format and building them from a response.

use crate::{
    datatype::Value,
    message::{FinishReason, Message, MessageOutput, Part, Role, TokenUsage},
};

/// An assistant message from a whole response. Tool calls are always `Some`, as a
/// finished stream gives them.
pub(crate) fn assistant_output(
    contents: Vec<Part>,
    thinking: Option<String>,
    signature: Option<String>,
    tool_calls: Vec<Part>,
    finish_reason: FinishReason,
    usage: Option<TokenUsage>,
) -> MessageOutput {
    let mut message = Message::new(Role::Assistant).with_contents(contents);
    message.thinking = thinking;
    message.signature = signature;
    message.tool_calls = Some(tool_calls);
    MessageOutput {
        message,
        finish_reason,
        usage,
        depth: None,
        source_agent: None,
    }
}

/// The text of every system message, joined by blank lines.
pub(crate) fn system_text(messages: &[Message]) -> String {
    messages
        .iter()
        .filter(|m| m.role == Role::System)
        .flat_map(|m| &m.contents)
        .filter_map(|p| match p {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The index of the last user message, after which assistant thinking is replayed; past
/// the end when there is none.
pub(crate) fn last_user_index(messages: &[Message]) -> usize {
    messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len())
}

/// A [`Part::Value`] as text, for wires that take only text there: a string as-is,
/// anything else as JSON.
pub(crate) fn value_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// A JSON number as a token count; anything else as `None`.
pub(crate) fn tokens(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
}
