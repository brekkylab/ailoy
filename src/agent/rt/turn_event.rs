use crate::{
    datatype::Value,
    message::{Message, MessageOutput, ToolResultDelta},
};

/// Events yielded by [`AgentRuntime::stream_turn`].
///
/// Callers can pattern-match on this enum to handle different stages of
/// a single agent turn: LLM responses, tool calls, tool results, and
/// intermediate streaming progress from long-running tools.
#[derive(Clone, Debug)]
pub enum TurnEvent {
    /// An assistant message from the LLM.
    ///
    /// When this is the last event in the stream, it is the final assistant
    /// message for the turn. If tool calls follow, more events will be emitted
    /// before the stream ends.
    AssistantMessage(MessageOutput),

    /// A tool call extracted from an assistant message.
    ///
    /// Emitted once per tool call, immediately after the corresponding
    /// [`TurnEvent::AssistantMessage`], before tool execution begins.
    ToolCall {
        id: Option<String>,
        name: String,
        arguments: Value,
    },

    /// A finalized, atomic tool result that has been added to history.
    ///
    /// Emitted after all [`TurnEvent::ToolDelta`] items (if any) for a given
    /// tool call. This is the value the LLM will see in the next turn.
    ToolResult(Message),

    /// An intermediate progress snapshot from a streaming tool.
    ///
    /// These are emitted during tool execution for UI progress display only.
    /// They are **not** accumulated into [`TurnEvent::ToolResult`] — each
    /// delta is an independent snapshot. The stream's final delta determines
    /// the tool result.
    ToolDelta(ToolResultDelta),
}

impl TurnEvent {
    /// Returns the `MessageOutput` if this is an `AssistantMessage`.
    pub fn as_assistant_message(&self) -> Option<&MessageOutput> {
        match self {
            TurnEvent::AssistantMessage(output) => Some(output),
            _ => None,
        }
    }

    /// Returns the `Message` if this is a `ToolResult`.
    pub fn as_tool_result(&self) -> Option<&Message> {
        match self {
            TurnEvent::ToolResult(msg) => Some(msg),
            _ => None,
        }
    }

    /// Returns the `ToolResultDelta` if this is a `ToolDelta`.
    pub fn as_tool_delta(&self) -> Option<&ToolResultDelta> {
        match self {
            TurnEvent::ToolDelta(delta) => Some(delta),
            _ => None,
        }
    }

    /// Returns `true` if this is a `ToolDelta`.
    pub fn is_tool_delta(&self) -> bool {
        matches!(self, TurnEvent::ToolDelta(_))
    }

    /// Returns `true` if this is a `ToolResult`.
    pub fn is_tool_result(&self) -> bool {
        matches!(self, TurnEvent::ToolResult(_))
    }
}
