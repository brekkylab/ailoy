use serde::{Deserialize, Serialize};

use crate::{datatype::Value, message::Part};

/// An intermediate progress update yielded by a streaming tool during execution.
///
/// Items are **independent snapshots** — they are NOT accumulated into the final
/// [`TurnEvent::ToolResult`][crate::agent::TurnEvent::ToolResult]. Callers should
/// use them for UI progress display only.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolResultDelta {
    /// ID of the tool call this delta belongs to.
    pub tool_call_id: Option<String>,
    /// Name of the tool producing this delta.
    pub tool_name: String,
    /// The content of this progress snapshot.
    pub content: Part,
}

/// Output yielded by a streaming tool function.
///
/// A streaming tool yields zero or more [`Delta`] items (intermediate progress,
/// for UI display only), followed by exactly one [`Result`] item (the definitive
/// tool output that will be added to the agent's history and returned to the LLM).
#[derive(Clone, Debug)]
pub enum StreamingToolOutput {
    /// Intermediate progress snapshot. Not added to history.
    Delta(ToolResultDelta),
    /// The definitive tool result. Must be the last item in the stream.
    Result(Value),
}
