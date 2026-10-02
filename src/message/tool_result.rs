use serde::{Deserialize, Serialize};

use crate::{datatype::Value, message::Part};

/// An intermediate progress update yielded by a streaming tool during execution.
///
/// Items are **independent snapshots**, not accumulated into the final tool result;
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

/// Output of a streaming tool: zero or more [`Delta`](Self::Delta)s, then exactly one [`Result`](Self::Result).
#[derive(Clone, Debug)]
pub enum StreamingToolOutput {
    /// Intermediate progress snapshot. Not added to history.
    Delta(ToolResultDelta),
    /// The definitive tool result. Must be the last item in the stream.
    Result(Value),
}
