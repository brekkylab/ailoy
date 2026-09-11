//! What a run tells the UI while it happens.
//!
//! One flat, tagged enum so every transport (Tauri channel, websocket, test
//! harness) carries the same shape: live fragments to paint, finalized messages
//! to persist, and the run's terminal state.

use ailoy::datatype::Value;
use ailoy::message::{Message, RateLimitInfo, TokenUsage};
use serde::{Deserialize, Serialize};

/// One event emitted over the life of a run.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEvent {
    /// The run was accepted and is now streaming.
    Started { run_id: String },
    /// Newly streamed assistant text, to append to the live bubble.
    TextDelta { text: String },
    /// Newly streamed reasoning text, to append to the live thinking region.
    ThinkingDelta { text: String },
    /// A tool call began executing (already approved, or not gated).
    ToolCallStarted {
        id: String,
        name: String,
        arguments: Value,
    },
    /// A message finalized at a stream boundary, with its store sequence number.
    Message {
        seq: i64,
        depth: u8,
        source_agent: Option<String>,
        message: Message,
        usage: Option<TokenUsage>,
    },
    /// Refreshed token/context accounting for the session.
    Usage {
        usage: Option<TokenUsage>,
        rate_limit: Option<RateLimitInfo>,
        context_used: Option<u64>,
        context_limit: Option<u64>,
    },
    /// A tool call is held pending the user's approval.
    AwaitingApproval {
        id: String,
        name: String,
        arguments: Value,
    },
    /// The run finished normally.
    Done,
    /// The run stopped because the user cancelled it.
    Cancelled,
    /// The run stopped because of a failure.
    Error { kind: String, message: String },
}
