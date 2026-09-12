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
    ///
    /// `kind` is the engine's own classification — `model`, `tool`, `stream`, `storage`,
    /// `console_unavailable`, `max_turns`, `internal`. `status` and `retryable` come from
    /// the provider and are only ever filled for `kind == "model"`: `status` is the HTTP
    /// status when a response arrived at all, and `retryable` is what a "retry" button
    /// should be enabled by. A cancelled run and a run that hit its turn limit are not
    /// failures of this kind — the first is [`RunEvent::Cancelled`], the second is an
    /// `Error` with `kind: "max_turns"` and nothing to retry.
    Error {
        kind: String,
        message: String,
        status: Option<u16>,
        retryable: bool,
    },
}

#[cfg(test)]
mod tests {
    use ailoy::datatype::Value;
    use ailoy::message::{Message, Part, Role};
    use serde_json::json;

    use super::*;

    /// The whole IPC contract for the event stream, one assertion per variant.
    ///
    /// Plan C generates its TypeScript from these shapes, and the webview matches on the
    /// `type` tag and reads the fields by name — so a rename anywhere here is a silent
    /// break on the other side of the boundary, visible only as a UI that stops painting.
    /// This test is what turns that into a compile-and-test failure on this side.
    #[test]
    fn every_run_event_keeps_its_wire_shape() {
        let v = |e: RunEvent| serde_json::to_value(e).unwrap();

        assert_eq!(
            v(RunEvent::Started {
                run_id: "r1".into()
            }),
            json!({ "type": "started", "run_id": "r1" })
        );
        assert_eq!(
            v(RunEvent::TextDelta { text: "hi".into() }),
            json!({ "type": "text_delta", "text": "hi" })
        );
        assert_eq!(
            v(RunEvent::ThinkingDelta { text: "hmm".into() }),
            json!({ "type": "thinking_delta", "text": "hmm" })
        );
        assert_eq!(
            v(RunEvent::ToolCallStarted {
                id: "c1".into(),
                name: "shell".into(),
                arguments: Value::from(json!({ "cmd": "ls" })),
            }),
            json!({
                "type": "tool_call_started",
                "id": "c1",
                "name": "shell",
                "arguments": { "cmd": "ls" },
            })
        );
        assert_eq!(
            v(RunEvent::AwaitingApproval {
                id: "c1".into(),
                name: "shell".into(),
                arguments: Value::from(json!({ "cmd": "ls" })),
            }),
            json!({
                "type": "awaiting_approval",
                "id": "c1",
                "name": "shell",
                "arguments": { "cmd": "ls" },
            })
        );

        // `message` is ailoy's own `Message` and is pinned by ailoy's tests, not here —
        // what this asserts is the envelope the webview indexes into.
        let message = v(RunEvent::Message {
            seq: 3,
            depth: 0,
            source_agent: None,
            message: Message::new(Role::Assistant).with_contents([Part::text("hello")]),
            usage: None,
        });
        assert_eq!(message["type"], json!("message"));
        assert_eq!(message["seq"], json!(3));
        assert_eq!(message["depth"], json!(0));
        assert_eq!(message["source_agent"], json!(null));
        assert_eq!(message["usage"], json!(null));
        assert!(message["message"].is_object(), "{message}");

        let usage = v(RunEvent::Usage {
            usage: Some(TokenUsage {
                input_tokens: 7,
                output_tokens: 2,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            }),
            rate_limit: None,
            context_used: Some(7),
            context_limit: Some(200_000),
        });
        assert_eq!(usage["type"], json!("usage"));
        assert_eq!(usage["rate_limit"], json!(null));
        assert_eq!(usage["context_used"], json!(7));
        assert_eq!(usage["context_limit"], json!(200_000));
        assert_eq!(usage["usage"]["input_tokens"], json!(7));
        assert_eq!(usage["usage"]["output_tokens"], json!(2));

        // The two unit variants carry the tag and nothing else.
        assert_eq!(v(RunEvent::Done), json!({ "type": "done" }));
        assert_eq!(v(RunEvent::Cancelled), json!({ "type": "cancelled" }));

        assert_eq!(
            v(RunEvent::Error {
                kind: "model".into(),
                message: "model request failed (HTTP 429): slow down".into(),
                status: Some(429),
                retryable: true,
            }),
            json!({
                "type": "error",
                "kind": "model",
                "message": "model request failed (HTTP 429): slow down",
                "status": 429,
                "retryable": true,
            })
        );
        // A non-model failure still carries both fields, as `null`/`false` — the webview
        // reads them unconditionally rather than testing for their presence.
        assert_eq!(
            v(RunEvent::Error {
                kind: "tool".into(),
                message: "tool execution failed".into(),
                status: None,
                retryable: false,
            }),
            json!({
                "type": "error",
                "kind": "tool",
                "message": "tool execution failed",
                "status": null,
                "retryable": false,
            })
        );
    }
}
