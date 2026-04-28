use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{Message, Part, Role},
};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContextManager {
    /// Triggers truncation when input_tokens from the previous API call exceeds this value.
    pub max_input_tokens: u64,
    /// Number of recent user turns to preserve after truncation (system message is always preserved separately).
    pub preserve_recent_turns: usize,
}

impl Default for ContextManager {
    fn default() -> Self {
        Self {
            max_input_tokens: 30_000,
            preserve_recent_turns: 3,
        }
    }
}

/// Truncate the conversation history to reduce context size.
///
/// ## Algorithm
///
/// 1. If `history[0]` is `Role::System`, always preserve it (never dropped).
/// 2. Walk backwards from the end of history, skipping `System` messages, and
///    count completed user-assistant turns.  Once `preserve_recent_turns` pairs
///    have been counted, the index of the oldest `User` message in that window
///    becomes the **preserve boundary** — everything at or after that index is
///    left untouched.
/// 3. For each `Role::Tool` message *before* the preserve boundary, replace its
///    contents with a `"[context truncated]"` placeholder **while keeping the
///    message's `id` intact**.  Anthropic's API returns HTTP 400 if a tool-use
///    `id` that appears in an assistant message has no matching tool-result, so
///    the id must never be discarded.
///
/// Note: Full group-level dropping (removing the oldest user + assistant + tool
/// triplet entirely) is left for a future iteration; it requires a reliable
/// post-truncation token estimate that is not yet available here.  For now,
/// placeholder replacement alone is sufficient to keep the context window
/// manageable for most workloads.
pub(crate) fn truncate_history(history: &mut Vec<Message>, spec: &ContextManager) {
    if history.is_empty() {
        return;
    }

    // ── Locate preserve boundary ───────────────────────────────────────────────
    let preserve_from = find_preserve_boundary(history, spec.preserve_recent_turns);

    let start_idx = if history[0].role == Role::System {
        1
    } else {
        0
    };

    // ── Replace Tool messages outside the preserve window with placeholders ────
    for i in start_idx..preserve_from {
        if history[i].role == Role::Tool {
            let original_id = history[i].id.clone();
            let placeholder = Message::new(Role::Tool).with_contents([Part::value(Value::string(
                "[context truncated]".to_string(),
            ))]);
            history[i] = if let Some(id) = original_id {
                placeholder.with_id(id)
            } else {
                placeholder
            };
        }
    }
}

/// Find the index from which messages should be preserved.
///
/// Scans backwards through `history`, skipping `System` messages, and counts
/// `Assistant` messages (each together with its preceding `User` message counts
/// as one "turn").  Returns the index of the `User` message that begins the
/// `preserve_recent_turns`-th turn from the end, or `0` when there are fewer
/// turns than requested (meaning: preserve everything).
fn find_preserve_boundary(history: &[Message], preserve_recent_turns: usize) -> usize {
    if preserve_recent_turns == 0 {
        return history.len();
    }

    let mut turns_found = 0usize;
    let mut i = history.len();

    while i > 0 {
        i -= 1;
        if history[i].role == Role::System {
            continue;
        }
        if history[i].role == Role::Assistant {
            turns_found += 1;
            if turns_found >= preserve_recent_turns {
                // Walk back to find the User message that opened this turn.
                let mut j = i;
                while j > 0 {
                    j -= 1;
                    if history[j].role == Role::System {
                        continue;
                    }
                    if history[j].role == Role::User {
                        return j;
                    }
                }
                return 0;
            }
        }
    }

    // Fewer turns than requested — preserve everything.
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Part, Role};

    fn sys() -> Message {
        Message::new(Role::System).with_contents([Part::text("system")])
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User).with_contents([Part::text(text)])
    }

    fn asst(text: &str) -> Message {
        Message::new(Role::Assistant).with_contents([Part::text(text)])
    }

    fn tool_result(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::value(Value::string("ok".to_string()))])
    }

    fn tool_call_asst(call_id: &str, tool_name: &str) -> Message {
        Message::new(Role::Assistant).with_tool_calls([Part::function(
            call_id,
            tool_name,
            crate::datatype::Value::null(),
        )])
    }

    #[test]
    fn test_preserve_recent_turns_boundary() {
        // history: sys, u1, a1, u2, a2, u3, a3
        // preserve_recent_turns = 2 → preserve from u2 (index 3) onwards
        let history = vec![
            sys(),
            user("u1"),
            asst("a1"),
            user("u2"),
            asst("a2"),
            user("u3"),
            asst("a3"),
        ];
        let boundary = find_preserve_boundary(&history, 2);
        assert_eq!(boundary, 3, "preserve boundary should be at u2 (index 3)");
    }

    #[test]
    fn test_no_change_when_all_within_preserve_window() {
        let mut history = vec![sys(), user("u1"), asst("a1"), user("u2"), asst("a2")];
        let spec = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 10,
        };
        let original_len = history.len();
        truncate_history(&mut history, &spec);
        assert_eq!(history.len(), original_len, "nothing should change");
    }

    #[test]
    fn test_tool_placeholder_replacement() {
        // history: sys, u1, tool_call_asst(call_1), tool_result(call_1), u2, a2
        // preserve_recent_turns = 1 → preserve (u2, a2) from index 4 onwards.
        // tool_result at index 3 is outside the preserve window → becomes placeholder.
        let mut history = vec![
            sys(),
            user("u1"),
            tool_call_asst("call_1", "my_tool"),
            tool_result("call_1"),
            user("u2"),
            asst("a2"),
        ];
        let spec = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        truncate_history(&mut history, &spec);

        // The Tool message must still be present (as a placeholder, not removed).
        let tool_msg = history.iter().find(|m| m.role == Role::Tool);
        assert!(
            tool_msg.is_some(),
            "Tool message must still exist as placeholder"
        );
        let tool_msg = tool_msg.unwrap();
        assert_eq!(
            tool_msg.id.as_deref(),
            Some("call_1"),
            "tool_use_id must be preserved to avoid Anthropic 400 errors"
        );
        let content = tool_msg
            .contents
            .first()
            .expect("placeholder must have content");
        let val = content
            .as_value()
            .expect("placeholder content must be a Value part");
        assert_eq!(
            val.as_str(),
            Some("[context truncated]"),
            "placeholder content must be '[context truncated]'"
        );
    }

    #[test]
    fn test_system_message_never_dropped() {
        let mut history = vec![
            sys(),
            user("u1"),
            asst("a1"),
            user("u2"),
            asst("a2"),
            user("u3"),
            asst("a3"),
        ];
        let spec = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        truncate_history(&mut history, &spec);
        assert_eq!(
            history[0].role,
            Role::System,
            "System message must always remain at index 0"
        );
        assert_eq!(history.len(), 7, "no messages should be dropped");
    }

    #[test]
    fn test_preserved_messages_untouched() {
        // Only messages outside the preserve window should be replaced.
        // The tool result inside the preserve window must keep its original content.
        let mut history = vec![
            sys(),
            user("u1"),
            asst("a1"),
            user("u2"),
            tool_call_asst("call_2", "tool_b"),
            tool_result("call_2"),
            asst("a2"),
        ];
        // preserve_recent_turns = 1 → boundary is at u2 (index 3).
        // tool_result("call_2") is at index 5 which is >= 3 → preserved.
        let spec = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        truncate_history(&mut history, &spec);
        let tool_msg = history
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("tool result must still be present");
        let val = tool_msg
            .contents
            .first()
            .and_then(|p| p.as_value())
            .expect("tool result content must be a Value part");
        assert_eq!(
            val.as_str(),
            Some("ok"),
            "tool result inside preserve window must not be replaced"
        );
    }
}
