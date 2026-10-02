use serde::{Deserialize, Serialize};

use crate::message::{Message, Part, Role};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ContextManager {
    /// Triggers truncation when the previous API call's whole prompt exceeded this value
    /// — `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, not
    /// `input_tokens` alone, which counts only the uncached remainder.
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

impl ContextManager {
    /// Truncate the conversation history to reduce context size.
    ///
    /// A leading system message is always kept, as is everything from the
    /// [preserve boundary](find_preserve_boundary) on. Each earlier `Role::Tool` message
    /// has its contents replaced by `"[context truncated]"` but keeps its `id`: Anthropic
    /// returns HTTP 400 for a tool-use id with no matching tool result.
    ///
    /// Whole turns are never dropped; that would need a post-truncation token estimate.
    pub(crate) fn truncate_history(&self, history: &mut [Message]) {
        if history.is_empty() {
            return;
        }

        // ── Locate preserve boundary ───────────────────────────────────────────────
        let preserve_from = find_preserve_boundary(history, self.preserve_recent_turns);

        let start_idx = if history[0].role == Role::System {
            1
        } else {
            0
        };

        // `preserve_from == 0` truncates nothing; otherwise the oldest preserved User turn
        // is at index >= 1, so start_idx <= preserve_from.
        debug_assert!(
            preserve_from == 0 || start_idx <= preserve_from,
            "start_idx ({start_idx}) > preserve_from ({preserve_from}): \
             truncation window would overlap the system message"
        );

        // ── Replace Tool messages outside the preserve window with placeholders ────
        for msg in history.iter_mut().take(preserve_from).skip(start_idx) {
            if msg.role == Role::Tool {
                let original_id = msg.id.clone();
                let placeholder =
                    Message::new(Role::Tool).with_contents([Part::text("[context truncated]")]);
                *msg = if let Some(id) = original_id {
                    placeholder.with_id(id)
                } else {
                    placeholder
                };
            }
        }
    }
}

/// Find the index from which messages should be preserved.
///
/// Returns the index of the `preserve_recent_turns`-th `User` message from the end, or
/// `0` (preserve everything) when there are fewer turns.
///
/// Counts `User` rather than `Assistant` messages because one user input may produce
/// several assistant messages (`asst(tool_call) → tool → asst(text)`).
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
        if history[i].role == Role::User {
            turns_found += 1;
            if turns_found >= preserve_recent_turns {
                return i;
            }
        }
    }

    // Fewer turns than requested — preserve everything.
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        datatype::Value,
        message::{Message, Part, Role},
    };

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
            Value::null(),
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
        let cm = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 10,
        };
        let original_len = history.len();
        cm.truncate_history(&mut history);
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
        let cm = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        cm.truncate_history(&mut history);

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
            .as_text()
            .expect("placeholder content must be a Value part");
        assert_eq!(
            val, "[context truncated]",
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
        let cm = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        cm.truncate_history(&mut history);
        assert_eq!(
            history[0].role,
            Role::System,
            "System message must always remain at index 0"
        );
        assert_eq!(history.len(), 7, "no messages should be dropped");
    }

    #[test]
    fn test_preserve_counts_user_messages_not_assistant() {
        // u2's turn spans asst(tool_call) → tool → asst("a2"); counting assistants would put
        // the boundary inside it.
        // history: sys(0), u1(1), asst("a1")(2), u2(3), asst(tool_call)(4), tool(5), asst("a2")(6)
        let history = vec![
            sys(),
            user("u1"),
            asst("a1"),
            user("u2"),
            tool_call_asst("call_1", "my_tool"),
            tool_result("call_1"),
            asst("a2"),
        ];
        let boundary = find_preserve_boundary(&history, 2);
        assert_eq!(
            boundary, 1,
            "preserve_recent_turns=2 must keep 2 user turns, landing at u1 (index 1)"
        );
    }

    #[test]
    fn test_preserved_messages_untouched() {
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
        let cm = ContextManager {
            max_input_tokens: 30_000,
            preserve_recent_turns: 1,
        };
        cm.truncate_history(&mut history);
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
