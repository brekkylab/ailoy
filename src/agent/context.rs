use serde::{Deserialize, Serialize};

use crate::message::{Message, Part, Role};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(default)]
pub struct ContextManager {
    /// Triggers truncation when input_tokens from the previous API call exceeds this value.
    /// `0` turns truncation off.
    pub max_input_tokens: u64,
    /// Number of recent user turns to preserve after truncation (system message is always preserved separately).
    pub preserve_recent_turns: usize,
    /// Images in tool results the history may hold at the start of a turn; over it, the
    /// oldest become a placeholder. `0` turns the cap off.
    pub max_images: usize,
    /// Images in tool results left once the cap is passed.
    pub preserve_recent_images: usize,
}

impl Default for ContextManager {
    fn default() -> Self {
        Self {
            max_input_tokens: 30_000,
            preserve_recent_turns: 3,
            max_images: 30,
            preserve_recent_images: 10,
        }
    }
}

/// What stands in for an image [`cap_images`](ContextManager::cap_images) replaced.
pub(super) const IMAGE_PLACEHOLDER: &str = "[image from an earlier turn removed]";

impl ContextManager {
    /// Truncate the conversation history to reduce context size.
    ///
    /// A leading system message is always kept, as is everything from the
    /// [preserve boundary](find_preserve_boundary) on. Each earlier `Role::Tool` message
    /// has its contents replaced by `"[context truncated]"` but keeps its `id`: Anthropic
    /// returns HTTP 400 for a tool-use id with no matching tool result.
    ///
    /// Whole turns are never dropped; that would need a post-truncation token estimate.
    pub(super) fn truncate_history(&self, history: &mut [Message]) {
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

    /// Over [`max_images`](Self::max_images) images in the tool results of `history`, replace
    /// the oldest by [`IMAGE_PLACEHOLDER`] until
    /// [`preserve_recent_images`](Self::preserve_recent_images) remain: many at once, so the prompt
    /// cache breaks once in several turns rather than every turn. Images a person attached are
    /// left alone.
    ///
    /// Only for the start of a turn: Claude rejects a history altered before a thinking block
    /// it is handed back, and at turn start none follows, since `marshal_messages` (Claude)
    /// and `conversation` (Bedrock) replay thinking only after the last user message.
    pub(super) fn cap_images(&self, history: &mut [Message]) {
        if self.max_images == 0 {
            return;
        }
        let images: Vec<(usize, usize)> = history
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role == Role::Tool)
            .flat_map(|(i, m)| {
                let parts = m.contents.iter().enumerate();
                parts
                    .filter(|(_, p)| p.is_image())
                    .map(move |(j, _)| (i, j))
            })
            .collect();
        if images.len() <= self.max_images {
            return;
        }
        let replaced = images.len().saturating_sub(self.preserve_recent_images);
        for (i, j) in images.into_iter().take(replaced) {
            history[i].contents[j] = Part::text(IMAGE_PLACEHOLDER);
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

    fn image() -> Part {
        Part::image_embedded("image/png", vec![0u8; 4].into()).unwrap()
    }

    fn shot(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::text("the page"), image()])
    }

    fn user_with_image(text: &str) -> Message {
        Message::new(Role::User).with_contents([Part::text(text), image()])
    }

    fn images_in(history: &[Message]) -> usize {
        history
            .iter()
            .flat_map(|m| &m.contents)
            .filter(|p| p.is_image())
            .count()
    }

    fn with_images(max_images: usize, preserve_recent_images: usize) -> ContextManager {
        ContextManager {
            max_images,
            preserve_recent_images,
            ..Default::default()
        }
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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

    #[test]
    fn test_over_the_cap_the_oldest_screenshots_go_until_the_preserved_remain() {
        // Five screenshots over a cap of four: the last two stay.
        let mut history = vec![sys(), user("u1")];
        history.extend((1..=5).map(|n| shot(&format!("c{n}"))));
        history.push(asst("a1"));
        with_images(4, 2).cap_images(&mut history);

        assert_eq!(history[2].contents[1].as_text(), Some(IMAGE_PLACEHOLDER));
        assert!(history[2..=4].iter().all(|m| m.contents[1].is_text()));
        assert!(history[5..=6].iter().all(|m| m.contents[1].is_image()));
        // The note beside each screenshot stays.
        assert_eq!(history[2].contents[0].as_text(), Some("the page"));
        assert_eq!(history[2].id.as_deref(), Some("c1"));
    }

    #[test]
    fn test_at_the_cap_nothing_goes() {
        let mut history = vec![sys(), user("u1")];
        history.extend((1..=4).map(|n| shot(&format!("c{n}"))));
        with_images(4, 2).cap_images(&mut history);
        assert_eq!(images_in(&history), 4);
    }

    #[test]
    fn test_a_persons_images_are_neither_counted_nor_replaced() {
        // Three of a person's images and two screenshots are under a cap of two.
        let mut history = vec![
            sys(),
            user_with_image("u1"),
            user_with_image("u2"),
            shot("c1"),
            user_with_image("u3"),
            shot("c2"),
            asst("a1"),
        ];
        with_images(2, 1).cap_images(&mut history);
        assert_eq!(images_in(&history), 5);

        // A third screenshot passes it: the two oldest go, the person's three stay.
        history.push(shot("c3"));
        with_images(2, 1).cap_images(&mut history);
        assert_eq!(images_in(&history), 4);
        assert!(history[3].contents[1].is_text());
        assert!(history[5].contents[1].is_text());
        assert!(history[7].contents[1].is_image());
        assert!(
            history
                .iter()
                .filter(|m| m.role == Role::User)
                .all(|m| m.contents[1].is_image())
        );
    }

    #[test]
    fn test_a_cap_of_zero_is_off_and_keeping_none_clears_them() {
        let mut history = vec![sys(), user("u1")];
        history.extend((1..=12).map(|n| shot(&format!("c{n}"))));

        with_images(0, 0).cap_images(&mut history);
        assert_eq!(images_in(&history), 12);

        with_images(3, 0).cap_images(&mut history);
        assert_eq!(images_in(&history), 0);
    }

    #[test]
    fn test_an_odd_cap_leaves_exactly_the_preserved() {
        let mut history = vec![sys(), user("u1")];
        history.extend((1..=4).map(|n| shot(&format!("c{n}"))));
        with_images(3, 1).cap_images(&mut history);
        assert_eq!(images_in(&history), 1);
        assert!(history[5].contents[1].is_image());
    }
}
