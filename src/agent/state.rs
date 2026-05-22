use std::sync::Arc;

use crate::{
    message::{Message, Part, Role},
    runenv::RunEnv,
    skill::{self, SkillMeta},
};

/// Conversation transcript paired with the policy that bounds it.
///
/// The [`messages`](Self::messages) vector grows turn-by-turn during an agent
/// run.  When the previous model call's [`last_input_tokens`](Self::last_input_tokens)
/// exceeds [`max_input_tokens`](Self::max_input_tokens), tool results outside
/// the [`preserve_recent_turns`](Self::preserve_recent_turns) window are
/// reduced to `"[context truncated]"` placeholders.
pub struct AgentHistory {
    pub messages: Vec<Message>,

    /// Triggers truncation when input_tokens from the previous API call exceeds this value.
    pub max_input_tokens: u64,

    /// Number of recent user turns to preserve after truncation (system message is always preserved separately).
    pub preserve_recent_turns: usize,

    /// Token count from the most recent model API call; used to decide when to truncate history.
    pub last_input_tokens: Option<u64>,
}

impl Default for AgentHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_input_tokens: 30_000,
            preserve_recent_turns: 3,
            last_input_tokens: None,
        }
    }

    /// Apply the configured truncation policy when the most recent input-token
    /// count exceeded [`max_input_tokens`](Self::max_input_tokens).
    pub fn truncate_if_needed(&mut self) {
        if self.last_input_tokens.unwrap_or(0) > self.max_input_tokens {
            self.truncate();
        }
    }

    /// Truncate the conversation history to reduce context size.
    ///
    /// ## Algorithm
    ///
    /// 1. If `messages[0]` is `Role::System`, always preserve it (never dropped).
    /// 2. Walk backwards from the end, skipping `System` messages, and count
    ///    `User` messages.  Once [`preserve_recent_turns`](Self::preserve_recent_turns)
    ///    `User` messages have been counted, the oldest of them becomes the
    ///    **preserve boundary** — everything at or after that index is left
    ///    untouched.  Counting `User` messages (not `Assistant` messages)
    ///    correctly handles tool-use sessions where a single user input may
    ///    expand into multiple assistant messages.
    /// 3. For each `Role::Tool` message *before* the preserve boundary, replace
    ///    its contents with a `"[context truncated]"` placeholder **while
    ///    keeping the message's `id` intact**.  Anthropic's API returns HTTP
    ///    400 if a tool-use `id` that appears in an assistant message has no
    ///    matching tool-result, so the id must never be discarded.
    ///
    /// Note: Full group-level dropping (removing the oldest user + assistant +
    /// tool triplet entirely) is left for a future iteration; it requires a
    /// reliable post-truncation token estimate that is not yet available here.
    /// For now, placeholder replacement alone is sufficient to keep the context
    /// window manageable for most workloads.
    pub(crate) fn truncate(&mut self) {
        if self.messages.is_empty() {
            return;
        }

        let preserve_from = self.find_preserve_boundary();

        let start_idx = if self.messages[0].role == Role::System {
            1
        } else {
            0
        };

        // `preserve_from = 0` means "fewer turns than requested — preserve everything":
        // `.take(0).skip(start_idx)` is always empty, so nothing is truncated.
        // When `preserve_from > 0` the System message always lands at index 0 and
        // the oldest preserved User turn is at index >= 1, so start_idx <= preserve_from.
        debug_assert!(
            preserve_from == 0 || start_idx <= preserve_from,
            "start_idx ({start_idx}) > preserve_from ({preserve_from}): \
             truncation window would overlap the system message"
        );

        for msg in self.messages.iter_mut().take(preserve_from).skip(start_idx) {
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

    /// Find the index from which messages should be preserved.
    ///
    /// Scans backwards through `messages`, skipping `System` messages, and
    /// counts `User` messages.  Returns the index of the `User` message that
    /// is [`preserve_recent_turns`](Self::preserve_recent_turns)-th from the
    /// end, or `0` when there are fewer turns than requested (meaning:
    /// preserve everything).
    ///
    /// Counting `User` messages (rather than `Assistant` messages) correctly
    /// handles tool-use sessions where one user input may produce multiple
    /// assistant messages (`asst(tool_call) → tool → asst(text)`).
    fn find_preserve_boundary(&self) -> usize {
        if self.preserve_recent_turns == 0 {
            return self.messages.len();
        }

        let mut turns_found = 0usize;
        let mut i = self.messages.len();

        while i > 0 {
            i -= 1;
            if self.messages[i].role == Role::System {
                continue;
            }
            if self.messages[i].role == Role::User {
                turns_found += 1;
                if turns_found >= self.preserve_recent_turns {
                    return i;
                }
            }
        }

        // Fewer turns than requested — preserve everything.
        0
    }
}

pub struct AgentState {
    pub history: AgentHistory,

    pub runenv: Arc<RunEnv>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: AgentHistory::new(),
            runenv: Arc::new(RunEnv::local()),
        }
    }

    pub fn messages(mut self, messages: impl IntoIterator<Item = Message>) -> Self {
        self.history.messages = messages.into_iter().collect();
        self
    }

    pub fn runenv(mut self, runenv: impl Into<Arc<RunEnv>>) -> Self {
        self.runenv = runenv.into();
        self
    }

    /// Truncate the conversation history to reduce context size.
    pub fn truncate_messages(&mut self) {
        self.history.truncate();
    }

    /// List skills available under `path` inside the agent's [`RunEnv`].
    ///
    /// Recursively scans for `SKILL.md` files and parses their `name` /
    /// `description` frontmatter.  Works against any backend (local FS or
    /// sandbox) since the walk is performed through the runenv's shell.
    pub async fn list_skills(&self, path: &std::path::Path) -> anyhow::Result<Vec<SkillMeta>> {
        let handle = self.runenv.get().await?;
        skill::list_skills(handle, path).await
    }

    /// Load a single skill from the directory `path` inside the agent's
    /// [`RunEnv`].  Returns `Ok(None)` when `path/SKILL.md` does not exist.
    pub async fn get_skill(&self, path: &std::path::Path) -> anyhow::Result<Option<SkillMeta>> {
        let handle = self.runenv.get().await?;
        skill::get_skill(handle, path).await
    }
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

    fn history_of(messages: Vec<Message>, preserve_recent_turns: usize) -> AgentHistory {
        AgentHistory {
            messages,
            max_input_tokens: 30_000,
            preserve_recent_turns,
            last_input_tokens: None,
        }
    }

    #[test]
    fn test_preserve_recent_turns_boundary() {
        // history: sys, u1, a1, u2, a2, u3, a3
        // preserve_recent_turns = 2 → preserve from u2 (index 3) onwards
        let history = history_of(
            vec![
                sys(),
                user("u1"),
                asst("a1"),
                user("u2"),
                asst("a2"),
                user("u3"),
                asst("a3"),
            ],
            2,
        );
        assert_eq!(
            history.find_preserve_boundary(),
            3,
            "preserve boundary should be at u2 (index 3)"
        );
    }

    #[test]
    fn test_no_change_when_all_within_preserve_window() {
        let mut history = history_of(
            vec![sys(), user("u1"), asst("a1"), user("u2"), asst("a2")],
            10,
        );
        let original_len = history.messages.len();
        history.truncate();
        assert_eq!(
            history.messages.len(),
            original_len,
            "nothing should change"
        );
    }

    #[test]
    fn test_tool_placeholder_replacement() {
        // history: sys, u1, tool_call_asst(call_1), tool_result(call_1), u2, a2
        // preserve_recent_turns = 1 → preserve (u2, a2) from index 4 onwards.
        // tool_result at index 3 is outside the preserve window → becomes placeholder.
        let mut history = history_of(
            vec![
                sys(),
                user("u1"),
                tool_call_asst("call_1", "my_tool"),
                tool_result("call_1"),
                user("u2"),
                asst("a2"),
            ],
            1,
        );
        history.truncate();

        // The Tool message must still be present (as a placeholder, not removed).
        let tool_msg = history.messages.iter().find(|m| m.role == Role::Tool);
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
        let mut history = history_of(
            vec![
                sys(),
                user("u1"),
                asst("a1"),
                user("u2"),
                asst("a2"),
                user("u3"),
                asst("a3"),
            ],
            1,
        );
        history.truncate();
        assert_eq!(
            history.messages[0].role,
            Role::System,
            "System message must always remain at index 0"
        );
        assert_eq!(history.messages.len(), 7, "no messages should be dropped");
    }

    #[test]
    fn test_preserve_counts_user_messages_not_assistant() {
        // A single user turn may produce multiple assistant messages in tool-use:
        //   u2 → asst(tool_call) → tool → asst("a2")
        // preserve_recent_turns = 2 must preserve both u1 and u2's full interactions,
        // not just u2's (which would happen if assistant messages were counted).
        //
        // history: sys(0), u1(1), asst("a1")(2), u2(3), asst(tool_call)(4), tool(5), asst("a2")(6)
        // Expected boundary: u1 at index 1  (2 user turns preserved)
        let history = history_of(
            vec![
                sys(),
                user("u1"),
                asst("a1"),
                user("u2"),
                tool_call_asst("call_1", "my_tool"),
                tool_result("call_1"),
                asst("a2"),
            ],
            2,
        );
        assert_eq!(
            history.find_preserve_boundary(),
            1,
            "preserve_recent_turns=2 must keep 2 user turns, landing at u1 (index 1)"
        );
    }

    #[test]
    fn test_preserved_messages_untouched() {
        // Only messages outside the preserve window should be replaced.
        // The tool result inside the preserve window must keep its original content.
        let mut history = history_of(
            vec![
                sys(),
                user("u1"),
                asst("a1"),
                user("u2"),
                tool_call_asst("call_2", "tool_b"),
                tool_result("call_2"),
                asst("a2"),
            ],
            1,
        );
        // preserve_recent_turns = 1 → boundary is at u2 (index 3).
        // tool_result("call_2") is at index 5 which is >= 3 → preserved.
        history.truncate();
        let tool_msg = history
            .messages
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
