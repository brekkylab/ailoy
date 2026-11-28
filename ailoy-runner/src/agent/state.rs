use crate::{
    agent::AgentState,
    message::{Part, Role},
};

use super::TruncationPolicy;

/// What stands in for an image [`truncate_history`] replaced.
pub(super) const IMAGE_PLACEHOLDER: &str = "[image from an earlier turn removed]";

/// What stands in for a tool result's contents [`truncate_history`] replaced.
pub(super) const TOOL_RESULT_PLACEHOLDER: &str = "[context truncated]";

/// Replace old images and tool results in the history of `state` by placeholders, as `policy`
/// says. No message is removed, and a tool result keeps its id, since APIs such as Claude's
/// refuse a call without its result. Images a person attached are left alone.
///
/// Only for the start of a turn, with the query already in the history: Claude refuses a
/// thinking block handed back once anything before it has changed, and at turn start none
/// follows, since the backends replay thinking only after the last user message.
pub(super) fn truncate_history(state: &mut AgentState, policy: &TruncationPolicy) {
    let history = &mut state.history;
    // Past `max_images` images in tool results, the oldest go until
    // `preserve_recent_images` remain: many at once, so the prompt cache breaks once in
    // several turns rather than every turn.
    if policy.max_images > 0 {
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
        if images.len() > policy.max_images {
            let replaced = images.len().saturating_sub(policy.preserve_recent_images);
            for (i, j) in images.into_iter().take(replaced) {
                history[i].contents[j] = Part::text(IMAGE_PLACEHOLDER);
            }
        }
    }

    // Once the last call read past `max_input_tokens`, the tool results before the
    // latest `preserve_recent_turns` user turns lose their contents. A turn counts from
    // its user message, since one turn may hold many assistant messages; with fewer turns
    // than that, nothing changes.
    if policy.max_input_tokens == 0 || state.context_tokens.unwrap_or(0) <= policy.max_input_tokens
    {
        return;
    }
    let users = history
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User);
    let Some((preserve_from, _)) = users.rev().nth(policy.preserve_recent_turns.max(1) - 1) else {
        return;
    };
    for message in &mut history[..preserve_from] {
        if message.role == Role::Tool {
            message.contents = vec![Part::text(TOOL_RESULT_PLACEHOLDER)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datatype::Value;
    use crate::message::Message;

    fn sys() -> Message {
        Message::new(Role::System).with_contents([Part::text("system")])
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User).with_contents([Part::text(text)])
    }

    fn asst(text: &str) -> Message {
        Message::new(Role::Assistant).with_contents([Part::text(text)])
    }

    fn call(id: &str) -> Message {
        Message::new(Role::Assistant).with_tool_calls([Part::function(id, "echo", Value::null())])
    }

    fn result(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::text("ok")])
    }

    fn image() -> Part {
        Part::image_embedded("image/png", vec![0u8; 4].into()).unwrap()
    }

    fn shot(id: &str) -> Message {
        Message::new(Role::Tool)
            .with_id(id)
            .with_contents([Part::text("the page"), image()])
    }

    fn images_in(history: &[Message]) -> usize {
        history
            .iter()
            .flat_map(|m| &m.contents)
            .filter(|p| p.is_image())
            .count()
    }

    /// Whether each tool result in `history` lost its contents.
    fn truncated(history: &[Message]) -> Vec<bool> {
        history
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.contents[0].as_text() == Some(TOOL_RESULT_PLACEHOLDER))
            .collect()
    }

    /// The tool results of the turns before the latest `preserve_recent_turns` lose their
    /// contents and keep their ids; no message goes. Turns count from user messages, so a
    /// turn of several assistant messages is one.
    #[test]
    fn test_old_tool_results_are_truncated_past_the_token_limit() {
        let history = vec![
            sys(),
            user("u1"),
            call("c1"),
            result("c1"),
            asst("a1"),
            user("u2"),
            call("c2"),
            result("c2"),
            asst("a2"),
            user("u3"),
        ];
        let policy = |preserve_recent_turns| TruncationPolicy {
            max_input_tokens: 100,
            preserve_recent_turns,
            ..Default::default()
        };

        let read = |context_tokens| AgentState {
            context_tokens,
            ..AgentState::new().with_history(history.clone())
        };

        let mut state = read(Some(200));
        truncate_history(&mut state, &policy(2));
        let cut = &state.history;
        assert_eq!(truncated(cut), [true, false]);
        assert_eq!(cut.len(), history.len());
        assert_eq!(cut[0].role, Role::System);
        assert_eq!(cut[3].id.as_deref(), Some("c1"));

        // The turn just begun is always kept, so `0` acts as `1`.
        for preserve in [0, 1] {
            let mut state = read(Some(200));
            truncate_history(&mut state, &policy(preserve));
            assert_eq!(
                truncated(&state.history),
                [true, true],
                "preserve: {preserve}"
            );
        }

        // Fewer turns than preserved, at the limit, or without a limit: nothing changes.
        for (preserve, max_input_tokens, tokens) in [(3, 100, 200), (1, 100, 100), (1, 0, 200)] {
            let mut state = read(Some(tokens));
            truncate_history(
                &mut state,
                &TruncationPolicy {
                    max_input_tokens,
                    preserve_recent_turns: preserve,
                    ..Default::default()
                },
            );
            assert_eq!(
                truncated(&state.history),
                [false, false],
                "{preserve} {max_input_tokens} {tokens}"
            );
        }
    }

    /// Past the cap the oldest images in tool results go until the preserved remain; at it,
    /// none does. The text beside each and a person's images stay.
    #[test]
    fn test_images_past_the_cap_are_replaced_oldest_first() {
        let images = |max_images, preserve_recent_images| TruncationPolicy {
            max_images,
            preserve_recent_images,
            ..Default::default()
        };
        let mut state = AgentState::new().with_history([
            sys(),
            Message::new(Role::User).with_contents([Part::text("u1"), image()]),
        ]);
        state
            .history
            .extend((1..=4).map(|n| shot(&format!("c{n}"))));

        truncate_history(&mut state, &images(4, 2));
        assert_eq!(images_in(&state.history), 5);

        state.history.push(shot("c5"));
        truncate_history(&mut state, &images(4, 2));
        assert_eq!(images_in(&state.history), 3);
        assert!(state.history[1].contents[1].is_image());
        assert!(
            state.history[2..=4]
                .iter()
                .all(|m| m.contents[1].as_text() == Some(IMAGE_PLACEHOLDER))
        );
        assert!(
            state.history[5..=6]
                .iter()
                .all(|m| m.contents[1].is_image())
        );
        assert_eq!(state.history[2].contents[0].as_text(), Some("the page"));
        assert_eq!(state.history[2].id.as_deref(), Some("c1"));

        // A cap of `0` is off; preserving none clears them all.
        truncate_history(&mut state, &images(0, 0));
        assert_eq!(images_in(&state.history), 3);
        truncate_history(&mut state, &images(1, 0));
        assert_eq!(images_in(&state.history), 1);
    }
}
