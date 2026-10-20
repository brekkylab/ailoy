use crate::{
    datatype::Value,
    message::{Message, MessageDelta, Part, PartDelta},
};

/// Per-part size limit for tool results, past which the middle is cut out.
pub(crate) const MAX_TOOL_RESULT_CHARS: usize = 30_000;

/// Clamp every part of a tool result, so large payloads (e.g. web pages) don't pile up in
/// the history. An oversized [`Part::Value`] is measured as JSON and replaced by the cut
/// JSON as a string.
pub(crate) fn cap_tool_result(mut message: Message) -> Message {
    for part in &mut message.contents {
        match part {
            Part::Value { value } => cap_value(value),
            Part::Text { text } => cap_text(text),
            _ => {}
        }
    }
    message
}

/// Clamp every part of a piece of a tool result, as [`cap_tool_result`] does the whole.
pub(crate) fn cap_tool_delta(mut delta: MessageDelta) -> MessageDelta {
    for part in &mut delta.contents {
        match part {
            PartDelta::Value { value } => cap_value(value),
            PartDelta::Text { text } => cap_text(text),
            _ => {}
        }
    }
    delta
}

fn cap_value(value: &mut Value) {
    let json = serde_json::to_string(value).unwrap_or_default();
    if json.len() > MAX_TOOL_RESULT_CHARS {
        *value = Value::string(middle_truncate(&json, MAX_TOOL_RESULT_CHARS));
    }
}

fn cap_text(text: &mut String) {
    if text.len() > MAX_TOOL_RESULT_CHARS {
        *text = middle_truncate(text, MAX_TOOL_RESULT_CHARS);
    }
}

/// Keep `max_chars` characters of `s`, half from each end, around a note of what was cut.
pub(crate) fn middle_truncate(s: &str, max_chars: usize) -> String {
    let len = s.chars().count();
    if len <= max_chars {
        return s.to_owned();
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let head_str: String = s.chars().take(head).collect();
    let tail_str: String = s.chars().skip(len - tail).collect();
    format!(
        "{head_str}\n\n... [{} characters omitted] ...\n\n{tail_str}",
        len - head - tail
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    #[test]
    fn test_middle_truncate_keeps_both_ends() {
        let cut = middle_truncate(&"a".repeat(10), 4);
        assert!(cut.starts_with("aa\n\n... [6 characters omitted]"), "{cut}");
        assert_eq!(middle_truncate("short", 10), "short");
    }

    #[test]
    fn test_a_piece_is_clamped_as_the_whole_result_is() {
        let long = "x".repeat(MAX_TOOL_RESULT_CHARS + 1);
        let delta = cap_tool_delta(
            MessageDelta::new()
                .with_role(Role::Tool)
                .with_contents([PartDelta::Text { text: long.clone() }]),
        );
        let message = cap_tool_result(Message::new(Role::Tool).with_contents([Part::text(long)]));
        let PartDelta::Text { text } = &delta.contents[0] else {
            panic!("a text piece stays text");
        };
        assert!(text.contains("characters omitted"));
        assert_eq!(Some(text.as_str()), message.contents[0].as_text());
    }
}
