use std::collections::VecDeque;

use crate::{
    datatype::Value,
    message::{
        Delta as _, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part,
        PartDelta,
    },
};

/// Per-part size limit for tool results, past which the middle is cut out.
pub(crate) const MAX_TOOL_RESULT_CHARS: usize = 30_000;

/// Clamp every part of a tool result, so large payloads (e.g. web pages) don't pile up in
/// the history. An oversized [`Part::Value`] that is not a string is measured as JSON and
/// replaced by the cut JSON as a string. A clamped part is within the limit, so clamping
/// again leaves it as it is.
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
    if let Some(text) = value.as_str() {
        if text.len() > MAX_TOOL_RESULT_CHARS {
            *value = Value::string(middle_truncate(text, MAX_TOOL_RESULT_CHARS));
        }
        return;
    }
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

/// `s` within `max_chars` characters: whole if it fits, else its two ends around a note of how
/// many characters were cut from between them.
pub(crate) fn middle_truncate(s: &str, max_chars: usize) -> String {
    let mut kept = KeptText::new(max_chars);
    kept.push(s);
    kept.finish()
}

/// Text pushed in pieces, holding only what [`middle_truncate`] keeps of the whole: the most
/// it could keep from each end, and a count of the rest.
pub(crate) struct KeptText {
    max_chars: usize,
    head: String,
    head_chars: usize,
    tail: VecDeque<char>,
    total: usize,
}

impl KeptText {
    pub(crate) fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            head: String::new(),
            head_chars: 0,
            tail: VecDeque::new(),
            total: 0,
        }
    }

    pub(crate) fn push(&mut self, text: &str) {
        let head_max = self.max_chars / 2;
        let tail_max = self.max_chars - head_max;
        for c in text.chars() {
            self.total += 1;
            if self.head_chars < head_max {
                self.head.push(c);
                self.head_chars += 1;
            } else {
                self.tail.push_back(c);
                if self.tail.len() > tail_max {
                    self.tail.pop_front();
                }
            }
        }
    }

    pub(crate) fn finish(self) -> String {
        if self.total <= self.max_chars {
            return self.head + &self.tail.iter().collect::<String>();
        }
        let note = |omitted: usize| format!("\n\n... [{omitted} characters omitted] ...\n\n");
        // The note counts toward the limit. Sized for `total`, which has at least as many
        // digits as what is omitted, so the cut fits whatever that turns out to be.
        let keep = self
            .max_chars
            .saturating_sub(note(self.total).chars().count());
        let head = keep / 2;
        let tail = keep - head;
        let head_str: String = self.head.chars().take(head).collect();
        let tail_str: String = self.tail.iter().skip(self.tail.len() - tail).collect();
        format!("{head_str}{}{tail_str}", note(self.total - head - tail))
    }
}

/// A tool result put together from its pieces, its text cut as the pieces arrive to what the
/// history keeps of it, so a long stream is not held whole.
#[derive(Default)]
pub(crate) struct ToolResult {
    rest: MessageDeltaOutput,
    contents: Vec<Content>,
}

enum Content {
    Text(KeptText),
    Other(PartDelta),
}

impl ToolResult {
    pub(crate) fn push(&mut self, mut delta: MessageDeltaOutput) -> anyhow::Result<()> {
        let parts = std::mem::take(&mut delta.delta.contents);
        self.rest = std::mem::take(&mut self.rest).accumulate(delta)?;
        for part in parts {
            match (self.contents.last_mut(), part) {
                (Some(Content::Text(kept)), PartDelta::Text { text }) => kept.push(&text),
                (_, PartDelta::Text { text }) => {
                    let mut kept = KeptText::new(MAX_TOOL_RESULT_CHARS);
                    kept.push(&text);
                    self.contents.push(Content::Text(kept));
                }
                (Some(Content::Other(last)), part @ PartDelta::Function { .. })
                    if matches!(last, PartDelta::Function { .. }) =>
                {
                    *last = std::mem::replace(last, PartDelta::Null {}).accumulate(part)?;
                }
                (_, part) => self.contents.push(Content::Other(part)),
            }
        }
        Ok(())
    }

    pub(crate) fn is_done(&self) -> bool {
        self.rest.finish_reason.is_some()
    }

    /// The result, clamped as the history keeps it; `Stop` if no delta ended it.
    pub(crate) fn finish(self) -> anyhow::Result<MessageOutput> {
        let mut acc = self.rest;
        acc.finish_reason.get_or_insert(FinishReason::Stop {});
        acc.delta.contents = self
            .contents
            .into_iter()
            .map(|content| match content {
                Content::Text(kept) => PartDelta::Text {
                    text: kept.finish(),
                },
                Content::Other(part) => part,
            })
            .collect();
        let mut output = acc.finish()?;
        output.message = cap_tool_result(output.message);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Role;

    #[test]
    fn test_middle_truncate_keeps_both_ends_within_the_limit() {
        assert_eq!(middle_truncate("short", 10), "short");

        let s: String = ('a'..='z').cycle().take(1_000).collect();
        let cut = middle_truncate(&s, 100);
        assert!(cut.chars().count() <= 100, "{}", cut.chars().count());
        let (head, rest) = cut.split_once("\n\n... [").unwrap();
        let (omitted, tail) = rest.split_once(" characters omitted] ...\n\n").unwrap();
        assert!(s.starts_with(head) && s.ends_with(tail));
        assert_eq!(
            head.len() + omitted.parse::<usize>().unwrap() + tail.len(),
            s.len()
        );
    }

    #[test]
    fn test_clamping_twice_is_clamping_once() {
        let long = "가".repeat(MAX_TOOL_RESULT_CHARS + 1);
        let once = cap_tool_result(Message::new(Role::Tool).with_contents([
            Part::text(long.clone()),
            Part::value(Value::string(long.clone())),
            Part::value(crate::to_value!({ "text": long.as_str() })),
        ]));
        let twice = cap_tool_result(once.clone());
        assert_eq!(
            serde_json::to_value(&once).unwrap(),
            serde_json::to_value(&twice).unwrap()
        );
    }

    /// Pushed in pieces, the text comes out as the whole would, while holding no more than
    /// the limit.
    #[test]
    fn test_kept_text_is_bounded_and_cuts_as_the_whole_would() {
        for len in [0, 99, 100, 101, 1_000, 123_457] {
            let s: String = "가a".chars().cycle().take(len).collect();
            let mut kept = KeptText::new(100);
            let chars: Vec<char> = s.chars().collect();
            for piece in chars.chunks(7) {
                kept.push(&piece.iter().collect::<String>());
                assert!(kept.head_chars + kept.tail.len() <= 100);
            }
            assert_eq!(kept.finish(), middle_truncate(&s, 100), "len {len}");
        }
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
