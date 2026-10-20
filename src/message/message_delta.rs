use std::fmt;

use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::message::{
    Delta, FinishReason, Message, MessageOutput, PartDelta, Role, TokenUsage,
    tool_result::cap_tool_result,
};

/// A streaming, incremental update to a [`Message`].
///
/// `MessageDelta` accumulates partial outputs (text chunks, tool-call fragments, IDs, signatures, etc.) until they can be materialized as a full [`Message`].
/// It implements [`Delta`] to support accumulation.
///
/// # Accumulation Rules
/// - `role`: merging two distinct roles fails.
/// - `thinking`: concatenated in arrival order.
/// - `contents`/`tool_calls`: last element is accumulated with the incoming delta when both are compatible (e.g., Text+Text, Function+Function with matching ID policy), otherwise appended as a new fragment.
/// - `id`/`signature`: last-writer-wins.
///
/// # Finalization
/// - `finish()` converts the accumulated deltas into a fully-formed [`Message`].
///   Fails if required fields (e.g., `role`) are missing or inner deltas cannot be finalized.
///
/// # Examples
/// ```rust
/// # use ailoy::message::{MessageDelta, PartDelta, Role, Delta};
/// let d1 = MessageDelta::new().with_role(Role::Assistant).with_contents([PartDelta::Text { text: "Hel".into() }]);
/// let d2 = MessageDelta::new().with_contents([PartDelta::Text { text: "lo".into() }]);
///
/// let merged = d1.accumulate(d2).unwrap();
/// let msg = merged.finish().unwrap();
/// assert_eq!(msg.contents[0].as_text().unwrap(), "Hello");
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    pub contents: Vec<PartDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    pub tool_calls: Vec<PartDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl MessageDelta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_role(mut self, role: impl Into<Role>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn with_thinking_signature(
        mut self,
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.thinking = Some(thinking.into());
        self.signature = Some(signature.into());
        self
    }

    pub fn with_contents(
        mut self,
        contents: impl IntoIterator<Item = impl Into<PartDelta>>,
    ) -> Self {
        self.contents = contents.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn with_tool_calls(
        mut self,
        tool_calls: impl IntoIterator<Item = impl Into<PartDelta>>,
    ) -> Self {
        self.tool_calls = tool_calls.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn to_message(self) -> anyhow::Result<Message> {
        self.finish()
    }
}

impl Delta for MessageDelta {
    type Item = Message;
    type Err = anyhow::Error; // TODO: Define custom error for this.

    fn accumulate(self, other: Self) -> anyhow::Result<Self> {
        let Self {
            mut role,
            mut contents,
            mut id,
            mut thinking,
            mut tool_calls,
            mut signature,
        } = self;

        if let Some(lhs) = &role
            && let Some(rhs) = &other.role
        {
            if lhs != rhs {
                bail!(
                    "Cannot accumulate two message deltas with differenct roles. ({} != {})",
                    lhs,
                    rhs
                );
            };
        } else if let Some(rhs) = other.role {
            role = Some(rhs);
        };

        if let Some(id_incoming) = other.id {
            id = Some(id_incoming);
        }

        if let Some(thinking_rhs) = other.thinking {
            if let Some(mut thinking_lhs) = thinking {
                thinking_lhs.push_str(&thinking_rhs);
                thinking = Some(thinking_lhs);
            } else {
                thinking = Some(thinking_rhs);
            }
        }

        for part_incoming in other.contents {
            if let Some(part_last) = contents.last() {
                match (part_last, &part_incoming) {
                    (PartDelta::Text { .. }, PartDelta::Text { .. })
                    | (PartDelta::Function { .. }, PartDelta::Function { .. }) => {
                        let v = contents.pop().unwrap().accumulate(part_incoming)?;
                        contents.push(v);
                    }
                    _ => contents.push(part_incoming),
                }
            } else {
                contents.push(part_incoming);
            }
        }

        // Assumes calls stream *sequentially*, with the `id` on each call's first
        // fragment and `id = None` continuations merging into the last call. True for
        // OpenAI, Anthropic, Gemini, Bedrock and the OpenAI-compatible backends
        // (DeepSeek / Kimi / Grok).
        //
        // KNOWN LIMITATION: `tool_calls[].index`, the documented OpenAI-compatible merge
        // key, is ignored. A backend interleaving parallel calls by `index` (or omitting
        // a new call's id) would leave one call with empty arguments and another with
        // invalid concatenated JSON, which finish() rejects. Fixing it needs `index`
        // carried through `PartDelta::Function` and matched here instead of `last()`.
        for part_incoming in other.tool_calls {
            if let Some(part_last) = tool_calls.last() {
                match (part_last, &part_incoming) {
                    (PartDelta::Text { .. }, PartDelta::Text { .. }) => {
                        let v = tool_calls.pop().unwrap().accumulate(part_incoming)?;
                        tool_calls.push(v);
                    }
                    (
                        PartDelta::Function {
                            id: id1,
                            function: f1,
                        },
                        PartDelta::Function {
                            id: id2,
                            function: f2,
                        },
                    ) => {
                        if let Some(id1) = id1
                            && let Some(id2) = id2
                            && id1 != id2
                        {
                            tool_calls.push(part_incoming);
                        } else if id2.is_none()
                            && std::mem::discriminant(f1) == std::mem::discriminant(f2)
                        {
                            let v = tool_calls.pop().unwrap().accumulate(part_incoming)?;
                            tool_calls.push(v);
                        } else {
                            tool_calls.push(part_incoming);
                        }
                    }
                    _ => tool_calls.push(part_incoming),
                }
            } else {
                tool_calls.push(part_incoming);
            }
        }

        if let Some(sig_incoming) = other.signature {
            signature = Some(sig_incoming);
        }

        Ok(Self {
            role,
            contents,
            id,
            thinking,
            tool_calls,
            signature,
        })
    }

    fn finish(self) -> anyhow::Result<Self::Item> {
        let Self {
            role,
            mut contents,
            id,
            thinking,
            mut tool_calls,
            signature,
        } = self;

        let Some(role) = role else {
            bail!("Role not specified")
        };
        let contents = {
            let mut contents_new = Vec::with_capacity(contents.len());
            for v in contents.drain(..) {
                contents_new.push(v.finish()?);
            }
            contents_new
        };
        let tool_calls = {
            let mut tool_calls_new = Vec::with_capacity(tool_calls.len());
            for v in tool_calls.drain(..) {
                tool_calls_new.push(v.finish()?);
            }
            Some(tool_calls_new)
        };
        Ok(Message {
            role,
            contents,
            id,
            thinking,
            tool_calls,
            signature,
        })
    }
}

impl fmt::Display for MessageDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "MessageDelta {}", s)
    }
}

/// A streamed [`MessageDelta`] plus its termination signal; `finish_reason` is
/// set on the delta that ends a message.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MessageDeltaOutput {
    pub delta: MessageDelta,

    pub finish_reason: Option<FinishReason>,

    pub usage: Option<TokenUsage>,
}

impl MessageDeltaOutput {
    pub fn new() -> Self {
        Self {
            delta: MessageDelta::new(),
            finish_reason: None,
            usage: None,
        }
    }
}

impl From<Message> for MessageDelta {
    /// Wraps a finalized [`Message`] as a complete delta (e.g. a tool result,
    /// produced whole, surfaced on the streaming path).
    fn from(msg: Message) -> Self {
        MessageDelta {
            role: Some(msg.role),
            contents: msg.contents.into_iter().map(PartDelta::from).collect(),
            id: msg.id,
            thinking: msg.thinking,
            tool_calls: msg
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .map(PartDelta::from)
                .collect(),
            signature: msg.signature,
        }
    }
}

impl From<MessageOutput> for MessageDeltaOutput {
    /// Wraps a finalized [`MessageOutput`] as a complete delta, carrying its
    /// `finish_reason` and usage.
    fn from(out: MessageOutput) -> Self {
        MessageDeltaOutput {
            delta: out.message.into(),
            finish_reason: Some(out.finish_reason),
            usage: out.usage,
        }
    }
}

/// Adapts a [`MessageDeltaOutput`] stream (e.g. from `Agent::run_stream`) into completed
/// [`MessageOutput`]s, the view `Agent::run` gives. A delta carrying a `finish_reason`
/// ends each message; ailoy's producers guarantee every message has one.
///
/// The results of one batch of tool calls stream concurrently, so their deltas interleave;
/// each is assembled by its call id, and clamped as the history keeps it.
pub fn into_messages<S>(
    deltas: S,
) -> impl futures::Stream<Item = anyhow::Result<MessageOutput>> + Send
where
    S: futures::Stream<Item = anyhow::Result<MessageDeltaOutput>> + Send,
{
    use futures::StreamExt as _;
    async_stream::try_stream! {
        let mut deltas = std::pin::pin!(deltas);
        let mut assembler = Assembler::default();
        while let Some(item) = deltas.next().await {
            if let Some(output) = assembler.push(item?)? {
                yield output;
            }
        }
        // Only for a non-conforming stream ending mid-message: keep the partials as Stop.
        for output in assembler.rest() {
            yield output?;
        }
    }
}

/// Assembles [`MessageOutput`]s from deltas: the model's one message at a time, and tool
/// results one per call id, since a batch's calls run concurrently.
#[derive(Default)]
pub(crate) struct Assembler {
    message: MessageDeltaOutput,
    tools: Vec<(String, MessageDeltaOutput)>,
}

impl Assembler {
    /// Take in `delta`, and give back the message it finishes, if any. A tool result comes
    /// out clamped, as the history keeps it.
    pub(crate) fn push(
        &mut self,
        delta: MessageDeltaOutput,
    ) -> anyhow::Result<Option<MessageOutput>> {
        let tool = match (&delta.delta.role, &delta.delta.id) {
            (Some(Role::Tool), Some(id)) => {
                Some(match self.tools.iter().position(|(other, _)| other == id) {
                    Some(at) => at,
                    None => {
                        self.tools.push((id.clone(), MessageDeltaOutput::new()));
                        self.tools.len() - 1
                    }
                })
            }
            _ => None,
        };
        let slot = match tool {
            Some(at) => &mut self.tools[at].1,
            None => &mut self.message,
        };
        *slot = std::mem::take(slot).accumulate(delta)?;
        if slot.finish_reason.is_none() {
            return Ok(None);
        }
        let done = match tool {
            Some(at) => self.tools.remove(at).1,
            None => std::mem::take(&mut self.message),
        };
        finish(done).map(Some)
    }

    /// The messages left unfinished, finished as `Stop`.
    pub(crate) fn rest(self) -> impl Iterator<Item = anyhow::Result<MessageOutput>> {
        std::iter::once(self.message)
            .chain(self.tools.into_iter().map(|(_, acc)| acc))
            .filter(|acc| acc.delta.role.is_some())
            .map(|mut acc| {
                acc.finish_reason = Some(FinishReason::Stop {});
                finish(acc)
            })
    }
}

fn finish(acc: MessageDeltaOutput) -> anyhow::Result<MessageOutput> {
    let mut output = acc.finish()?;
    if output.message.role == Role::Tool {
        output.message = cap_tool_result(output.message);
    }
    Ok(output)
}

impl Delta for MessageDeltaOutput {
    type Item = MessageOutput;
    type Err = anyhow::Error;

    fn accumulate(self, other: Self) -> anyhow::Result<Self> {
        let delta = self.delta.accumulate(other.delta)?;
        let finish_reason = other.finish_reason.or(self.finish_reason);
        let usage = merge_token_usage(self.usage, other.usage);
        Ok(Self {
            delta,
            finish_reason,
            usage,
        })
    }

    fn finish(self) -> anyhow::Result<Self::Item> {
        let message = self.delta.finish()?;
        let mut finish_reason = self
            .finish_reason
            .ok_or_else(|| anyhow::anyhow!("finish_reason not specified"))?;
        // Promote Stop to ToolCall when tool calls exist. Done on the accumulated
        // message because Gemini (2.5 Pro / 3.x) can send the functionCall part and
        // `finishReason: STOP` in separate SSE chunks.
        if matches!(finish_reason, FinishReason::Stop {})
            && message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
        {
            finish_reason = FinishReason::ToolCall {};
        }
        Ok(MessageOutput {
            message,
            finish_reason,
            usage: self.usage,
        })
    }
}

impl fmt::Display for MessageDeltaOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "MessageDeltaOutput {}", s)
    }
}

/// Merges two streamed [`TokenUsage`] snapshots field-wise, taking the larger
/// value per field.
///
/// Providers spread usage across events (e.g. Anthropic sends `input_tokens` in
/// `message_start` and cumulative `output_tokens` in `message_delta`), so replacing the
/// whole value would drop counters. `max` is correct because the counters are monotonic.
fn merge_token_usage(a: Option<TokenUsage>, b: Option<TokenUsage>) -> Option<TokenUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(TokenUsage {
            input_tokens: a.input_tokens.max(b.input_tokens),
            output_tokens: a.output_tokens.max(b.output_tokens),
            cache_creation_input_tokens: merge_opt_max(
                a.cache_creation_input_tokens,
                b.cache_creation_input_tokens,
            ),
            cache_read_input_tokens: merge_opt_max(
                a.cache_read_input_tokens,
                b.cache_read_input_tokens,
            ),
        }),
        (a, b) => b.or(a),
    }
}

fn merge_opt_max(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => b.or(a),
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::message::Part;

    fn text_delta(text: &str) -> MessageDeltaOutput {
        MessageDeltaOutput {
            delta: MessageDelta::new()
                .with_role(Role::Assistant)
                .with_contents([PartDelta::Text { text: text.into() }]),
            ..MessageDeltaOutput::new()
        }
    }

    /// into_messages re-segments a delta stream at finish_reason boundaries:
    /// fragmented assistant deltas + a one-shot tool delta become two messages.
    #[tokio::test]
    async fn test_into_messages_segments_at_boundaries() {
        let assistant_end = MessageDeltaOutput {
            finish_reason: Some(FinishReason::ToolCall {}),
            ..MessageDeltaOutput::new()
        };
        let tool_result: MessageDeltaOutput = MessageOutput {
            message: Message::new(Role::Tool)
                .with_contents([Part::text("42")])
                .with_id("calc/1"),
            finish_reason: FinishReason::Stop {},
            usage: None,
        }
        .into();

        let deltas = futures::stream::iter(
            [
                text_delta("Hel"),
                text_delta("lo"),
                assistant_end,
                tool_result,
            ]
            .map(anyhow::Ok),
        );
        let messages: Vec<_> = into_messages(deltas).map(|m| m.unwrap()).collect().await;

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message.role, Role::Assistant);
        assert_eq!(messages[0].message.contents[0].as_text(), Some("Hello"));
        assert_eq!(messages[0].finish_reason, FinishReason::ToolCall {});
        assert_eq!(messages[1].message.role, Role::Tool);
        assert_eq!(messages[1].message.contents[0].as_text(), Some("42"));
    }

    /// A role change without an intervening finish_reason violates the producer contract
    /// and must error rather than silently split.
    #[tokio::test]
    async fn test_into_messages_errors_on_role_change_without_finish() {
        let tool_delta = MessageDeltaOutput {
            delta: MessageDelta::new()
                .with_role(Role::Tool)
                .with_contents([PartDelta::Text { text: "42".into() }]),
            ..MessageDeltaOutput::new()
        };
        // Assistant content (no finish_reason) flowing straight into a Tool role.
        let deltas = futures::stream::iter([text_delta("Hi"), tool_delta].map(anyhow::Ok));
        let results: Vec<_> = into_messages(deltas).collect().await;

        assert!(
            results.iter().any(|r| r.is_err()),
            "role change without a finish_reason must error, not silently split"
        );
    }

    fn tool_delta(id: &str, text: &str, done: bool) -> MessageDeltaOutput {
        MessageDeltaOutput {
            delta: MessageDelta::new()
                .with_role(Role::Tool)
                .with_id(id)
                .with_contents([PartDelta::Text { text: text.into() }]),
            finish_reason: done.then_some(FinishReason::Stop {}),
            usage: None,
        }
    }

    /// Two calls' results interleave; each is assembled by its id.
    #[tokio::test]
    async fn test_into_messages_assembles_tool_results_by_call_id() {
        let deltas = futures::stream::iter(
            [
                tool_delta("a", "1", false),
                tool_delta("b", "x", false),
                tool_delta("a", "2", false),
                tool_delta("b", "y", true),
                tool_delta("a", "3", true),
            ]
            .map(anyhow::Ok),
        );
        let messages: Vec<_> = into_messages(deltas).map(|m| m.unwrap()).collect().await;

        let results: Vec<_> = messages
            .iter()
            .map(|m| {
                (
                    m.message.id.as_deref().unwrap(),
                    m.message.contents[0].as_text().unwrap(),
                )
            })
            .collect();
        assert_eq!(results, [("b", "xy"), ("a", "123")]);
    }

    /// A tool result comes out clamped as the history keeps it, though its pieces were not.
    #[tokio::test]
    async fn test_into_messages_clamps_a_tool_result() {
        let half = "x".repeat(20_000);
        let deltas = futures::stream::iter(
            [tool_delta("a", &half, false), tool_delta("a", &half, true)].map(anyhow::Ok),
        );
        let messages: Vec<_> = into_messages(deltas).map(|m| m.unwrap()).collect().await;

        let text = messages[0].message.contents[0].as_text().unwrap();
        assert!(text.contains("characters omitted"), "{}", text.len());
    }

    /// A stream that ends mid-message (no terminal finish_reason) still yields
    /// the partial content as a Stop-finished message instead of dropping it.
    #[tokio::test]
    async fn test_into_messages_finishes_trailing_partial() {
        let deltas = futures::stream::iter([anyhow::Ok(text_delta("partial"))]);
        let messages: Vec<_> = into_messages(deltas).map(|m| m.unwrap()).collect().await;

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].finish_reason, FinishReason::Stop {});
        assert_eq!(messages[0].message.contents[0].as_text(), Some("partial"));
    }
}
