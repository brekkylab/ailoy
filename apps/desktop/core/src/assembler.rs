//! Reassembles ailoy's raw streaming output into discrete, actionable items.
//!
//! `Agent::run_stream` yields a uniform stream of [`MessageDeltaOutput`]s.
//! ailoy's stream contract guarantees every message ends with a delta carrying a
//! `finish_reason` (the LangModel layer synthesizes a terminal Stop delta when a
//! provider ends the stream without one), so a message boundary is exactly
//! "`finish_reason` present". [`MessageAssembler`] folds deltas to that boundary
//! while also passing the assistant's text and thinking through for live
//! rendering — giving the desktop UI both views of the same stream: the
//! characters to paint now, and the finalized message to persist. A
//! non-conforming stream (role change with no intervening `finish_reason`) fails
//! loudly via `accumulate` rather than being silently healed.

use ailoy::message::{
    Delta as _, FinishReason, MessageDeltaOutput, MessageOutput, PartDelta, PartDeltaFunction,
    RateLimitInfo, Role, TokenUsage,
};

/// How much of a call's arguments is passed through while the model writes them.
///
/// Enough for what names a call — a path, a command, a query — which the model writes near
/// the start; not the file body a `write` carries after it, which nothing live draws and
/// which can run to hundreds of kilobytes. The whole of the arguments still arrives with
/// the finished message.
pub const ARGS_PREVIEW_MAX: usize = 4096;

/// One actionable item derived from the raw delta stream.
#[derive(Debug)]
pub enum AssembledItem {
    /// Newly streamed top-level assistant text for live rendering. Empty
    /// fragments are never emitted, and tool results and sub-agent (depth ≥ 1)
    /// output never appear here — they surface only as
    /// [`AssembledItem::Completed`], matching the pre-streaming-API semantics.
    Text(String),
    /// Newly streamed top-level reasoning text, under the same conditions as
    /// [`AssembledItem::Text`]. Kept a separate variant so the UI can render
    /// thinking in its own region instead of splicing it into the answer.
    Thinking(String),
    /// A message finalized at a boundary: a completed assistant turn or a tool
    /// result. Boxed so this variant doesn't bloat the small `Text` case.
    Completed(Box<MessageOutput>),
    /// Accounting that arrived *after* a message boundary: it belongs to the
    /// message that just completed, not to the next one.
    ///
    /// The ChatCompletion schema reports a turn's usage in a frame of its own
    /// (ailoy asks for it with `stream_options.include_usage`; xAI, DeepSeek,
    /// Moonshot Kimi and custom OpenAI-compatible endpoints all answer that
    /// way). That frame has empty `choices` and arrives *after* the one carrying
    /// `finish_reason`, so ailoy unmarshals it as a role-less, content-less,
    /// finish-less delta carrying only `usage` — landing here once this
    /// assembler has already cut the message. Folding it into the accumulator
    /// instead would bill the *next* message (in a tool-using run, the tool
    /// result) for the model's turn, and a stream that ends right after it would
    /// lose the turn's tokens altogether.
    ///
    /// Boxed for the same reason as [`AssembledItem::Completed`]: `RateLimitInfo`
    /// alone is four windows wide, and this variant is rare.
    UsageTrailer(Box<UsageTrailer>),
    /// A top-level tool call the model has begun writing: its id and name are known, its
    /// arguments are not yet. Providers name a call before they stream what goes in it —
    /// Anthropic's `content_block_start`, the ChatCompletion call's first chunk, Bedrock's
    /// `contentBlockStart` — and a `write` spends almost all of its time in between, writing
    /// out the file body. Without this the window shows nothing for that whole stretch and
    /// then, all at once, a call that has already finished.
    ToolCallBegan { id: String, name: String },
    /// More of a begun call's arguments, as the model writes them: raw JSON text, in order,
    /// up to [`ARGS_PREVIEW_MAX`] bytes a call.
    ToolCallArgs { id: String, chunk: String },
}

/// Accounting that belongs to the message before it — see
/// [`AssembledItem::UsageTrailer`].
#[derive(Debug)]
pub struct UsageTrailer {
    pub usage: Option<TokenUsage>,
    pub rate_limit: Option<RateLimitInfo>,
    /// The nesting level of the turn it accounts for, as the delta reported it.
    pub depth: Option<u8>,
}

/// Accumulates streamed [`MessageDeltaOutput`]s and emits [`AssembledItem`]s at
/// message boundaries. Feed each delta to [`push`](Self::push); call
/// [`finish`](Self::finish) once the stream ends to flush any trailing message.
#[derive(Default)]
pub struct MessageAssembler {
    /// The message currently being streamed, accumulated across deltas.
    acc: MessageDeltaOutput,
    /// Per tool call of the message being streamed, by position: its id once it has been
    /// announced, and how many bytes of its arguments have been passed through.
    announced: Vec<(String, usize)>,
}

impl MessageAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one streamed delta. Returns the items to act on, in order: an
    /// optional `Text` and an optional `Thinking` (the live fragments from this
    /// delta), then an optional `Completed` (when this delta carries the
    /// message's `finish_reason`). A usage-only delta arriving between messages
    /// is the exception: it yields a lone [`AssembledItem::UsageTrailer`] and
    /// touches nothing. `Err` carries a finalization/accumulation failure
    /// message; a role change with no intervening `finish_reason` (contract
    /// violation) lands here via `accumulate`'s role-mismatch rejection.
    pub fn push(&mut self, delta: MessageDeltaOutput) -> Result<Vec<AssembledItem>, String> {
        // 0. A usage-only frame *between* messages closes the accounting of the
        //    one that just ended — see `AssembledItem::UsageTrailer`. Only
        //    between: the same frame shape mid-message is ordinary interim usage
        //    (Anthropic sends input counts early, output last) and must keep
        //    accumulating into the message being built.
        if self.is_fresh() && is_usage_only(&delta) {
            return Ok(vec![AssembledItem::UsageTrailer(Box::new(UsageTrailer {
                usage: delta.usage,
                rate_limit: delta.rate_limit,
                depth: delta.depth,
            }))]);
        }

        let mut items = Vec::new();

        // 1. Live assistant text — top-level only. A sub-agent's answer is
        //    re-emitted on this same stream as a role=Assistant one-shot at
        //    depth ≥ 1; streaming it here would splice the sub-agent's internal
        //    text into the top-level answer (and a stop would persist that mix).
        //    A continuation delta carries no role/depth, so fall back to the
        //    in-progress message's; exclude Tool rather than require Assistant
        //    so text streamed before the role marker isn't dropped.
        let effective_role = delta.delta.role.as_ref().or(self.acc.delta.role.as_ref());
        let is_top_level = matches!(delta.depth.or(self.acc.depth), None | Some(0));
        if is_top_level && !matches!(effective_role, Some(Role::Tool)) {
            let mut fragment = String::new();
            for part in &delta.delta.contents {
                if let PartDelta::Text { text } = part {
                    fragment.push_str(text);
                }
            }
            if !fragment.is_empty() {
                items.push(AssembledItem::Text(fragment));
            }
            if let Some(th) = delta.delta.thinking.as_deref().filter(|t| !t.is_empty()) {
                items.push(AssembledItem::Thinking(th.to_string()));
            }
        }

        // 2. Fold this delta into the running message.
        let acc = std::mem::take(&mut self.acc);
        self.acc = acc.accumulate(delta).map_err(|e| e.to_string())?;

        // 2b. Tool calls being written — top-level assistant calls only, for the reason
        //     step 1 gives. Read off the accumulated message rather than off the delta, so
        //     which call a fragment belongs to is decided exactly as `accumulate` decided it
        //     (a fragment without an id continues the last call) and never a second way.
        let is_top_level = matches!(self.acc.depth, None | Some(0));
        if is_top_level && !matches!(self.acc.delta.role, Some(Role::Tool)) {
            self.announce_calls(&mut items);
        }

        // 3. A delta carrying finish_reason finalizes the message — per ailoy's
        //    stream contract, this is the message boundary.
        if self.acc.finish_reason.is_some() {
            self.announced.clear();
            let done = std::mem::take(&mut self.acc);
            items.push(AssembledItem::Completed(Box::new(
                done.finish().map_err(|e| e.to_string())?,
            )));
        }

        Ok(items)
    }

    /// Flush a message still accumulating when the stream ends. Defensive:
    /// ailoy's contract (every message ends with a finish_reason delta) makes
    /// this a no-op for conforming streams — `Some` here means an upstream
    /// producer broke the contract, so the caller should log it. The message is
    /// still returned (finalized as Stop) rather than dropped, because losing a
    /// completed answer is strictly worse than surfacing a contract bug quietly.
    /// Returns `None` if nothing pending.
    pub fn finish(&mut self) -> Result<Option<MessageOutput>, String> {
        if self.acc.delta.role.is_none() {
            return Ok(None);
        }
        self.announced.clear();
        let mut done = std::mem::take(&mut self.acc);
        done.finish_reason.get_or_insert(FinishReason::Stop {});
        Ok(Some(done.finish().map_err(|e| e.to_string())?))
    }

    /// Emit what is new about the calls the running message holds: a call that now has an
    /// id and a name, and the arguments written since the last delta, up to the cap.
    fn announce_calls(&mut self, items: &mut Vec<AssembledItem>) {
        for (i, part) in self.acc.delta.tool_calls.iter().enumerate() {
            let PartDelta::Function {
                id: Some(id),
                function: PartDeltaFunction::WithStringArgs { name, arguments },
            } = part
            else {
                // Positions are what tie a call to what was sent about it, so a call this
                // cannot read yet holds back the ones after it rather than shifting them.
                break;
            };
            if i == self.announced.len() {
                // A call is announced once it can be named; one whose first fragment came
                // without a name waits for the fragment that has it.
                if name.is_empty() {
                    break;
                }
                items.push(AssembledItem::ToolCallBegan {
                    id: id.clone(),
                    name: name.clone(),
                });
                self.announced.push((id.clone(), 0));
            }
            let Some((_, sent)) = self.announced.get_mut(i) else {
                break;
            };
            let mut end = arguments.len().min(ARGS_PREVIEW_MAX);
            while !arguments.is_char_boundary(end) {
                end -= 1;
            }
            if end > *sent {
                items.push(AssembledItem::ToolCallArgs {
                    id: id.clone(),
                    chunk: arguments[*sent..end].to_string(),
                });
                *sent = end;
            }
        }
    }

    /// Whether no message is being built right now — i.e. the last one was cut
    /// at its `finish_reason` and the next has not started.
    fn is_fresh(&self) -> bool {
        self.acc.delta.role.is_none()
            && self.acc.delta.contents.is_empty()
            && self.acc.delta.tool_calls.is_empty()
            && self.acc.delta.thinking.is_none()
    }
}

/// A delta that carries accounting and nothing else: no role, no content, no
/// tool call, no thinking, and no `finish_reason` of its own.
fn is_usage_only(delta: &MessageDeltaOutput) -> bool {
    delta.delta.role.is_none()
        && delta.delta.contents.is_empty()
        && delta.delta.tool_calls.is_empty()
        && delta.delta.thinking.is_none()
        && delta.finish_reason.is_none()
        && (delta.usage.is_some() || delta.rate_limit.is_some())
}

#[cfg(test)]
mod tests {
    use ailoy::message::MessageDelta;

    use super::*;

    /// Build a streamed delta with an optional role, text, and finish_reason.
    fn delta(role: Option<Role>, text: &str, finish: bool) -> MessageDeltaOutput {
        let mut md = MessageDelta::new();
        if let Some(r) = role {
            md = md.with_role(r);
        }
        if !text.is_empty() {
            md = md.with_contents([PartDelta::Text { text: text.into() }]);
        }
        let mut out = MessageDeltaOutput::new();
        out.delta = md;
        out.depth = Some(0);
        if finish {
            out.finish_reason = Some(FinishReason::Stop {});
        }
        out
    }

    fn text_of(items: &[AssembledItem]) -> Vec<String> {
        items
            .iter()
            .filter_map(|i| match i {
                AssembledItem::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    fn completed(items: Vec<AssembledItem>) -> Vec<MessageOutput> {
        items
            .into_iter()
            .filter_map(|i| match i {
                AssembledItem::Completed(o) => Some(*o),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn streams_text_then_completes_on_finish_reason() {
        let mut a = MessageAssembler::new();

        let first = a.push(delta(Some(Role::Assistant), "Hel", false)).unwrap();
        assert_eq!(text_of(&first), vec!["Hel"]);
        assert!(completed(first).is_empty());

        let second = a.push(delta(None, "lo", true)).unwrap();
        assert_eq!(text_of(&second), vec!["lo"]);
        let done = completed(second);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].message.role, Role::Assistant);
        assert_eq!(done[0].message.contents[0].as_text().unwrap(), "Hello");
        assert!(matches!(done[0].finish_reason, FinishReason::Stop {}));

        // Nothing pending after a clean finish.
        assert!(a.finish().unwrap().is_none());
    }

    #[test]
    fn flushes_trailing_message_when_stream_ends_without_finish_reason() {
        // Defensive path: ailoy's contract means a conforming stream never ends
        // mid-message, but if one does (upstream bug), the trailing message must
        // be recovered — losing a completed answer is worse than the anomaly.
        let mut a = MessageAssembler::new();

        let items = a
            .push(delta(Some(Role::Assistant), "answer", false))
            .unwrap();
        assert_eq!(text_of(&items), vec!["answer"]);
        assert!(completed(items).is_empty());

        let trailing = a.finish().unwrap().expect("trailing message must flush");
        assert_eq!(trailing.message.role, Role::Assistant);
        assert_eq!(trailing.message.contents[0].as_text().unwrap(), "answer");
        assert!(matches!(trailing.finish_reason, FinishReason::Stop {}));
    }

    #[test]
    fn tool_result_surfaces_only_as_completed_never_as_delta() {
        let mut a = MessageAssembler::new();
        // Finish an assistant turn first.
        a.push(delta(Some(Role::Assistant), "calling", true))
            .unwrap();

        // A tool result arrives as its own role=Tool one-shot delta.
        let items = a
            .push(delta(Some(Role::Tool), "tool output", true))
            .unwrap();
        assert!(
            text_of(&items).is_empty(),
            "tool result must not stream as assistant text"
        );
        let done = completed(items);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].message.role, Role::Tool);
    }

    #[test]
    fn role_change_without_finish_reason_errors_loudly() {
        // ailoy's contract guarantees a finish_reason delta closes every message
        // before the role can change; a violation must surface as an error (via
        // accumulate's role-mismatch rejection), not be silently healed into a
        // split — silent healing would mask a broken producer.
        let mut a = MessageAssembler::new();
        let first = a
            .push(delta(Some(Role::Assistant), "thinking", false))
            .unwrap();
        assert_eq!(text_of(&first), vec!["thinking"]);
        assert!(completed(first).is_empty());

        let err = a
            .push(delta(Some(Role::Tool), "result", true))
            .expect_err("role change without finish_reason must be rejected");
        assert!(
            err.contains("role"),
            "error should mention the role mismatch: {err}"
        );
    }

    #[test]
    fn subagent_output_never_streams_as_live_delta() {
        // A sub-agent's answer is re-emitted on the stream as a one-shot
        // role=Assistant delta at depth >= 1. It must not surface as live
        // top-level text (that would splice the sub-agent's internal answer
        // into the main bubble, and a stop would persist the mix) — only as a
        // Completed message, which the client routes to the sub-agent UI.
        let mut a = MessageAssembler::new();
        let mut d = delta(Some(Role::Assistant), "subagent internal answer", true);
        d.depth = Some(1);
        let items = a.push(d).unwrap();
        assert!(
            text_of(&items).is_empty(),
            "sub-agent text must not stream as the top-level answer"
        );
        let done = completed(items);
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].depth, Some(1));
        assert_eq!(
            done[0].message.contents[0].as_text().unwrap(),
            "subagent internal answer"
        );
    }

    #[test]
    fn empty_fragments_are_not_emitted() {
        let mut a = MessageAssembler::new();
        // A role-only opener (e.g. message_start) carries no text.
        let items = a.push(delta(Some(Role::Assistant), "", false)).unwrap();
        assert!(text_of(&items).is_empty());
        assert!(completed(items).is_empty());
    }

    #[test]
    fn finish_on_empty_assembler_yields_nothing() {
        let mut a = MessageAssembler::new();
        assert!(a.finish().unwrap().is_none());
    }

    /// A frame carrying only `usage` — what a ChatCompletion provider sends after
    /// the `finish_reason` one when `stream_options.include_usage` is set.
    fn usage_only(input: u64, output: u64) -> MessageDeltaOutput {
        let mut out = MessageDeltaOutput::new();
        out.depth = Some(0);
        out.usage = Some(TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        });
        out
    }

    #[test]
    fn usage_after_a_message_boundary_is_a_trailer() {
        let mut a = MessageAssembler::new();
        let items = a.push(delta(Some(Role::Assistant), "Hello", true)).unwrap();
        assert!(matches!(
            items.as_slice(),
            [AssembledItem::Text(t), AssembledItem::Completed(_)] if t == "Hello"
        ));

        // The usage frame arrives after the cut. It must not start a new message.
        let items = a.push(usage_only(7, 2)).unwrap();
        let [AssembledItem::UsageTrailer(t)] = items.as_slice() else {
            panic!("expected a lone usage trailer, got {items:?}");
        };
        assert_eq!(t.usage.as_ref().unwrap().input_tokens, 7);
        assert_eq!(t.usage.as_ref().unwrap().output_tokens, 2);
        assert!(t.rate_limit.is_none());
        assert_eq!(t.depth, Some(0));

        // Nothing was accumulated, so the stream can end cleanly right here.
        assert!(a.finish().unwrap().is_none());
    }

    #[test]
    fn usage_mid_message_is_not_a_trailer() {
        // Anthropic reports input counts on an early delta and `output_tokens` on
        // the last one: accounting that arrives while a message is being built is
        // that message's own and must keep accumulating.
        let mut a = MessageAssembler::new();
        a.push(delta(Some(Role::Assistant), "Hel", false)).unwrap();

        let items = a.push(usage_only(7, 0)).unwrap();
        assert!(
            items.is_empty(),
            "interim usage is not a trailer and streams nothing: {items:?}"
        );

        let done = completed(a.push(delta(None, "lo", true)).unwrap());
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].message.contents[0].as_text().unwrap(), "Hello");
        assert_eq!(done[0].usage.as_ref().unwrap().input_tokens, 7);
    }

    #[test]
    fn thinking_fragments_are_emitted_separately() {
        let mut a = MessageAssembler::new();
        let mut d = delta(Some(Role::Assistant), "", false);
        d.delta.thinking = Some("hmm".into());
        let items = a.push(d).unwrap();
        assert!(matches!(items.as_slice(), [AssembledItem::Thinking(t)] if t == "hmm"));
        let items = a.push(delta(None, "answer", true)).unwrap();
        assert!(matches!(&items[0], AssembledItem::Text(t) if t == "answer"));
        assert!(matches!(&items[1], AssembledItem::Completed(_)));
    }

    /// A fragment of a tool call, the way the providers stream one: the first carries the
    /// id and the name, the rest only more of the arguments.
    fn call(id: Option<&str>, name: &str, args: &str) -> MessageDeltaOutput {
        let mut out = MessageDeltaOutput::new();
        out.delta = MessageDelta::new().with_tool_calls([PartDelta::Function {
            id: id.map(str::to_string),
            function: PartDeltaFunction::WithStringArgs {
                name: name.into(),
                arguments: args.into(),
            },
        }]);
        out.depth = Some(0);
        out
    }

    fn began(items: &[AssembledItem]) -> Vec<(String, String)> {
        items
            .iter()
            .filter_map(|i| match i {
                AssembledItem::ToolCallBegan { id, name } => Some((id.clone(), name.clone())),
                _ => None,
            })
            .collect()
    }

    fn args(items: &[AssembledItem]) -> String {
        items
            .iter()
            .filter_map(|i| match i {
                AssembledItem::ToolCallArgs { chunk, .. } => Some(chunk.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_call_is_announced_when_it_is_named_and_its_arguments_follow() {
        let mut a = MessageAssembler::new();
        let first = a.push(delta(Some(Role::Assistant), "", false)).unwrap();
        assert!(began(&first).is_empty());

        let named = a.push(call(Some("c1"), "write", "")).unwrap();
        assert_eq!(began(&named), [("c1".into(), "write".into())]);
        assert_eq!(args(&named), "");

        let mut streamed = String::new();
        for piece in [r#"{"pa"#, r#"th":"/tmp/a.md","#, r#""content":"hello"}"#] {
            let items = a.push(call(None, "", piece)).unwrap();
            assert!(began(&items).is_empty(), "announced once");
            streamed.push_str(&args(&items));
        }
        assert_eq!(streamed, r#"{"path":"/tmp/a.md","content":"hello"}"#);

        // The message ends; the next one starts announcing from scratch.
        let mut end = call(None, "", "");
        end.finish_reason = Some(FinishReason::ToolCall {});
        let items = a.push(end).unwrap();
        assert!(matches!(items.last(), Some(AssembledItem::Completed(_))));
        a.push(delta(Some(Role::Assistant), "", false)).unwrap();
        let again = a
            .push(call(Some("c2"), "shell", r#"{"cmd":"ls"}"#))
            .unwrap();
        assert_eq!(began(&again), [("c2".into(), "shell".into())]);
        assert_eq!(args(&again), r#"{"cmd":"ls"}"#);
    }

    #[test]
    fn only_the_start_of_a_long_argument_is_passed_through() {
        let mut a = MessageAssembler::new();
        a.push(delta(Some(Role::Assistant), "", false)).unwrap();
        a.push(call(Some("c1"), "write", r#"{"path":"a","content":""#))
            .unwrap();
        let mut total = r#"{"path":"a","content":""#.len();
        // Multi-byte text, so the cap has to land on a character boundary.
        for _ in 0..400 {
            let items = a.push(call(None, "", "한글로 된 본문 ")).unwrap();
            total += args(&items).len();
        }
        assert!(total <= ARGS_PREVIEW_MAX);
        assert!(
            total > ARGS_PREVIEW_MAX - 8,
            "up to the cap, less at most one character"
        );
    }

    #[test]
    fn a_sub_agents_calls_are_not_announced() {
        let mut a = MessageAssembler::new();
        let mut d = delta(Some(Role::Assistant), "", false);
        d.depth = Some(1);
        a.push(d).unwrap();
        let mut c = call(Some("c1"), "shell", r#"{"cmd":"ls"}"#);
        c.depth = Some(1);
        let items = a.push(c).unwrap();
        assert!(began(&items).is_empty());
        assert!(args(&items).is_empty());
    }
}
