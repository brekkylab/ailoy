use std::fmt;

use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::message::{Delta, FinishReason, Message, MessageOutput, PartDelta, Role, TokenUsage};

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

        // Merge role
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

        // Merge ID
        if let Some(id_incoming) = other.id {
            id = Some(id_incoming);
        }

        // Merge think
        if let Some(thinking_rhs) = other.thinking {
            if let Some(mut thinking_lhs) = thinking {
                thinking_lhs.push_str(&thinking_rhs);
                thinking = Some(thinking_lhs);
            } else {
                thinking = Some(thinking_rhs);
            }
        }

        // Merge content
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

        // Merge tool calls.
        //
        // Assumes the provider streams calls *sequentially* — all fragments of
        // one call before the next — with the call `id` on each call's first
        // fragment; continuation fragments have `id = None` and merge into the
        // last call. True for OpenAI and Anthropic. A backend that interleaves
        // calls by `index`, or omits the id on a new call's first fragment,
        // would misroute argument fragments into the wrong call (the
        // `tool_calls[].index` field is currently ignored). If that ever
        // happens, carry `index` through `PartDelta::Function` and match on it
        // here instead of comparing against `last()`.
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

        // Merge signature
        if let Some(sig_incoming) = other.signature {
            signature = Some(sig_incoming);
        }

        // Return
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

/// A container for a streamed message delta and its termination signal.
///
/// During streaming, `delta` carries the incremental payload; once a terminal
/// condition is reached, `finish_reason` may be populated to explain why.
///
/// # Examples
/// ```rust
/// # use ailoy::message::{MessageDeltaOutput, MessageDelta, PartDelta, Role};
/// let mut out = MessageDeltaOutput::new();
/// out.delta = MessageDelta::new().with_role(Role::Assistant).with_contents([PartDelta::Text { text: "Hi".into() }]);
/// assert!(out.finish_reason.is_none());
/// ```
///
/// # Lifecycle
/// - While streaming: `finish_reason` is typically `None`.
/// - On completion: `finish_reason` is set; callers can then `finish()` the delta to obtain a concrete [`Message`].
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
        let finish_reason = self
            .finish_reason
            .ok_or_else(|| anyhow::anyhow!("finish_reason not specified"))?;
        Ok(MessageOutput {
            message,
            finish_reason,
            usage: self.usage,
            depth: None,
            source_agent: None,
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
/// Providers spread usage across stream events — e.g. Anthropic reports
/// `input_tokens` (and cache tokens) once in `message_start` and the final
/// cumulative `output_tokens` in `message_delta`. A whole-value replace would
/// drop whichever the latest event omits, so each counter is merged
/// independently; `max` is correct because these counters are monotonic
/// (constant input, cumulative output).
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
