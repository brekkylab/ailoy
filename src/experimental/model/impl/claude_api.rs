//! A language model that calls the Anthropic Messages API directly over HTTP.
//!
//! Every call is streamed (`"stream": true`) and read as server-sent events;
//! [`infer`](InferLangModel::infer) accumulates the same stream.

use anyhow::{Context as _, bail};
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    experimental::model::{InferLangModel, LangModelOptions, ThinkingEffort},
    message::{
        Delta as _, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part,
        PartDelta, PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    to_value,
    tool::ToolDesc,
};

use super::schema::close_objects;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// The API requires `max_tokens`; this is the one sent. It covers thinking too, and
/// stays above the largest thinking budget.
const MAX_TOKENS: u64 = 32000;

/// The API's name for `effort`.
fn effort_str(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
    }
}

/// Thinking budget for models before Claude 4.6, which take one instead of an effort.
fn budget_tokens(effort: ThinkingEffort) -> u64 {
    match effort {
        ThinkingEffort::Low => 2048,
        ThinkingEffort::Medium => 8192,
        ThinkingEffort::High => 24576,
    }
}

/// The API id for one of the `claude` CLI's aliases, which name the latest model of a
/// family, so the same name works through either; any other name as is.
fn resolve_alias(model: String) -> String {
    match model.as_str() {
        "fable" => "claude-fable-5-1".to_owned(),
        "opus" => "claude-opus-5-5".to_owned(),
        "sonnet" => "claude-sonnet-5-5".to_owned(),
        "haiku" => "claude-haiku-4-5-20251001".to_owned(),
        _ => model,
    }
}

/// Runs Claude through the Anthropic Messages API (`/v1/messages`).
///
/// Unlike [`ClaudeModel`](super::ClaudeModel), tools are declared to the API
/// and the full message history, images included, is sent as is.
#[derive(Clone, Debug)]
pub struct ClaudeApiModel {
    api_key: String,
    model: String,
}

impl ClaudeApiModel {
    /// Runs `model`, an API model id (e.g. "claude-opus-5-5"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: resolve_alias(model.into()),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `ANTHROPIC_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("ANTHROPIC_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The streamed request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> serde_json::Value {
        let mut body = to_value!({
            "model": self.model.as_str(),
            "messages": marshal_messages(messages),
            "stream": true,
        });
        let obj = body.as_object_mut().unwrap();

        let system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .collect::<Vec<_>>()
            .join("\n\n");
        if !system.is_empty() {
            obj.insert("system".into(), system.into());
        }

        if !tools.is_empty() {
            obj.insert(
                "tools".into(),
                Value::Array(tools.iter().map(marshal_tool).collect()),
            );
            obj.insert("tool_choice".into(), to_value!({"type": "auto"}));
        }

        let mut output_config = Value::object_empty();
        obj.insert("max_tokens".into(), (MAX_TOKENS as i64).into());
        // Without an effort, thinking stays off, the API default.
        if let Some(effort) = options.thinking_effort {
            if takes_thinking_budget(&self.model) {
                // The budget must be below max_tokens, which every level's is.
                let budget = budget_tokens(effort) as i64;
                obj.insert(
                    "thinking".into(),
                    to_value!({"type": "enabled", "budget_tokens": budget}),
                );
            } else {
                // Newer models omit the thinking text unless a summary is asked for.
                obj.insert(
                    "thinking".into(),
                    to_value!({"type": "adaptive", "display": "summarized"}),
                );
                output_config
                    .as_object_mut()
                    .unwrap()
                    .insert("effort".into(), effort_str(effort).into());
            }
        }

        if let Some(schema) = &options.output_schema {
            output_config.as_object_mut().unwrap().insert(
                "format".into(),
                to_value!({"type": "json_schema", "schema": close_objects(schema)}),
            );
        }
        if !output_config.as_object().unwrap().is_empty() {
            obj.insert("output_config".into(), output_config);
        }

        body.into()
    }
}

impl InferLangModel for ClaudeApiModel {
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let mut deltas = self.infer_stream(messages, tools, options);
        Box::pin(async move {
            let mut acc = MessageDeltaOutput::new();
            while let Some(delta) = deltas.next().await {
                acc = acc.accumulate(delta?)?;
            }
            acc.finish()
        })
    }

    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        if messages
            .iter()
            .any(|m| m.role == Role::Tool && m.id.is_none())
        {
            let e = anyhow::anyhow!("a tool message has no id to match its call");
            return Box::pin(stream::once(async move { Err(e) }));
        }
        let body = self.body(messages, tools, options);
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let response = send(&api_key, &body).await?;

            let mut seen_role = None;
            let mut saw_finish = false;
            let mut bytes = response.bytes_stream();
            let mut buf = Vec::new();
            // Network chunks don't align with events, so events are cut out of a buffer.
            loop {
                let chunk = bytes.next().await.transpose()?;
                let at_eof = chunk.is_none();
                match chunk {
                    Some(chunk) => buf.extend_from_slice(&chunk),
                    // A final event may lack its trailing blank line.
                    None => buf.extend_from_slice(b"\n\n"),
                }
                while let Some(data) = next_event(&mut buf) {
                    if let Some(out) = parse_event(&data)? {
                        if seen_role.is_none() {
                            seen_role = out.delta.role.clone();
                        }
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
                if at_eof {
                    break;
                }
            }

            // Every message ends with a finish_reason, even if the stream closed without one.
            // Not a let-chain: `try_stream!` rejects them.
            #[allow(clippy::collapsible_if)]
            if !saw_finish {
                if let Some(role) = seen_role {
                    let mut closer = MessageDeltaOutput::new();
                    closer.delta.role = Some(role);
                    closer.finish_reason = Some(FinishReason::Stop {});
                    yield closer;
                }
            }
        })
    }
}

/// POSTs `body`, retrying 429s with backoff, and returns the successful response unread.
async fn send(api_key: &str, body: &serde_json::Value) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;

    let client = reqwest::Client::new();
    for attempt in 0..=MAX_RETRIES {
        let request = client
            .post(API_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", API_VERSION)
            .header("accept", "text/event-stream")
            .json(body);
        #[cfg(target_arch = "wasm32")]
        let request = request.header("anthropic-dangerous-direct-browser-access", "true");
        let response = request.send().await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if status.as_u16() == 429 && attempt < MAX_RETRIES {
            let wait_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1u64 << attempt)
                .min(MAX_WAIT_SECS);
            log::warn!(
                "Rate limited (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("Anthropic API request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Cuts the next complete SSE event out of `buf` and returns its joined `data:` lines;
/// `None` until a blank line ends one. Events without data come back empty.
fn next_event(buf: &mut Vec<u8>) -> Option<String> {
    let (pos, len) = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| (p, 2))
        .or_else(|| {
            buf.windows(4)
                .position(|w| w == b"\r\n\r\n")
                .map(|p| (p, 4))
        })?;
    let raw: Vec<u8> = buf.drain(..pos + len).collect();
    Some(
        String::from_utf8_lossy(&raw)
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(|data| data.strip_prefix(' ').unwrap_or(data))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// Parses one stream event into a delta; control, empty and unknown events give `None`.
fn parse_event(data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
    if data.is_empty() {
        return Ok(None);
    }
    let val: Value = serde_json::from_str(data)?;
    let str_at = |ptr: &str| val.pointer(ptr).and_then(|v| v.as_str());

    let mut out = MessageDeltaOutput::new();
    match str_at("/type").unwrap_or("") {
        "message_start" => {
            out.delta.role = Some(Role::Assistant);
            out.usage = val.pointer("/message/usage").and_then(parse_usage);
        }
        "content_block_start" => {
            if str_at("/content_block/type") != Some("tool_use") {
                return Ok(None);
            }
            out.delta = MessageDelta::new().with_tool_calls([PartDelta::Function {
                id: str_at("/content_block/id").map(str::to_owned),
                function: PartDeltaFunction::WithStringArgs {
                    name: str_at("/content_block/name").unwrap_or_default().to_owned(),
                    arguments: String::new(),
                },
            }]);
        }
        "content_block_delta" => match str_at("/delta/type").unwrap_or("") {
            "text_delta" => {
                out.delta = MessageDelta::new().with_contents([PartDelta::Text {
                    text: str_at("/delta/text").unwrap_or_default().to_owned(),
                }]);
            }
            "thinking_delta" => out.delta.thinking = str_at("/delta/thinking").map(str::to_owned),
            "signature_delta" => {
                out.delta.signature = str_at("/delta/signature").map(str::to_owned)
            }
            "input_json_delta" => {
                out.delta = MessageDelta::new().with_tool_calls([PartDelta::Function {
                    id: None,
                    function: PartDeltaFunction::WithStringArgs {
                        name: String::new(),
                        arguments: str_at("/delta/partial_json").unwrap_or_default().to_owned(),
                    },
                }]);
            }
            _ => return Ok(None),
        },
        "message_delta" => {
            out.finish_reason = str_at("/delta/stop_reason").map(parse_finish_reason);
            out.usage = val.pointer("/usage").and_then(parse_usage);
        }
        "error" => bail!(
            "Anthropic stream error ({}): {}",
            str_at("/error/type").unwrap_or("unknown"),
            str_at("/error/message").unwrap_or("(no message)"),
        ),
        // `ping`, `content_block_stop`, `message_stop`, and types added later.
        _ => return Ok(None),
    }
    Ok(Some(out))
}

fn parse_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "pause_turn" | "stop_sequence" => FinishReason::Stop {},
        "max_tokens" => FinishReason::Length {},
        "tool_use" => FinishReason::ToolCall {},
        other => FinishReason::Refusal {
            reason: format!("reason: {other}"),
        },
    }
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let u = usage.as_object()?;
    let count = |key: &str| u.get(key).and_then(|v| v.as_integer()).map(|v| v as u64);
    Some(TokenUsage {
        input_tokens: count("input_tokens").unwrap_or(0),
        output_tokens: count("output_tokens").unwrap_or(0),
        cache_creation_input_tokens: count("cache_creation_input_tokens"),
        cache_read_input_tokens: count("cache_read_input_tokens"),
    })
}

/// Messages in the API's form. System messages go in the top-level `system` instead.
/// Thinking is replayed only on assistant turns after the last user message, as the API
/// requires.
fn marshal_messages(messages: &[Message]) -> Value {
    let last_user = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len());
    Value::Array(
        messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .map(|(i, m)| marshal_message(m, i > last_user))
            .collect(),
    )
}

fn marshal_message(msg: &Message, include_thinking: bool) -> Value {
    if msg.role == Role::Tool {
        // A tool result is sent as a user turn; ids were checked before marshaling.
        let content: Vec<Value> = msg
            .contents
            .iter()
            .map(|part| match part {
                Part::Value { value } => {
                    let text = match value {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    to_value!({"type": "text", "text": text})
                }
                other => marshal_part(other),
            })
            .collect();
        return to_value!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.id.clone().unwrap_or_default(),
                "content": content,
            }],
        });
    }

    let mut content = Vec::new();
    if include_thinking && let Some(thinking) = msg.thinking.as_ref().filter(|t| !t.is_empty()) {
        let mut part = to_value!({"type": "thinking", "thinking": thinking});
        if let Some(sig) = &msg.signature {
            part.as_object_mut()
                .unwrap()
                .insert("signature".into(), sig.into());
        }
        content.push(part);
    }
    content.extend(msg.contents.iter().map(marshal_part));
    content.extend(msg.tool_calls.iter().flatten().map(marshal_part));
    to_value!({"role": msg.role.to_string(), "content": content})
}

fn marshal_part(part: &Part) -> Value {
    match part {
        Part::Text { text } => to_value!({"type": "text", "text": text}),
        Part::Function {
            id,
            function: PartFunction { name, arguments },
        } => to_value!({"type": "tool_use", "id": id, "name": name, "input": arguments.clone()}),
        Part::Value { value } => value.clone(),
        Part::Image {
            image: PartImage::Embedded { mime_type, data },
        } => to_value!({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data.base64()},
        }),
        Part::Image {
            image: PartImage::Url { url },
        } => to_value!({"type": "image", "source": {"type": "url", "url": url}}),
    }
}

fn marshal_tool(tool: &ToolDesc) -> Value {
    let mut out = to_value!({"name": &tool.name, "input_schema": tool.parameters.clone()});
    if let Some(desc) = &tool.description {
        out.as_object_mut()
            .unwrap()
            .insert("description".into(), desc.into());
    }
    out
}

/// Whether a model predates adaptive thinking (Claude 4.6) and thinks only on a
/// `budget_tokens` budget. Reads `claude-opus-5-5`, dated `claude-haiku-4-5-20251001` and
/// legacy `claude-3-7-sonnet-…` ids; an unreadable id counts as newer.
fn takes_thinking_budget(model: &str) -> bool {
    let Some((_, rest)) = model.split_once("claude-") else {
        return false;
    };
    let is_num = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut tokens = rest.split('-').skip_while(|t| !is_num(t));
    let Some(major) = tokens.next().and_then(|t| t.parse::<u32>().ok()) else {
        return false;
    };
    // A date (`20250514`) after the major version is no minor version.
    let minor = tokens
        .next()
        .filter(|t| is_num(t) && t.len() <= 2)
        .and_then(|t| t.parse::<u32>().ok())
        .unwrap_or(0);
    (major, minor) < (4, 6)
}
