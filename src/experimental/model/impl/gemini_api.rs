//! A language model that calls the Gemini API (`generativelanguage.googleapis.com`)
//! directly over HTTP.
//!
//! Every call is streamed (`:streamGenerateContent?alt=sse`) and read as server-sent
//! events; [`infer`](InferLangModel::infer) accumulates the same stream.

use std::collections::HashMap;

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
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    to_value,
    tool::ToolDesc,
};

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Runs Gemini through the Gemini API.
///
/// Unlike [`GeminiCliModel`](super::GeminiCliModel), images are sent too, as inline data;
/// image URLs other than `data:` URIs are rejected.
#[derive(Clone, Debug)]
pub struct GeminiApiModel {
    api_key: String,
    model: String,
}

impl GeminiApiModel {
    /// Runs `model`, an API model id (e.g. "gemini-3.5-flash"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `GEMINI_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("GEMINI_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body = to_value!({"contents": contents(messages)?});
        let obj = body.as_object_mut().unwrap();

        let system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .collect::<Vec<_>>()
            .join("\n\n");
        if !system.is_empty() {
            obj.insert(
                "systemInstruction".into(),
                to_value!({"parts": [{"text": system}]}),
            );
        }

        if !tools.is_empty() {
            let declarations: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let mut out = to_value!({
                        "name": &tool.name,
                        "parametersJsonSchema": tool.parameters.clone(),
                    });
                    if let Some(desc) = &tool.description {
                        out.as_object_mut()
                            .unwrap()
                            .insert("description".into(), desc.into());
                    }
                    out
                })
                .collect();
            obj.insert(
                "tools".into(),
                to_value!([{"functionDeclarations": declarations}]),
            );
        }

        let mut config = Value::object_empty();
        let config_obj = config.as_object_mut().unwrap();
        // Unset keeps the model's default.
        if let Some(effort) = options.thinking_effort {
            // Gemini 2.x takes a token budget, later models a level. Thoughts come back
            // only when asked for.
            let thinking = if self.model.starts_with("gemini-2") {
                let budget: i64 = match effort {
                    ThinkingEffort::Low => 2048,
                    ThinkingEffort::Medium => 8192,
                    ThinkingEffort::High => 24576,
                };
                to_value!({"thinkingBudget": budget, "includeThoughts": true})
            } else {
                let level = match effort {
                    ThinkingEffort::Low => "low",
                    ThinkingEffort::Medium => "medium",
                    ThinkingEffort::High => "high",
                };
                to_value!({"thinkingLevel": level, "includeThoughts": true})
            };
            config_obj.insert("thinkingConfig".into(), thinking);
        }
        if let Some(schema) = &options.output_schema {
            config_obj.insert("responseMimeType".into(), "application/json".into());
            config_obj.insert("responseJsonSchema".into(), schema.clone());
        }
        if !config_obj.is_empty() {
            obj.insert("generationConfig".into(), config);
        }

        Ok(body.into())
    }
}

impl InferLangModel for GeminiApiModel {
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
        let body = match self.body(messages, tools, options) {
            Ok(body) => body,
            Err(e) => return Box::pin(stream::once(async move { Err(e) })),
        };
        let url = format!("{API_BASE}/{}:streamGenerateContent?alt=sse", self.model);
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let response = send(&url, &api_key, &body).await?;

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
                    if let Some(out) = parse_chunk(&data)? {
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
                if at_eof {
                    break;
                }
            }

            // Every message ends with a finish_reason, even if the stream closed without one.
            if !saw_finish {
                let mut closer = MessageDeltaOutput::new();
                closer.delta.role = Some(Role::Assistant);
                closer.finish_reason = Some(FinishReason::Stop {});
                yield closer;
            }
        })
    }
}

/// POSTs `body`, retrying transient 429s with backoff, and returns the successful response
/// unread.
async fn send(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;

    let client = reqwest::Client::new();
    for attempt in 0..=MAX_RETRIES {
        let response = client
            .post(url)
            .header("x-goog-api-key", api_key)
            .header("accept", "text/event-stream")
            .json(body)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if status.as_u16() == 429 && attempt < MAX_RETRIES {
            let text = response.text().await.unwrap_or_default();
            // A quota that is used up comes without a RetryInfo, and does not pass with time.
            if !text.contains("google.rpc.RetryInfo") {
                bail!("Gemini API request failed with status {status}: {text}");
            }
            let wait_secs = (1u64 << attempt).min(MAX_WAIT_SECS);
            log::warn!(
                "Rate limited (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("Gemini API request failed with status {status}: {text}");
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

/// Parses one streamed `GenerateContentResponse` into a delta; a chunk without a candidate
/// gives `None`.
fn parse_chunk(data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
    if data.is_empty() {
        return Ok(None);
    }
    let val: Value = serde_json::from_str(data)?;
    if let Some(reason) = val
        .pointer("/promptFeedback/blockReason")
        .and_then(|v| v.as_str())
    {
        bail!("Gemini blocked the prompt: {reason}");
    }
    let Some(candidate) = val.pointer("/candidates/0") else {
        return Ok(None);
    };

    let mut out = MessageDeltaOutput::new();
    // Candidates are always the model's; the role is set even on a chunk without content.
    out.delta.role = Some(Role::Assistant);
    for part in candidate
        .pointer("/content/parts")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        if let Some(sig) = part.pointer("/thoughtSignature").and_then(|v| v.as_str()) {
            out.delta.signature = Some(sig.to_owned());
        }
        let is_thought = part.pointer("/thought").and_then(|v| v.as_bool()) == Some(true);
        if let Some(text) = part.pointer("/text").and_then(|v| v.as_str()) {
            if is_thought {
                out.delta.thinking = Some(out.delta.thinking.unwrap_or_default() + text);
            } else if !text.is_empty() {
                out.delta = out.delta.with_contents([PartDelta::Text {
                    text: text.to_owned(),
                }]);
            }
        } else if let Some(call) = part.pointer("/functionCall") {
            // Newer models give each call an id; otherwise one is made up.
            let id = call
                .pointer("/id")
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    format!("call-{}", &uuid::Uuid::new_v4().simple().to_string()[..12])
                });
            out.delta = out.delta.with_tool_calls([PartDelta::Function {
                id: Some(id),
                function: PartDeltaFunction::WithParsedArgs {
                    name: call
                        .pointer("/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call.pointer("/args").cloned().unwrap_or(Value::Null),
                },
            }]);
        }
    }

    // Reported as STOP for tool calls too; `finish` turns it into a tool call.
    out.finish_reason = candidate
        .pointer("/finishReason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "STOP" => FinishReason::Stop {},
            "MAX_TOKENS" => FinishReason::Length {},
            other => FinishReason::Refusal {
                reason: other.to_owned(),
            },
        });
    // Every chunk reports the usage so far; the last one is the total.
    if out.finish_reason.is_some() {
        out.usage = val.pointer("/usageMetadata").and_then(parse_usage);
    }
    Ok(Some(out))
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let count = |key: &str| {
        usage
            .pointer(key)
            .and_then(|v| v.as_integer())
            .map(|v| v as u64)
    };
    usage.as_object()?;
    Some(TokenUsage {
        input_tokens: count("/promptTokenCount").unwrap_or(0),
        // Thinking is billed as output.
        output_tokens: count("/candidatesTokenCount").unwrap_or(0)
            + count("/thoughtsTokenCount").unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: count("/cachedContentTokenCount"),
    })
}

/// The conversation as Gemini `contents`. System messages go in `systemInstruction`
/// instead; consecutive tool results go in one user turn, as the API expects.
fn contents(messages: &[Message]) -> anyhow::Result<Value> {
    // A function response names its function, which the tool message knows only by call id.
    let call_names: HashMap<&str, &str> = messages
        .iter()
        .flat_map(|m| m.tool_calls.iter().flatten())
        .filter_map(|call| match call {
            Part::Function { id, function } => Some((id.as_str(), function.name.as_str())),
            _ => None,
        })
        .collect();

    let mut contents: Vec<Value> = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {}
            Role::Tool => {
                let id = msg
                    .id
                    .as_deref()
                    .context("a tool message has no id to match its call")?;
                let name = call_names
                    .get(id)
                    .with_context(|| format!("no tool call with id {id} precedes its result"))?;
                let parts = function_response_parts(id, name, &msg.contents)?;
                // Joins the previous tool results, if this follows them.
                match contents.last_mut() {
                    Some(last) if is_function_responses(last) => {
                        let list = last
                            .as_object_mut()
                            .unwrap()
                            .get_mut("parts")
                            .unwrap()
                            .as_array_mut()
                            .unwrap();
                        list.extend(parts);
                    }
                    _ => contents.push(to_value!({"role": "user", "parts": parts})),
                }
            }
            Role::User | Role::Assistant => {
                let mut parts: Vec<Value> = msg
                    .contents
                    .iter()
                    .map(content_part)
                    .collect::<anyhow::Result<_>>()?;
                let mut calls: Vec<Value> = msg
                    .tool_calls
                    .iter()
                    .flatten()
                    .filter_map(|call| match call {
                        Part::Function {
                            id,
                            function: PartFunction { name, arguments },
                        } => Some(to_value!({
                            "functionCall": {"id": id, "name": name, "args": arguments.clone()}
                        })),
                        _ => None,
                    })
                    .collect();
                // The signature goes back where it came from: the first call, or else the text.
                if let Some(sig) = &msg.signature {
                    let target = calls.first_mut().or(parts.last_mut());
                    if let Some(part) = target {
                        part.as_object_mut()
                            .unwrap()
                            .insert("thoughtSignature".into(), sig.into());
                    }
                }
                parts.extend(calls);
                if parts.is_empty() {
                    continue;
                }
                let role = if msg.role == Role::Assistant {
                    "model"
                } else {
                    "user"
                };
                contents.push(to_value!({"role": role, "parts": parts}));
            }
        }
    }
    Ok(Value::Array(contents))
}

/// Whether `content` is a user turn of function responses only.
fn is_function_responses(content: &Value) -> bool {
    content.pointer("/role").and_then(|v| v.as_str()) == Some("user")
        && content
            .pointer("/parts")
            .and_then(|v| v.as_array())
            .is_some_and(|parts| {
                parts.iter().all(|p| {
                    p.pointer("/functionResponse").is_some() || p.pointer("/inlineData").is_some()
                })
            })
}

/// A tool result as a `functionResponse`, with its images as inline data beside it.
fn function_response_parts(id: &str, name: &str, contents: &[Part]) -> anyhow::Result<Vec<Value>> {
    let mut texts = Vec::new();
    let mut images = Vec::new();
    for part in contents {
        match part {
            Part::Text { text } => texts.push(text.clone()),
            Part::Value { value } => texts.push(match value {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }),
            Part::Image { image } => images.push(inline_image(image)?),
            Part::Function { .. } => {}
        }
    }
    let mut parts = vec![to_value!({
        "functionResponse": {
            "id": id,
            "name": name,
            "response": {"result": texts.join("\n")},
        }
    })];
    parts.extend(images);
    Ok(parts)
}

fn content_part(part: &Part) -> anyhow::Result<Value> {
    Ok(match part {
        Part::Text { text } => to_value!({"text": text}),
        Part::Image { image } => inline_image(image)?,
        Part::Value { value } => to_value!({"text": match value {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }}),
        Part::Function {
            function: PartFunction { name, arguments },
            ..
        } => to_value!({"functionCall": {"name": name, "args": arguments.clone()}}),
    })
}

/// An image as `inlineData`. The API takes no image URLs, so only `data:` URIs pass.
fn inline_image(image: &PartImage) -> anyhow::Result<Value> {
    let (mime_type, data) = match image {
        PartImage::Embedded { mime_type, data } => (mime_type.clone(), data.base64()),
        PartImage::Url { url } => {
            let rest = url
                .strip_prefix("data:")
                .with_context(|| format!("Gemini takes no image URLs, only data URIs: {url}"))?;
            let (meta, data) = rest
                .split_once(',')
                .with_context(|| format!("malformed data URI: {url}"))?;
            let mime_type = meta
                .strip_suffix(";base64")
                .with_context(|| format!("only base64 data URIs are supported: {url}"))?;
            (mime_type.to_owned(), data.to_owned())
        }
    };
    Ok(to_value!({"inlineData": {"mimeType": mime_type, "data": data}}))
}
