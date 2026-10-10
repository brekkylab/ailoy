//! A language model that calls the OpenAI Responses API directly over HTTP.
//!
//! Every call is streamed (`"stream": true`) and read as server-sent events;
//! [`infer`](InferLangModel::infer) accumulates the same stream. Nothing is stored on
//! OpenAI's side (`"store": false`): each call sends the whole conversation.

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

const API_URL: &str = "https://api.openai.com/v1/responses";

/// Runs OpenAI models through the Responses API (`/v1/responses`).
///
/// Unlike [`CodexCliModel`](super::CodexCliModel), images are sent too. Reasoning is not
/// carried across calls: earlier turns are sent without it.
#[derive(Clone, Debug)]
pub struct GptApiModel {
    api_key: String,
    model: String,
}

impl GptApiModel {
    /// Runs `model`, an API model id (e.g. "gpt-5.5"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `OPENAI_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("OPENAI_API_KEY is not set")?;
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
            "input": input_items(messages),
            "stream": true,
            "store": false,
        });
        let obj = body.as_object_mut().unwrap();

        let instructions = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .collect::<Vec<_>>()
            .join("\n\n");
        if !instructions.is_empty() {
            obj.insert("instructions".into(), instructions.into());
        }

        if !tools.is_empty() {
            let tools = tools
                .iter()
                .map(|tool| {
                    let mut out = to_value!({
                        "type": "function",
                        "name": &tool.name,
                        "parameters": tool.parameters.clone(),
                    });
                    if let Some(desc) = &tool.description {
                        out.as_object_mut()
                            .unwrap()
                            .insert("description".into(), desc.into());
                    }
                    out
                })
                .collect();
            obj.insert("tools".into(), Value::Array(tools));
            obj.insert("tool_choice".into(), "auto".into());
        }

        // Unset keeps the model's default; a model that cannot reason rejects it.
        if let Some(effort) = options.thinking_effort {
            let effort = match effort {
                ThinkingEffort::Low => "low",
                ThinkingEffort::Medium => "medium",
                ThinkingEffort::High => "high",
            };
            // The reasoning comes back only as a summary, and only when one is asked for.
            obj.insert(
                "reasoning".into(),
                to_value!({"effort": effort, "summary": "auto"}),
            );
        }

        if let Some(schema) = &options.output_schema {
            obj.insert(
                "text".into(),
                to_value!({
                    "format": {
                        "type": "json_schema",
                        "name": "output",
                        "strict": true,
                        "schema": strict_schema(schema),
                    }
                }),
            );
        }

        body.into()
    }
}

impl InferLangModel for GptApiModel {
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
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
                if at_eof {
                    break;
                }
            }

            if !saw_finish {
                Err(anyhow::anyhow!("the OpenAI stream ended before the response completed"))?;
            }
        })
    }
}

/// POSTs `body`, retrying transient 429s with backoff, and returns the successful response
/// unread.
async fn send(api_key: &str, body: &serde_json::Value) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;

    let client = reqwest::Client::new();
    for attempt in 0..=MAX_RETRIES {
        let response = client
            .post(API_URL)
            .bearer_auth(api_key)
            .header("accept", "text/event-stream")
            .json(body)
            .send()
            .await?;

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
            let text = response.text().await.unwrap_or_default();
            // Running out of quota or credit does not pass with time.
            if text.contains("insufficient_quota") {
                bail!("OpenAI API request failed with status {status}: {text}");
            }
            log::warn!(
                "Rate limited (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("OpenAI API request failed with status {status}: {text}");
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

/// Parses one `response.*` event into a delta; lifecycle, empty and unknown events give
/// `None`.
fn parse_event(data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let val: Value = serde_json::from_str(data)?;
    let str_at = |ptr: &str| val.pointer(ptr).and_then(|v| v.as_str());

    // Every delta carries the role, so a reply of reasoning or calls alone still has one.
    let mut out = MessageDeltaOutput::new();
    out.delta.role = Some(Role::Assistant);
    match str_at("/type").unwrap_or("") {
        "response.output_text.delta" => {
            out.delta = MessageDelta::new()
                .with_role(Role::Assistant)
                .with_contents([PartDelta::Text {
                    text: str_at("/delta").unwrap_or_default().to_owned(),
                }]);
        }
        "response.reasoning_summary_text.delta" => {
            out.delta.thinking = str_at("/delta").map(str::to_owned);
        }
        // A function call is taken whole once done, with its full arguments.
        "response.output_item.done" if str_at("/item/type") == Some("function_call") => {
            out.delta = MessageDelta::new()
                .with_role(Role::Assistant)
                .with_tool_calls([PartDelta::Function {
                    id: str_at("/item/call_id").map(str::to_owned),
                    function: PartDeltaFunction::WithStringArgs {
                        name: str_at("/item/name").unwrap_or_default().to_owned(),
                        arguments: str_at("/item/arguments").unwrap_or_default().to_owned(),
                    },
                }]);
        }
        "response.completed" => {
            // Reported for tool calls too; `finish` turns it into a tool call.
            out.finish_reason = Some(FinishReason::Stop {});
            out.usage = val.pointer("/response/usage").and_then(parse_usage);
        }
        "response.incomplete" => {
            out.finish_reason = Some(match str_at("/response/incomplete_details/reason") {
                Some("max_output_tokens") => FinishReason::Length {},
                Some(other) => FinishReason::Refusal {
                    reason: format!("reason: {other}"),
                },
                None => FinishReason::Refusal {
                    reason: "reason: unknown".to_owned(),
                },
            });
            out.usage = val.pointer("/response/usage").and_then(parse_usage);
        }
        "response.failed" => bail!(
            "OpenAI response failed: {}",
            str_at("/response/error/message").unwrap_or("(no message)")
        ),
        "error" => bail!(
            "OpenAI stream error: {}",
            str_at("/message")
                .or_else(|| str_at("/error/message"))
                .unwrap_or("(no message)")
        ),
        _ => return Ok(None),
    }
    Ok(Some(out))
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let u = usage.as_object()?;
    let count = |ptr: &str| {
        usage
            .pointer(ptr)
            .and_then(|v| v.as_integer())
            .map(|v| v as u64)
    };
    Some(TokenUsage {
        input_tokens: u
            .get("input_tokens")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as u64,
        output_tokens: u
            .get("output_tokens")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as u64,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: count("/input_tokens_details/cached_tokens"),
    })
}

/// The conversation as Responses API input items. System messages go in `instructions`
/// instead; earlier reasoning is left out, since it cannot be replayed without being stored.
fn input_items(messages: &[Message]) -> Value {
    let mut items = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {}
            Role::Tool => {
                // ids were checked before marshaling.
                let output: Vec<Value> = msg.contents.iter().map(tool_output_part).collect();
                items.push(to_value!({
                    "type": "function_call_output",
                    "call_id": msg.id.clone().unwrap_or_default(),
                    "output": output,
                }));
            }
            Role::User | Role::Assistant => {
                if !msg.contents.is_empty() {
                    let content: Vec<Value> = msg
                        .contents
                        .iter()
                        .map(|part| content_part(part, msg.role == Role::Assistant))
                        .collect();
                    items.push(to_value!({"role": msg.role.to_string(), "content": content}));
                }
                for call in msg.tool_calls.iter().flatten() {
                    if let Part::Function {
                        id,
                        function: PartFunction { name, arguments },
                    } = call
                    {
                        items.push(to_value!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                        }));
                    }
                }
            }
        }
    }
    Value::Array(items)
}

/// A message part as Responses API content; text the model wrote is `output_text`.
fn content_part(part: &Part, from_model: bool) -> Value {
    match part {
        Part::Text { text } => {
            let ty = if from_model {
                "output_text"
            } else {
                "input_text"
            };
            to_value!({"type": ty, "text": text})
        }
        Part::Image { image } => to_value!({"type": "input_image", "image_url": image_url(image)}),
        Part::Value { value } => to_value!({"type": "input_text", "text": value_text(value)}),
        Part::Function {
            function: PartFunction { name, arguments },
            ..
        } => to_value!({
            "type": "input_text",
            "text": format!("{name}({})", serde_json::to_string(arguments).unwrap_or_default()),
        }),
    }
}

/// A tool result part as `function_call_output` content.
fn tool_output_part(part: &Part) -> Value {
    match part {
        Part::Text { text } => to_value!({"type": "input_text", "text": text}),
        Part::Image { image } => to_value!({"type": "input_image", "image_url": image_url(image)}),
        Part::Value { value } => to_value!({"type": "input_text", "text": value_text(value)}),
        Part::Function { .. } => to_value!({"type": "input_text", "text": ""}),
    }
}

fn image_url(image: &PartImage) -> String {
    match image {
        PartImage::Embedded { mime_type, data } => {
            format!("data:{mime_type};base64,{}", data.base64())
        }
        PartImage::Url { url } => url.clone(),
    }
}

/// A value as text, strings without their quotes.
fn value_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Closes every object sub-schema (`"additionalProperties": false`) that does not say
/// otherwise, as strict mode requires.
fn strict_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(obj) => {
            let mut out: indexmap::IndexMap<String, Value> = obj
                .iter()
                .map(|(k, v)| {
                    let v = match k.as_str() {
                        // Maps of sub-schemas.
                        "properties" | "$defs" | "definitions" => match v {
                            Value::Object(inner) => Value::Object(
                                inner
                                    .iter()
                                    .map(|(k, v)| (k.clone(), strict_schema(v)))
                                    .collect(),
                            ),
                            other => other.clone(),
                        },
                        // A sub-schema, or a list of them.
                        "items" | "not" | "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                            strict_schema(v)
                        }
                        _ => v.clone(),
                    };
                    (k.clone(), v)
                })
                .collect();
            let is_object = out.get("type").and_then(|t| t.as_str()) == Some("object");
            if (is_object || out.contains_key("properties"))
                && !out.contains_key("additionalProperties")
            {
                out.insert("additionalProperties".into(), Value::Bool(false));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(strict_schema).collect()),
        other => other.clone(),
    }
}
