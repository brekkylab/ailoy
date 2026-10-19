//! A language model that calls the OpenAI Responses API directly over HTTP.
//!
//! Every call is streamed (`"stream": true`) and read as server-sent events;
//! [`infer`](InferLangModel::infer) accumulates the same stream. Nothing is stored on
//! OpenAI's side (`"store": false`): each call sends the whole conversation, reasoning
//! included, as the encrypted items the API returned.

use anyhow::{Context as _, bail};
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    message::{
        Delta as _, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part,
        PartDelta, PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    model::{InferLangModel, LangModelOptions, ThinkingEffort},
    to_value,
    tool::ToolDesc,
};

const API_URL: &str = "https://api.openai.com/v1/responses";

/// Runs OpenAI models through the Responses API (`/v1/responses`).
///
/// Unlike [`CodexCliModel`](super::CodexCliModel), images are sent too. Reasoning is
/// carried across calls as the API's encrypted reasoning items, kept as JSON in the
/// message's `signature`; its `thinking` holds only the summary.
#[derive(Clone)]
pub struct GptApiModel {
    api_key: String,
    model: String,
}

impl std::fmt::Debug for GptApiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptApiModel")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

impl GptApiModel {
    /// Runs `model`, an API model id (e.g. "gpt-6.1-sol"), with `api_key`.
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
        // Unstored reasoning can be replayed only in its encrypted form, which comes back
        // only when asked for. A model that does not reason takes the ask all the same.
        obj.insert("include".into(), to_value!(["reasoning.encrypted_content"]));

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

/// One client for every call, so connections are reused. The read timeout bounds each
/// read, not the whole stream, so a long reply is not cut off.
static CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("the HTTP client builds")
});

/// POSTs `body`, retrying transient 429s and 5xx overloads with backoff, and returns the
/// successful response unread.
async fn send(api_key: &str, body: &serde_json::Value) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    // Rate limits, and the transient errors and overloads (529) servers recover from.
    const RETRY_STATUSES: [u16; 6] = [429, 500, 502, 503, 504, 529];

    for attempt in 0..=MAX_RETRIES {
        let response = CLIENT
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
        if RETRY_STATUSES.contains(&status.as_u16()) && attempt < MAX_RETRIES {
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
                "Request failed with status {status}. Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
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
            let output = val
                .pointer("/response/output")
                .and_then(|v| v.as_array())
                .map(Vec::as_slice)
                .unwrap_or_default();
            // A refusal comes as a content part of its own, in place of the text.
            let refusal = output
                .iter()
                .flat_map(|item| item.pointer("/content").and_then(|v| v.as_array()))
                .flatten()
                .find(|part| part.pointer("/type").and_then(|v| v.as_str()) == Some("refusal"))
                .and_then(|part| part.pointer("/refusal").and_then(|v| v.as_str()));
            out.finish_reason = Some(match refusal {
                Some(reason) => FinishReason::Refusal {
                    reason: reason.to_owned(),
                },
                // Reported for tool calls too; `finish` turns it into a tool call.
                None => FinishReason::Stop {},
            });
            // The reasoning items, kept to be sent back with the next call. Without its
            // encrypted content, an unstored item cannot be.
            let reasoning: Vec<Value> = output
                .iter()
                .filter(|item| item.pointer("/type").and_then(|v| v.as_str()) == Some("reasoning"))
                .filter(|item| {
                    item.pointer("/encrypted_content")
                        .is_some_and(|v| v.is_string())
                })
                .cloned()
                .collect();
            if !reasoning.is_empty() {
                out.delta.signature = serde_json::to_string(&reasoning).ok();
            }
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
/// instead; reasoning goes back as the items it came as, ahead of the reply it led to.
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
                if msg.role == Role::Assistant {
                    items.extend(reasoning_items(msg.signature.as_deref()));
                }
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

pub(super) fn image_url(image: &PartImage) -> String {
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

/// The reasoning items kept in a message's `signature`; none if it holds something else,
/// such as another API's signature.
fn reasoning_items(signature: Option<&str>) -> Vec<Value> {
    signature
        .and_then(|sig| serde_json::from_str::<Vec<Value>>(sig).ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|item| item.pointer("/type").and_then(|v| v.as_str()) == Some("reasoning"))
        .collect()
}

/// Readies a response schema for strict mode, which requires every object sub-schema to be
/// closed and to list all of its properties in `required`.
///
/// Adds `"additionalProperties": false` to every object sub-schema that omits it, and
/// makes every property `required`. A property that was optional becomes nullable
/// instead (`"type": ["integer", "null"]`, or `anyOf` with `{"type": "null"}` where it
/// has no `type`), so the model can still leave it out by sending `null`.
pub(super) fn strict_schema(schema: &Value) -> Value {
    let Value::Object(obj) = schema else {
        return schema.clone();
    };
    let map_values = |v: &Value| match v {
        Value::Object(inner) => Value::Object(
            inner
                .iter()
                .map(|(k, v)| (k.clone(), strict_schema(v)))
                .collect(),
        ),
        other => other.clone(),
    };
    let map_items = |v: &Value| match v {
        Value::Array(items) => Value::Array(items.iter().map(strict_schema).collect()),
        other => other.clone(),
    };
    let mut out: indexmap::IndexMap<String, Value> = obj
        .iter()
        .map(|(k, v)| {
            let v = match k.as_str() {
                "properties" | "$defs" | "definitions" => map_values(v),
                "items" | "not" => strict_schema(v),
                "prefixItems" | "anyOf" | "oneOf" | "allOf" => map_items(v),
                _ => v.clone(),
            };
            (k.clone(), v)
        })
        .collect();
    let is_object = out.get("type").and_then(|t| t.as_str()) == Some("object");
    if (is_object || out.contains_key("properties")) && !out.contains_key("additionalProperties") {
        out.insert("additionalProperties".into(), Value::Bool(false));
    }
    if let Some(Value::Object(props)) = out.get("properties") {
        let required: Vec<&str> = out
            .get("required")
            .and_then(|r| r.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_str())
            .collect();
        let props: indexmap::IndexMap<String, Value> = props
            .iter()
            .map(|(k, v)| {
                let v = if required.contains(&k.as_str()) {
                    v.clone()
                } else {
                    nullable(v)
                };
                (k.clone(), v)
            })
            .collect();
        let all: Vec<Value> = props.keys().map(|k| Value::String(k.clone())).collect();
        out.insert("properties".into(), Value::Object(props));
        out.insert("required".into(), Value::Array(all));
    }
    Value::Object(out)
}

/// `schema`, also accepting `null`.
fn nullable(schema: &Value) -> Value {
    let null = || Value::String("null".into());
    let Value::Object(obj) = schema else {
        return schema.clone();
    };
    let mut out = obj.clone();
    match out.get_mut("type") {
        Some(Value::String(ty)) if ty == "null" => return schema.clone(),
        Some(Value::String(ty)) => {
            let ty = std::mem::take(ty);
            out.insert("type".into(), Value::Array(vec![Value::String(ty), null()]));
        }
        Some(Value::Array(types)) => {
            if !types.iter().any(|t| t.as_str() == Some("null")) {
                types.push(null());
            }
        }
        _ => {
            let wrapped: Value = Value::Array(vec![
                schema.clone(),
                Value::Object([("type".to_owned(), null())].into_iter().collect()),
            ]);
            return Value::Object([("anyOf".to_owned(), wrapped)].into_iter().collect());
        }
    }
    // An `enum` lists every value allowed, so it must list `null` too.
    if let Some(Value::Array(values)) = out.get_mut("enum")
        && !values.iter().any(|v| matches!(v, Value::Null))
    {
        values.push(Value::Null);
    }
    Value::Object(out)
}
