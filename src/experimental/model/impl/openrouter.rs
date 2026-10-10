//! A language model served by OpenRouter, called through its OpenAI-compatible Chat
//! Completions API.
//!
//! Every call is streamed (`"stream": true`) and read as server-sent events;
//! [`infer`](InferLangModel::infer) accumulates the same stream.
//!
//! Some models need their reasoning sent back with the turn it came from, Gemini's thought
//! signatures above all. OpenRouter returns it as `reasoning_details`, which is kept, as
//! JSON, in the message's [`signature`](Message::signature) and sent back from there.

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

const API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Runs a model through OpenRouter.
#[derive(Clone, Debug)]
pub struct OpenRouterModel {
    model: String,
    api_key: String,
}

impl OpenRouterModel {
    /// Runs `model`, an OpenRouter model id (e.g. `"anthropic/claude-sonnet-4.5"`), with
    /// `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `OPENROUTER_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("OPENROUTER_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The streamed request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body = to_value!({
            "model": self.model.as_str(),
            "messages": chat_messages(messages)?,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        let obj = body.as_object_mut().unwrap();

        if !tools.is_empty() {
            let tools: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let mut function = to_value!({
                        "name": &tool.name,
                        "parameters": tool.parameters.clone(),
                    });
                    if let Some(desc) = &tool.description {
                        function
                            .as_object_mut()
                            .unwrap()
                            .insert("description".into(), desc.into());
                    }
                    to_value!({"type": "function", "function": function})
                })
                .collect();
            obj.insert("tools".into(), Value::Array(tools));
            obj.insert("tool_choice".into(), "auto".into());
        }

        // Unset keeps the model's default. OpenRouter maps the effort to each model's own
        // control, a level or a token budget.
        if let Some(effort) = options.thinking_effort {
            let effort = match effort {
                ThinkingEffort::Low => "low",
                ThinkingEffort::Medium => "medium",
                ThinkingEffort::High => "high",
            };
            obj.insert("reasoning".into(), to_value!({"effort": effort}));
        }

        if let Some(schema) = &options.output_schema {
            obj.insert(
                "response_format".into(),
                to_value!({
                    "type": "json_schema",
                    "json_schema": {"name": "output", "strict": true, "schema": closed_schema(schema)},
                }),
            );
        }

        Ok(body.into())
    }
}

impl InferLangModel for OpenRouterModel {
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
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let response = send(&api_key, &body).await?;

            let mut saw_finish = false;
            // `reasoning_details` arrive in pieces, and are sent on whole at the end.
            let mut details: Vec<serde_json::Value> = Vec::new();
            let mut bytes = response.bytes_stream();
            let mut buf = Vec::new();
            // Network chunks don't align with events, so events are cut out of a buffer.
            // The stream goes on past the finish reason: the usage comes after it.
            loop {
                let chunk = bytes.next().await.transpose()?;
                let at_eof = chunk.is_none();
                match chunk {
                    Some(chunk) => buf.extend_from_slice(&chunk),
                    // A final event may lack its trailing blank line.
                    None => buf.extend_from_slice(b"\n\n"),
                }
                while let Some(data) = next_event(&mut buf) {
                    if let Some(out) = parse_chunk(&data, &mut details)? {
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
                if at_eof {
                    break;
                }
            }

            if !saw_finish {
                Err(anyhow::anyhow!("the OpenRouter stream ended before the reply finished"))?;
            }
            if !details.is_empty() {
                let mut out = MessageDeltaOutput::new();
                out.delta.role = Some(Role::Assistant);
                out.delta.signature = Some(serde_json::to_string(&details)?);
                yield out;
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
            log::warn!(
                "Rate limited (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("OpenRouter request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Cuts the next complete SSE event out of `buf` and returns its joined `data:` lines;
/// `None` until a blank line ends one. Comments (OpenRouter's keep-alives) and events
/// without data come back empty.
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

/// Parses one stream chunk into a delta, gathering its `reasoning_details` into `details`;
/// chunks that carry nothing else give `None`.
fn parse_chunk(
    data: &str,
    details: &mut Vec<serde_json::Value>,
) -> anyhow::Result<Option<MessageDeltaOutput>> {
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let val: serde_json::Value = serde_json::from_str(data)?;
    if let Some(error) = val.get("error") {
        bail!(
            "OpenRouter stream error: {}",
            error["message"].as_str().unwrap_or("(no message)")
        );
    }

    let mut out = MessageDeltaOutput::new();
    // Every delta carries the role, so a reply of calls or usage alone still has one.
    out.delta.role = Some(Role::Assistant);
    let mut empty = true;

    if let Some(choice) = val.pointer("/choices/0") {
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.delta = out.delta.with_contents([PartDelta::Text {
                text: text.to_owned(),
            }]);
            empty = false;
        }
        if let Some(text) = delta["reasoning"].as_str().filter(|t| !t.is_empty()) {
            out.delta.thinking = Some(text.to_owned());
            empty = false;
        }
        for detail in delta["reasoning_details"].as_array().into_iter().flatten() {
            merge_detail(details, detail);
        }
        // A call's first piece has its id and name; the rest continue the last call.
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            out.delta = out.delta.with_tool_calls([PartDelta::Function {
                id: call["id"].as_str().map(str::to_owned),
                function: PartDeltaFunction::WithStringArgs {
                    name: call["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                },
            }]);
            empty = false;
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            out.finish_reason = Some(match reason {
                "stop" => FinishReason::Stop {},
                "tool_calls" => FinishReason::ToolCall {},
                "length" => FinishReason::Length {},
                other => FinishReason::Refusal {
                    reason: format!("reason: {other}"),
                },
            });
            empty = false;
        }
    }

    if let Some(usage) = val.get("usage").filter(|u| u.is_object()) {
        let count = |ptr: &str| usage.pointer(ptr).and_then(|v| v.as_u64());
        out.usage = Some(TokenUsage {
            input_tokens: count("/prompt_tokens").unwrap_or(0),
            output_tokens: count("/completion_tokens").unwrap_or(0),
            cache_creation_input_tokens: count("/prompt_tokens_details/cache_write_tokens"),
            cache_read_input_tokens: count("/prompt_tokens_details/cached_tokens"),
        });
        empty = false;
    }

    Ok((!empty).then_some(out))
}

/// Adds a streamed `reasoning_details` piece: a piece with the `index` of one already seen
/// continues it, its text fields appended and the rest taken as they come.
fn merge_detail(details: &mut Vec<serde_json::Value>, piece: &serde_json::Value) {
    let index = piece.get("index").and_then(|v| v.as_u64());
    let existing = index.and_then(|i| {
        details
            .iter_mut()
            .find(|d| d.get("index").and_then(|v| v.as_u64()) == Some(i))
    });
    let Some(existing) = existing.and_then(|d| d.as_object_mut()) else {
        details.push(piece.clone());
        return;
    };
    for (key, value) in piece.as_object().into_iter().flatten() {
        match (key.as_str(), existing.get_mut(key), value.as_str()) {
            ("text" | "summary" | "data", Some(serde_json::Value::String(acc)), Some(more)) => {
                acc.push_str(more)
            }
            _ => {
                existing.insert(key.clone(), value.clone());
            }
        }
    }
}

/// The conversation as Chat Completions messages.
fn chat_messages(messages: &[Message]) -> anyhow::Result<Value> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {
                let text = msg
                    .contents
                    .iter()
                    .filter_map(|p| p.as_text())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                out.push(to_value!({"role": "system", "content": text}));
            }
            Role::Tool => {
                let id = msg
                    .id
                    .as_deref()
                    .context("a tool message has no id to match its call")?;
                let content = msg
                    .contents
                    .iter()
                    .map(content_part)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                out.push(to_value!({"role": "tool", "tool_call_id": id, "content": content}));
            }
            Role::User => {
                let content = msg
                    .contents
                    .iter()
                    .map(content_part)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                out.push(to_value!({"role": "user", "content": content}));
            }
            Role::Assistant => {
                // Assistant content is text only.
                let text = msg
                    .contents
                    .iter()
                    .filter_map(|p| p.as_text())
                    .collect::<String>();
                let mut turn = to_value!({"role": "assistant", "content": text});
                let obj = turn.as_object_mut().unwrap();
                let calls: Vec<Value> = msg
                    .tool_calls
                    .iter()
                    .flatten()
                    .filter_map(|call| match call {
                        Part::Function {
                            id,
                            function: PartFunction { name, arguments },
                        } => Some(to_value!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                if !calls.is_empty() {
                    obj.insert("tool_calls".into(), Value::Array(calls));
                }
                // The reasoning this turn came with, if it came through OpenRouter.
                if let Some(details) = msg
                    .signature
                    .as_deref()
                    .and_then(|s| serde_json::from_str::<Value>(s).ok())
                    .filter(|d| d.is_array())
                {
                    obj.insert("reasoning_details".into(), details);
                }
                out.push(turn);
            }
        }
    }
    Ok(Value::Array(out))
}

fn content_part(part: &Part) -> anyhow::Result<Value> {
    Ok(match part {
        Part::Text { text } => to_value!({"type": "text", "text": text}),
        Part::Image { image } => {
            let url = match image {
                PartImage::Embedded { mime_type, data } => {
                    format!("data:{mime_type};base64,{}", data.base64())
                }
                PartImage::Url { url } => url.clone(),
            };
            to_value!({"type": "image_url", "image_url": {"url": url}})
        }
        Part::Value { value } => to_value!({
            "type": "text",
            "text": match value {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            },
        }),
        Part::Function { .. } => bail!("a function call cannot be message content"),
    })
}

/// Closes every object sub-schema (`"additionalProperties": false`) that does not say
/// otherwise, as strict mode requires.
fn closed_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(obj) => {
            let mut out: indexmap::IndexMap<String, Value> = obj
                .iter()
                .map(|(k, v)| {
                    let v = match k.as_str() {
                        "properties" | "$defs" | "definitions" => match v {
                            Value::Object(inner) => Value::Object(
                                inner
                                    .iter()
                                    .map(|(k, v)| (k.clone(), closed_schema(v)))
                                    .collect(),
                            ),
                            other => other.clone(),
                        },
                        "items" | "not" | "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                            closed_schema(v)
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
        Value::Array(items) => Value::Array(items.iter().map(closed_schema).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_accumulate_into_text_calls_and_usage() {
        let chunks = [
            r#"{"choices":[{"delta":{"role":"assistant","content":"Hi"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#,
            "[DONE]",
        ];
        let mut details = Vec::new();
        let mut acc = MessageDeltaOutput::new();
        for chunk in chunks {
            if let Some(out) = parse_chunk(chunk, &mut details).unwrap() {
                acc = acc.accumulate(out).unwrap();
            }
        }
        let out = acc.finish().unwrap();
        assert_eq!(out.message.contents[0].as_text(), Some("Hi"));
        assert_eq!(out.message.tool_calls.unwrap().len(), 1);
        assert_eq!(out.usage.unwrap().output_tokens, 5);
    }

    #[test]
    fn reasoning_details_merge_by_index() {
        let mut details = Vec::new();
        merge_detail(
            &mut details,
            &serde_json::json!({"type": "reasoning.text", "index": 0, "text": "a"}),
        );
        merge_detail(
            &mut details,
            &serde_json::json!({"index": 0, "text": "b", "signature": "s"}),
        );
        merge_detail(
            &mut details,
            &serde_json::json!({"type": "reasoning.encrypted", "index": 1, "data": "x"}),
        );
        assert_eq!(
            details,
            vec![
                serde_json::json!({"type": "reasoning.text", "index": 0, "text": "ab", "signature": "s"}),
                serde_json::json!({"type": "reasoning.encrypted", "index": 1, "data": "x"}),
            ]
        );
    }

    #[test]
    fn reasoning_details_go_back_with_their_turn() {
        let msg = Message::new(Role::Assistant)
            .with_contents([Part::text("ok")])
            .with_signatured_thinking("", r#"[{"type":"reasoning.encrypted","data":"x"}]"#);
        let out: serde_json::Value = chat_messages(&[msg]).unwrap().into();
        assert_eq!(out[0]["reasoning_details"][0]["data"], "x");
    }
}
