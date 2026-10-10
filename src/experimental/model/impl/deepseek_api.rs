//! A language model that calls the DeepSeek API directly over HTTP.
//!
//! The API is OpenAI-compatible Chat Completions. Every call is streamed
//! (`"stream": true`) and read as server-sent events; [`infer`](InferLangModel::infer)
//! accumulates the same stream.

use anyhow::{Context as _, bail};
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};

use crate::{
    datatype::Value,
    experimental::model::{InferLangModel, LangModelOptions},
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, Role, TokenUsage,
    },
    to_value,
    tool::ToolDesc,
};

const API_URL: &str = "https://api.deepseek.com/chat/completions";

/// Runs DeepSeek models through the DeepSeek API.
///
/// - Thinking is on or off, not graded: any [`ThinkingEffort`](crate::experimental::model::ThinkingEffort)
///   turns it on. The thinking comes back as `reasoning_content`, and goes back with its
///   turn, as the API needs while tools are in use.
/// - The API constrains output to JSON but takes no schema, so an output schema is put in
///   the system prompt for the model to follow.
/// - Text only: image parts are rejected.
#[derive(Clone, Debug)]
pub struct DeepSeekApiModel {
    model: String,
    api_key: String,
}

impl DeepSeekApiModel {
    /// Runs `model`, an API model id (e.g. `"deepseek-chat"`), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `DEEPSEEK_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("DEEPSEEK_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The streamed request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .collect::<Vec<_>>()
            .join("\n\n");
        if let Some(schema) = &options.output_schema {
            let schema: serde_json::Value = schema.clone().into();
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&format!(
                "Reply with a single JSON object that matches this JSON schema, and nothing \
                 else:\n{schema}"
            ));
        }

        let mut chat = Vec::new();
        if !system.is_empty() {
            chat.push(to_value!({"role": "system", "content": system}));
        }
        chat.extend(chat_messages(messages)?);

        let mut body = to_value!({
            "model": self.model.as_str(),
            "messages": chat,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        let obj = body.as_object_mut().unwrap();

        if !tools.is_empty() {
            obj.insert(
                "tools".into(),
                Value::Array(tools.iter().map(tool_spec).collect()),
            );
            obj.insert("tool_choice".into(), "auto".into());
        }
        // Unset keeps the model's default.
        if options.thinking_effort.is_some() {
            obj.insert("thinking".into(), to_value!({"type": "enabled"}));
        }
        if options.output_schema.is_some() {
            obj.insert("response_format".into(), to_value!({"type": "json_object"}));
        }

        Ok(body.into())
    }
}

impl InferLangModel for DeepSeekApiModel {
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
                    if let Some(out) = parse_chunk(&data)? {
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
                if at_eof {
                    break;
                }
            }
            if !saw_finish {
                Err(anyhow::anyhow!("the DeepSeek stream ended before the reply finished"))?;
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
            let wait_secs = (1u64 << attempt).min(MAX_WAIT_SECS);
            log::warn!(
                "Rate limited (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("DeepSeek API request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Cuts the next complete SSE event out of `buf` and returns its joined `data:` lines;
/// `None` until a blank line ends one. Comments (keep-alives) and events without data come
/// back empty.
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

/// Parses one stream chunk into a delta; chunks that carry nothing give `None`.
fn parse_chunk(data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    let val: serde_json::Value = serde_json::from_str(data)?;
    if let Some(error) = val.get("error") {
        bail!(
            "DeepSeek stream error: {}",
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
        if let Some(text) = delta["reasoning_content"]
            .as_str()
            .filter(|t| !t.is_empty())
        {
            out.delta.thinking = Some(text.to_owned());
            empty = false;
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
        let count = |key: &str| usage[key].as_u64();
        out.usage = Some(TokenUsage {
            input_tokens: count("prompt_tokens").unwrap_or(0),
            output_tokens: count("completion_tokens").unwrap_or(0),
            cache_creation_input_tokens: count("prompt_cache_miss_tokens"),
            cache_read_input_tokens: count("prompt_cache_hit_tokens"),
        });
        empty = false;
    }

    Ok((!empty).then_some(out))
}

fn tool_spec(tool: &ToolDesc) -> Value {
    let mut function = to_value!({"name": &tool.name, "parameters": tool.parameters.clone()});
    if let Some(desc) = &tool.description {
        function
            .as_object_mut()
            .unwrap()
            .insert("description".into(), desc.into());
    }
    to_value!({"type": "function", "function": function})
}

/// The conversation, system messages aside, as Chat Completions messages.
fn chat_messages(messages: &[Message]) -> anyhow::Result<Vec<Value>> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => {}
            Role::Tool => {
                let id = msg
                    .id
                    .as_deref()
                    .context("a tool message has no id to match its call")?;
                out.push(to_value!({"role": "tool", "tool_call_id": id, "content": text_of(msg)?}));
            }
            Role::User => out.push(to_value!({"role": "user", "content": text_of(msg)?})),
            Role::Assistant => {
                let mut turn = to_value!({"role": "assistant", "content": text_of(msg)?});
                let obj = turn.as_object_mut().unwrap();
                if let Some(thinking) = msg.thinking.as_ref().filter(|t| !t.is_empty()) {
                    obj.insert("reasoning_content".into(), thinking.into());
                }
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
                out.push(turn);
            }
        }
    }
    Ok(out)
}

/// A message's contents as one text; images are rejected.
fn text_of(msg: &Message) -> anyhow::Result<String> {
    let mut text = String::new();
    for part in &msg.contents {
        match part {
            Part::Text { text: t } => text.push_str(t),
            Part::Value { value } => text.push_str(&match value {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }),
            Part::Image { .. } => bail!("the DeepSeek API takes no images"),
            Part::Function { .. } => {}
        }
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_and_calls_accumulate() {
        let chunks = [
            r#"{"choices":[{"delta":{"role":"assistant","reasoning_content":"think"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#,
            "[DONE]",
        ];
        let mut acc = MessageDeltaOutput::new();
        for chunk in chunks {
            if let Some(out) = parse_chunk(chunk).unwrap() {
                acc = acc.accumulate(out).unwrap();
            }
        }
        let out = acc.finish().unwrap();
        assert_eq!(out.message.thinking.as_deref(), Some("think"));
        assert_eq!(out.message.tool_calls.unwrap().len(), 1);
    }
}
