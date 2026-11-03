//! A language model served over the OpenAI-compatible Chat Completions API, called over HTTP.
//!
//! vLLM, llama-server, Ollama, LM Studio and SGLang all serve it. Every call is streamed
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
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    model::{InferLangModel, LangModelOptions, api_error},
    to_value,
    tool::ToolDesc,
};

/// Runs a model on an OpenAI-compatible server.
///
/// - Thinking is on or off, not graded: any
///   [`ThinkingEffort`](crate::model::ThinkingEffort) sets the chat template's
///   `enable_thinking` (read by Qwen's and GLM's) and `thinking` (read by DeepSeek's) through
///   `chat_template_kwargs`; a server or template that reads neither keeps its default. The
///   thinking comes back when the server runs a reasoning parser, as `reasoning` (or
///   `reasoning_content` on older servers), and is not sent back.
/// - An output schema is sent as a `json_schema` `response_format`, which vLLM, llama-server,
///   Ollama and LM Studio enforce.
#[derive(Clone)]
pub struct OpenAiCompatibleModel {
    model: String,
    base_url: String,
    api_key: Option<String>,
}

// Hand-written so the key stays out of logs.
impl std::fmt::Debug for OpenAiCompatibleModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleModel")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl OpenAiCompatibleModel {
    /// Runs `model`, the id the server serves it as (`"zai-org/GLM-4.6"` on vLLM,
    /// `"llama3.3:70b"` on Ollama), on the server at `base_url` (e.g.
    /// `"http://localhost:8000/v1"`).
    pub fn new(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key: None,
        }
    }

    /// Sends `api_key` as a bearer token, for a server that wants one (`vllm serve
    /// --api-key`, `llama-server --api-key`).
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// The streamed request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .collect::<Vec<_>>()
            .join("\n\n");

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
            let tools: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let mut function =
                        to_value!({"name": &tool.name, "parameters": tool.parameters.clone()});
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
        // Unset keeps the model's default.
        if options.thinking_effort.is_some() {
            obj.insert(
                "chat_template_kwargs".into(),
                to_value!({"enable_thinking": true, "thinking": true}),
            );
        }
        if let Some(schema) = &options.output_schema {
            obj.insert(
                "response_format".into(),
                to_value!({
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema.clone()},
                }),
            );
        }

        Ok(body.into())
    }
}

impl InferLangModel for OpenAiCompatibleModel {
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
        let url = format!("{}/chat/completions", self.base_url);
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let response = send(&url, api_key.as_deref(), &body).await?;

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
            if !saw_finish {
                Err(anyhow::anyhow!("the stream ended before the reply finished"))?;
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

/// POSTs `body`, retrying 429s and 5xx overloads with backoff, and returns the successful
/// response unread.
async fn send(
    url: &str,
    api_key: Option<&str>,
    body: &serde_json::Value,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    // Rate limits, and the transient errors and overloads (529) servers recover from.
    const RETRY_STATUSES: [u16; 6] = [429, 500, 502, 503, 504, 529];

    for attempt in 0..=MAX_RETRIES {
        let mut request = CLIENT.post(url).header("accept", "text/event-stream");
        if let Some(api_key) = api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.json(body).send().await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if RETRY_STATUSES.contains(&status.as_u16()) && attempt < MAX_RETRIES {
            let wait_secs = (1u64 << attempt).min(MAX_WAIT_SECS);
            log::warn!(
                "Got {status}. Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        return Err(api_error(
            status.as_u16(),
            format!("chat completions request failed with status {status}: {text}"),
        ));
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
            "chat completions stream error: {}",
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
            out.delta.contents.push(PartDelta::Text {
                text: text.to_owned(),
            });
            empty = false;
        }
        // Newer servers name it `reasoning`, older ones `reasoning_content`.
        if let Some(text) = delta["reasoning"]
            .as_str()
            .or_else(|| delta["reasoning_content"].as_str())
            .filter(|t| !t.is_empty())
        {
            out.delta.thinking = Some(text.to_owned());
            empty = false;
        }
        // A call's first piece has its id and name; the rest continue the last call.
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            out.delta.tool_calls.push(PartDelta::Function {
                id: call["id"]
                    .as_str()
                    .filter(|id| !id.is_empty())
                    .map(str::to_owned),
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
            });
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
            cache_creation_input_tokens: None,
            cache_read_input_tokens: count("/prompt_tokens_details/cached_tokens"),
        });
        empty = false;
    }

    Ok((!empty).then_some(out))
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
                let text = msg
                    .contents
                    .iter()
                    .map(|p| match p {
                        Part::Text { text } => text.clone(),
                        Part::Value { value } => value_text(value),
                        // A tool result stays in the history, so rejecting its image would
                        // fail every later call; a label tells the model one was returned.
                        Part::Image {
                            image: PartImage::Embedded { mime_type, .. },
                        } => format!("[image: {mime_type}]"),
                        Part::Image {
                            image: PartImage::Url { url },
                        } => format!("[image at {url}]"),
                        Part::Function { .. } => String::new(),
                    })
                    .collect::<String>();
                out.push(to_value!({"role": "tool", "tool_call_id": id, "content": text}));
            }
            Role::User => {
                let content: Vec<Value> = msg
                    .contents
                    .iter()
                    .map(|p| match p {
                        Part::Text { text } => to_value!({"type": "text", "text": text}),
                        Part::Value { value } => {
                            to_value!({"type": "text", "text": value_text(value)})
                        }
                        Part::Image { image } => {
                            let url = match image {
                                PartImage::Embedded { mime_type, data } => {
                                    format!("data:{mime_type};base64,{}", data.base64())
                                }
                                PartImage::Url { url } => url.clone(),
                            };
                            to_value!({"type": "image_url", "image_url": {"url": url}})
                        }
                        Part::Function { .. } => to_value!({"type": "text", "text": ""}),
                    })
                    .collect();
                out.push(to_value!({"role": "user", "content": content}));
            }
            Role::Assistant => {
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
                out.push(turn);
            }
        }
    }
    Ok(out)
}

fn value_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}
