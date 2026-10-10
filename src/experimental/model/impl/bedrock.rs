//! A language model served by Amazon Bedrock, called through the Converse API.
//!
//! Every call is streamed (`/converse-stream`), and the reply is read as Amazon's binary
//! event stream (`application/vnd.amazon.eventstream`); [`infer`](InferLangModel::infer)
//! accumulates the same stream. Requests carry a Bedrock API key as a bearer token, so no
//! request signing is needed.

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

/// Region used when none is set in the environment.
const DEFAULT_REGION: &str = "us-east-1";

/// `maxTokens` sent while Claude thinks: thinking counts against it, and the Converse
/// default is too small to hold it. It stays above the largest thinking budget.
const CLAUDE_THINKING_MAX_TOKENS: i64 = 32000;

/// Runs a model on Amazon Bedrock through the Converse API.
///
/// Converse takes the same request for every model family; what it has no common field
/// for, the thinking effort, goes in as the family's own field (Claude and OpenAI models).
/// Image URLs are rejected, since Bedrock does not fetch them.
#[derive(Clone, Debug)]
pub struct BedrockModel {
    model: String,
    api_key: String,
    region: String,
}

impl BedrockModel {
    /// Runs `model`, a Bedrock model id or inference profile (e.g.
    /// `"us.anthropic.claude-sonnet-4-5-20250929-v1:0"`), in `region` with `api_key`.
    pub fn new(
        model: impl Into<String>,
        api_key: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            region: region.into(),
        }
    }

    /// Runs `model` with the key read from `AWS_BEARER_TOKEN_BEDROCK` and the region from
    /// `AWS_REGION` or `AWS_DEFAULT_REGION` (`us-east-1` if neither is set), as the AWS
    /// SDKs read them.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let env = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        let api_key =
            env("AWS_BEARER_TOKEN_BEDROCK").context("AWS_BEARER_TOKEN_BEDROCK is not set")?;
        let region = env("AWS_REGION")
            .or_else(|| env("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        Ok(Self::new(model, api_key, region))
    }

    /// Whether the model id names `vendor`, as `<vendor>.…` or `<geo>.<vendor>.…`. An
    /// application inference profile ARN names none.
    fn is_vendor(&self, vendor: &str) -> bool {
        self.model.split('.').take(2).any(|s| s == vendor)
    }

    fn url(&self) -> String {
        // An ARN's `/` is escaped so the id stays one path segment.
        format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/converse-stream",
            self.region,
            self.model.replace('/', "%2F")
        )
    }

    /// The request body for one call.
    fn body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body =
            to_value!({"messages": conversation(messages, self.is_vendor("anthropic"))?});
        let obj = body.as_object_mut().unwrap();

        let system: Vec<Value> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter().filter_map(|p| p.as_text()))
            .filter(|t| !t.is_empty())
            .map(|t| to_value!({"text": t}))
            .collect();
        if !system.is_empty() {
            obj.insert("system".into(), Value::Array(system));
        }

        if !tools.is_empty() {
            let specs: Vec<Value> = tools
                .iter()
                .map(|tool| {
                    let mut spec = to_value!({
                        "name": &tool.name,
                        "inputSchema": {"json": tool.parameters.clone()},
                    });
                    // Converse rejects an empty description.
                    if let Some(desc) = tool.description.as_deref().filter(|d| !d.is_empty()) {
                        spec.as_object_mut()
                            .unwrap()
                            .insert("description".into(), desc.into());
                    }
                    to_value!({"toolSpec": spec})
                })
                .collect();
            obj.insert(
                "toolConfig".into(),
                to_value!({"tools": specs, "toolChoice": {"auto": {}}}),
            );
        }

        // Unset keeps the model's default.
        if let Some(effort) = options.thinking_effort {
            if self.is_vendor("anthropic") {
                let thinking = if takes_thinking_budget(&self.model) {
                    let budget: i64 = match effort {
                        ThinkingEffort::Low => 2048,
                        ThinkingEffort::Medium => 8192,
                        ThinkingEffort::High => 24576,
                    };
                    to_value!({"thinking": {"type": "enabled", "budget_tokens": budget}})
                } else {
                    to_value!({
                        "thinking": {"type": "adaptive", "display": "summarized"},
                        "output_config": {"effort": effort_name(effort)},
                    })
                };
                obj.insert("additionalModelRequestFields".into(), thinking);
                obj.insert(
                    "inferenceConfig".into(),
                    to_value!({"maxTokens": CLAUDE_THINKING_MAX_TOKENS}),
                );
            } else if self.is_vendor("openai") {
                obj.insert(
                    "additionalModelRequestFields".into(),
                    to_value!({"reasoning_effort": effort_name(effort)}),
                );
            } else {
                log::warn!(
                    "{}: no thinking control known on Bedrock; ignored",
                    self.model
                );
            }
        }

        if let Some(schema) = &options.output_schema {
            // Converse takes the schema as a JSON string.
            let schema: serde_json::Value = closed_schema(schema).into();
            obj.insert(
                "outputConfig".into(),
                to_value!({
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {"jsonSchema": {"name": "output", "schema": schema.to_string()}},
                    }
                }),
            );
        }

        Ok(body.into())
    }
}

impl InferLangModel for BedrockModel {
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
        let url = self.url();
        let api_key = self.api_key.clone();

        Box::pin(async_stream::try_stream! {
            let response = send(&url, &api_key, &body).await?;

            let mut saw_finish = false;
            let mut bytes = response.bytes_stream();
            let mut buf = Vec::new();
            // Frames don't align with network chunks, so they are cut out of a buffer.
            // The stream goes on past `messageStop`: the usage comes after it.
            while let Some(chunk) = bytes.next().await {
                buf.extend_from_slice(&chunk?);
                while let Some(frame) = next_frame(&mut buf)? {
                    if let Some(out) = parse_frame(&frame)? {
                        saw_finish |= out.finish_reason.is_some();
                        yield out;
                    }
                }
            }
            if !buf.is_empty() {
                Err(anyhow::anyhow!("the Bedrock stream ended inside a frame"))?;
            }
            if !saw_finish {
                Err(anyhow::anyhow!("the Bedrock stream ended before the message stopped"))?;
            }
        })
    }
}

fn effort_name(effort: ThinkingEffort) -> &'static str {
    match effort {
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
    }
}

/// Whether a Claude model predates adaptive thinking (Claude 4.6) and thinks only on a
/// `budget_tokens` budget. Reads Bedrock ids such as
/// `us.anthropic.claude-haiku-4-5-20251001-v1:0` and `anthropic.claude-3-7-sonnet-…`; an
/// unreadable id counts as newer.
fn takes_thinking_budget(model: &str) -> bool {
    let Some((_, rest)) = model.split_once("claude-") else {
        return false;
    };
    let is_num = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut parts = rest.split('-').skip_while(|t| !is_num(t));
    let Some(major) = parts.next().and_then(|t| t.parse::<u32>().ok()) else {
        return false;
    };
    // A date (`20250514`) after the major version is no minor version.
    let minor = parts
        .next()
        .filter(|t| is_num(t) && t.len() <= 2)
        .and_then(|t| t.parse::<u32>().ok())
        .unwrap_or(0);
    (major, minor) < (4, 6)
}

/// POSTs `body`, retrying throttled requests (429) with backoff, and returns the successful
/// response unread.
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
            .bearer_auth(api_key)
            .header("accept", "application/vnd.amazon.eventstream")
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
                "Throttled (429). Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("Bedrock request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// One event stream frame: its string headers and its payload.
struct Frame {
    headers: Vec<(String, String)>,
    payload: Vec<u8>,
}

impl Frame {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Cuts the next complete frame out of `buf`; `None` until one has fully arrived.
///
/// A frame is a 12-byte prelude (total length, headers length, CRC of the two), the
/// headers, the payload and a CRC of everything before it; integers are big-endian and
/// both CRCs CRC-32. A bad CRC fails, since the stream cannot be picked up again after it.
fn next_frame(buf: &mut Vec<u8>) -> anyhow::Result<Option<Frame>> {
    const PRELUDE: usize = 12;
    const CRC: usize = 4;
    if buf.len() < PRELUDE {
        return Ok(None);
    }
    let be32 = |b: &[u8]| u32::from_be_bytes(b.try_into().unwrap());
    let total = be32(&buf[0..4]) as usize;
    let headers_len = be32(&buf[4..8]) as usize;
    if be32(&buf[8..12]) != crc32(&buf[0..8]) {
        bail!("Bedrock event stream: prelude CRC mismatch");
    }
    if total < PRELUDE + headers_len + CRC {
        bail!("Bedrock event stream: inconsistent frame lengths");
    }
    if buf.len() < total {
        return Ok(None);
    }
    let frame: Vec<u8> = buf.drain(..total).collect();
    if be32(&frame[total - CRC..]) != crc32(&frame[..total - CRC]) {
        bail!("Bedrock event stream: message CRC mismatch");
    }
    Ok(Some(Frame {
        headers: parse_headers(&frame[PRELUDE..PRELUDE + headers_len])?,
        payload: frame[PRELUDE + headers_len..total - CRC].to_vec(),
    }))
}

/// Headers are `name length (u8), name, value type (u8), value`; only string values
/// (type 7) are kept, which all the headers read here are.
fn parse_headers(mut raw: &[u8]) -> anyhow::Result<Vec<(String, String)>> {
    let mut headers = Vec::new();
    while let Some((&name_len, rest)) = raw.split_first() {
        let name = rest
            .get(..name_len as usize)
            .context("Bedrock event stream: truncated header")?;
        let name = String::from_utf8_lossy(name).into_owned();
        let rest = &rest[name_len as usize..];
        let (&ty, rest) = rest
            .split_first()
            .context("Bedrock event stream: truncated header")?;
        let (len, rest) = match ty {
            0 | 1 => (0, rest),
            2 => (1, rest),
            3 => (2, rest),
            4 => (4, rest),
            5 | 8 => (8, rest),
            9 => (16, rest),
            6 | 7 => {
                let len = rest
                    .get(..2)
                    .context("Bedrock event stream: truncated header")?;
                (u16::from_be_bytes([len[0], len[1]]) as usize, &rest[2..])
            }
            other => bail!("Bedrock event stream: unknown header type {other}"),
        };
        let value = rest
            .get(..len)
            .context("Bedrock event stream: truncated header")?;
        if ty == 7 {
            headers.push((name, String::from_utf8_lossy(value).into_owned()));
        }
        raw = &rest[len..];
    }
    Ok(headers)
}

/// CRC-32 (IEEE), bit by bit.
fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Parses one frame into a delta; frames that carry none give `None`, and exception
/// frames fail.
fn parse_frame(frame: &Frame) -> anyhow::Result<Option<MessageDeltaOutput>> {
    let payload: Value = if frame.payload.is_empty() {
        Value::object_empty()
    } else {
        serde_json::from_slice(&frame.payload).context("Bedrock event payload is not JSON")?
    };
    let str_at = |ptr: &str| payload.pointer(ptr).and_then(|v| v.as_str());

    match frame.header(":message-type").unwrap_or("event") {
        "event" => {}
        "exception" => bail!(
            "Bedrock stream exception ({}): {}",
            frame.header(":exception-type").unwrap_or("unknown"),
            str_at("/message").unwrap_or("(no message)")
        ),
        other => bail!(
            "Bedrock stream error ({}): {}",
            frame.header(":error-code").unwrap_or(other),
            frame.header(":error-message").unwrap_or("(no message)")
        ),
    }

    let mut out = MessageDeltaOutput::new();
    // Every delta carries the role, so a reply of calls or usage alone still has one.
    out.delta.role = Some(Role::Assistant);
    match frame.header(":event-type").unwrap_or("") {
        "messageStart" => {}
        "contentBlockStart" => {
            let Some(tool) = payload.pointer("/start/toolUse") else {
                return Ok(None);
            };
            out.delta = MessageDelta::new()
                .with_role(Role::Assistant)
                .with_tool_calls([PartDelta::Function {
                    id: tool
                        .pointer("/toolUseId")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned),
                    function: PartDeltaFunction::WithStringArgs {
                        name: tool
                            .pointer("/name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_owned(),
                        arguments: String::new(),
                    },
                }]);
        }
        "contentBlockDelta" => {
            if let Some(text) = str_at("/delta/text") {
                if text.is_empty() {
                    return Ok(None);
                }
                out.delta = out.delta.with_contents([PartDelta::Text {
                    text: text.to_owned(),
                }]);
            } else if let Some(args) = str_at("/delta/toolUse/input") {
                out.delta = out.delta.with_tool_calls([PartDelta::Function {
                    id: None,
                    function: PartDeltaFunction::WithStringArgs {
                        name: String::new(),
                        arguments: args.to_owned(),
                    },
                }]);
            } else if let Some(text) = str_at("/delta/reasoningContent/text") {
                out.delta.thinking = Some(text.to_owned());
            } else if let Some(sig) = str_at("/delta/reasoningContent/signature") {
                out.delta.signature = Some(sig.to_owned());
            } else {
                return Ok(None);
            }
        }
        "messageStop" => {
            out.finish_reason = Some(match str_at("/stopReason").unwrap_or("end_turn") {
                "end_turn" | "stop_sequence" => FinishReason::Stop {},
                "tool_use" => FinishReason::ToolCall {},
                "max_tokens" | "model_context_window_exceeded" => FinishReason::Length {},
                other => FinishReason::Refusal {
                    reason: format!("reason: {other}"),
                },
            });
        }
        "metadata" => {
            let Some(usage) = payload.pointer("/usage") else {
                return Ok(None);
            };
            let count = |key: &str| {
                usage
                    .pointer(key)
                    .and_then(|v| v.as_integer())
                    .map(|v| v as u64)
            };
            out.usage = Some(TokenUsage {
                input_tokens: count("/inputTokens").unwrap_or(0),
                output_tokens: count("/outputTokens").unwrap_or(0),
                cache_creation_input_tokens: count("/cacheWriteInputTokens"),
                cache_read_input_tokens: count("/cacheReadInputTokens"),
            });
        }
        // `contentBlockStop`, and event types added later.
        _ => return Ok(None),
    }
    Ok(Some(out))
}

/// The conversation as Converse `messages`. System messages go in `system` instead. Tool
/// results become user turns, and consecutive turns of one role are merged, since Converse
/// requires users and the assistant to alternate. Thinking is replayed, when
/// `replay_thinking`, only on assistant turns after the last user message, as Claude
/// requires; other models are not sent it, since some echo it back as text.
fn conversation(messages: &[Message], replay_thinking: bool) -> anyhow::Result<Value> {
    let last_user = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len());
    let mut turns: Vec<(&'static str, Vec<Value>)> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        let (role, blocks) = match msg.role {
            Role::System => continue,
            Role::Tool => {
                let id = msg
                    .id
                    .as_deref()
                    .context("a tool message has no id to match its call")?;
                let mut content = msg
                    .contents
                    .iter()
                    .map(tool_result_block)
                    .collect::<anyhow::Result<Vec<_>>>()?;
                if content.is_empty() {
                    // Converse rejects an empty result.
                    content.push(to_value!({"text": "(no output)"}));
                }
                (
                    "user",
                    vec![to_value!({"toolResult": {"toolUseId": id, "content": content}})],
                )
            }
            Role::User | Role::Assistant => {
                let mut blocks = Vec::new();
                if replay_thinking
                    && i > last_user
                    && let Some(thinking) = msg.thinking.as_ref().filter(|t| !t.is_empty())
                {
                    let mut text = to_value!({"text": thinking});
                    if let Some(sig) = &msg.signature {
                        text.as_object_mut()
                            .unwrap()
                            .insert("signature".into(), sig.into());
                    }
                    blocks.push(to_value!({"reasoningContent": {"reasoningText": text}}));
                }
                for part in &msg.contents {
                    if let Some(block) = content_block(part)? {
                        blocks.push(block);
                    }
                }
                for call in msg.tool_calls.iter().flatten() {
                    if let Part::Function {
                        id,
                        function: PartFunction { name, arguments },
                    } = call
                    {
                        blocks.push(to_value!({
                            "toolUse": {"toolUseId": id, "name": name, "input": arguments.clone()}
                        }));
                    }
                }
                let role = if msg.role == Role::Assistant {
                    "assistant"
                } else {
                    "user"
                };
                (role, blocks)
            }
        };
        match turns.last_mut() {
            Some((last_role, last_blocks)) if *last_role == role => last_blocks.extend(blocks),
            _ => turns.push((role, blocks)),
        }
    }
    Ok(Value::Array(
        turns
            .into_iter()
            .filter(|(_, blocks)| !blocks.is_empty())
            .map(|(role, blocks)| to_value!({"role": role, "content": blocks}))
            .collect(),
    ))
}

/// A user or assistant content block; Converse rejects empty text, so that gives `None`.
fn content_block(part: &Part) -> anyhow::Result<Option<Value>> {
    Ok(match part {
        Part::Text { text } if text.is_empty() => None,
        Part::Text { text } => Some(to_value!({"text": text})),
        Part::Image { image } => Some(image_block(image)?),
        Part::Value { value } => Some(to_value!({
            "text": serde_json::to_string(value).unwrap_or_default()
        })),
        Part::Function {
            id,
            function: PartFunction { name, arguments },
        } => Some(to_value!({
            "toolUse": {"toolUseId": id, "name": name, "input": arguments.clone()}
        })),
    })
}

/// A `toolResult` content block, where JSON objects may go as they are.
fn tool_result_block(part: &Part) -> anyhow::Result<Value> {
    Ok(match part {
        Part::Text { text } => to_value!({"text": text}),
        Part::Image { image } => image_block(image)?,
        Part::Value { value } if value.is_object() => to_value!({"json": value.clone()}),
        Part::Value { value } => to_value!({
            "text": match value {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }
        }),
        Part::Function { .. } => to_value!({"text": ""}),
    })
}

fn image_block(image: &PartImage) -> anyhow::Result<Value> {
    match image {
        PartImage::Embedded { mime_type, data } => {
            let format = match mime_type.strip_prefix("image/").unwrap_or(mime_type) {
                "jpg" => "jpeg",
                other => other,
            };
            Ok(to_value!({"image": {"format": format, "source": {"bytes": data.base64()}}}))
        }
        PartImage::Url { url } => bail!("Bedrock takes no image URLs; embed the image: {url}"),
    }
}

/// Closes every object sub-schema (`"additionalProperties": false`) that does not say
/// otherwise, as structured output requires.
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

    /// A frame with string headers, the inverse of [`next_frame`].
    fn frame(event_type: &str, payload: &str) -> Vec<u8> {
        let mut headers = Vec::new();
        for (k, v) in [(":message-type", "event"), (":event-type", event_type)] {
            headers.push(k.len() as u8);
            headers.extend_from_slice(k.as_bytes());
            headers.push(7);
            headers.extend_from_slice(&(v.len() as u16).to_be_bytes());
            headers.extend_from_slice(v.as_bytes());
        }
        let total = 12 + headers.len() + payload.len() + 4;
        let mut out = Vec::new();
        out.extend_from_slice(&(total as u32).to_be_bytes());
        out.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        out.extend_from_slice(&crc32(&out[..8]).to_be_bytes());
        out.extend_from_slice(&headers);
        out.extend_from_slice(payload.as_bytes());
        let crc = crc32(&out);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }

    #[test]
    fn crc32_matches_the_reference() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn frames_split_across_chunks_parse_into_a_tool_call() {
        let mut wire = frame(
            "contentBlockStart",
            r#"{"start":{"toolUse":{"toolUseId":"t1","name":"get_weather"}},"contentBlockIndex":0}"#,
        );
        wire.extend(frame(
            "contentBlockDelta",
            r#"{"delta":{"toolUse":{"input":"{\"city\":\"Seoul\"}"}},"contentBlockIndex":0}"#,
        ));
        wire.extend(frame("messageStop", r#"{"stopReason":"tool_use"}"#));

        let mut buf = wire[..7].to_vec();
        assert!(next_frame(&mut buf).unwrap().is_none());
        buf.extend_from_slice(&wire[7..]);
        let mut acc = MessageDeltaOutput::new();
        while let Some(f) = next_frame(&mut buf).unwrap() {
            if let Some(out) = parse_frame(&f).unwrap() {
                acc = acc.accumulate(out).unwrap();
            }
        }
        let out = acc.finish().unwrap();
        let calls = out.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert!(matches!(out.finish_reason, FinishReason::ToolCall {}));
    }
}
