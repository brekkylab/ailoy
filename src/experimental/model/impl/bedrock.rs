use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{
    EventParser, assistant_output, bearer, close_objects, eventstream_drain, eventstream_flush,
    last_user_index, rate_limit_only, request_json, request_stream, system_text, tokens,
    value_text,
};
use crate::{
    datatype::Value,
    experimental::model::{LangModelInference, ProviderEntry, find_entry},
    message::{
        FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    tool::ToolDesc,
};

/// Request options for [`Bedrock`]. `None` leaves a field out of the request, so the
/// model's default applies. The Converse API is the same for every model family, so
/// anything family-specific (Claude's `thinking`, OpenAI's `reasoning_effort`, …) goes in
/// [`additional_model_request_fields`](Self::additional_model_request_fields).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BedrockOption {
    /// Sent as `inferenceConfig.maxTokens`. No default: the ceiling differs by family and
    /// an over-limit value is rejected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Strings that end the response when generated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// The model family's own request fields, passed through as-is; sent as
    /// `additionalModelRequestFields`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_model_request_fields: Option<Value>,

    /// Constrains the response to a JSON schema; sent as `outputConfig.textFormat`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<BedrockOutputFormat>,

    /// Whether and which tools the model must call; sent as `toolConfig.toolChoice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<BedrockToolChoice>,
}

/// Sent as a `json_schema` text format, under the name `response`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BedrockOutputFormat {
    JsonSchema { schema: Value },
}

/// The `toolChoice` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BedrockToolChoice {
    /// The model decides; the default when tools are given.
    Auto {},
    /// Some tool must be called.
    Any {},
    /// The named tool must be called.
    Tool { name: String },
}

/// A model on Amazon Bedrock over the Converse API. The provider's `bedrock` entry, a
/// [`ProviderEntry::Bedrock`], gives the API key and the region.
///
/// The model id is one Bedrock accepts for on-demand throughput, e.g. the inference-profile
/// id `global.anthropic.claude-sonnet-5`; plain foundation-model ids are rejected.
#[derive(Clone, Debug)]
pub struct Bedrock {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `bedrock` entry each request uses.
    provider: String,
    option: BedrockOption,
}

impl Bedrock {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: BedrockOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: BedrockOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &BedrockOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut BedrockOption {
        &mut self.option
    }

    /// The URL, headers and body of a request, with the region and API key from the
    /// provider.
    fn request(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        stream: bool,
    ) -> anyhow::Result<(String, HeaderMap, serde_json::Value)> {
        let ProviderEntry::Bedrock { region, api_key } = find_entry(&self.provider, "bedrock")?
        else {
            anyhow::bail!("Bedrock takes a region and an API key, as `ProviderEntry::Bedrock`");
        };
        Ok((
            self.request_url(&region, stream)?,
            self.request_headers(&api_key, stream)?,
            self.request_body(messages, tools)?,
        ))
    }

    /// `/model/<model>/converse[-stream]` on the region's runtime endpoint. A `/` in the
    /// model id (an inference-profile ARN) is escaped to keep it one path segment.
    fn request_url(&self, region: &str, stream: bool) -> anyhow::Result<String> {
        let region_ok = !region.is_empty()
            && region
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !region_ok {
            anyhow::bail!("{region:?} is not an AWS region");
        }
        let action = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        Ok(format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/{action}",
            region,
            self.model.replace('/', "%2F"),
        ))
    }

    /// The Converse body. System messages are joined into one `system` block.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut wire = wire_messages(messages)?;
        if takes_images_outside_tool_results(&self.model) {
            lift_tool_result_images(&mut wire);
        }
        let mut body = json!({"messages": wire});
        let fields = body.as_object_mut().unwrap();

        let system = system_text(messages);
        if !system.is_empty() {
            fields.insert("system".into(), json!([{"text": system}]));
        }
        if !tools.is_empty() {
            let mut tool_config = json!({"tools": tools.iter().map(wire_tool).collect::<Vec<_>>()});
            if let Some(tool_choice) = &option.tool_choice {
                tool_config["toolChoice"] = serde_json::to_value(tool_choice)?;
            }
            fields.insert("toolConfig".into(), tool_config);
        }

        let mut inference = serde_json::Map::new();
        if let Some(max_tokens) = option.max_tokens {
            inference.insert("maxTokens".into(), max_tokens.into());
        }
        if let Some(temperature) = option.temperature {
            inference.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = option.top_p {
            inference.insert("topP".into(), top_p.into());
        }
        if !option.stop_sequences.is_empty() {
            inference.insert("stopSequences".into(), json!(option.stop_sequences));
        }
        if !inference.is_empty() {
            fields.insert("inferenceConfig".into(), inference.into());
        }
        if let Some(additional) = &option.additional_model_request_fields {
            fields.insert(
                "additionalModelRequestFields".into(),
                additional.clone().into(),
            );
        }
        if let Some(BedrockOutputFormat::JsonSchema { schema }) = &option.format {
            // Converse takes the schema as a JSON string, not an object.
            let schema = close_objects(&schema.clone().into());
            fields.insert(
                "outputConfig".into(),
                json!({"textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {"name": "response", "schema": schema.to_string()},
                    },
                }}),
            );
        }
        Ok(body)
    }

    /// The API key as a bearer token (no SigV4), and for a stream the binary event stream
    /// as `accept`.
    fn request_headers(&self, api_key: &str, stream: bool) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer(api_key)?);
        if stream {
            headers.insert(
                ACCEPT,
                HeaderValue::from_static("application/vnd.amazon.eventstream"),
            );
        }
        Ok(headers)
    }
}

impl LangModelInference for Bedrock {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let (url, request) = match self.request(messages, tools, true) {
            Ok((url, headers, body)) => (url, Ok((headers, body))),
            Err(e) => (String::new(), Err(e)),
        };
        request_stream(url, request, rate_limit_only, BedrockEvents)
    }

    fn infer<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let (url, headers, body) = self.request(messages, tools, false)?;
            parse_response(&request_json(&url, headers, &body, rate_limit_only).await?)
        })
    }
}

/// Converse `messages`. System messages are left out, as they go in `system`. A tool
/// result is a user turn, and turns of one role in a row are merged, since Converse needs
/// user and assistant to alternate. Thinking is replayed only for assistant turns after
/// the last user message.
fn wire_messages(messages: &[Message]) -> anyhow::Result<serde_json::Value> {
    let last_user = last_user_index(messages);
    let mut out: Vec<serde_json::Value> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::System {
            continue;
        }
        let mut wire = wire_message(msg, i > last_user)?;
        match out.last_mut() {
            Some(prev) if prev["role"] == wire["role"] => {
                let blocks = std::mem::take(wire["content"].as_array_mut().unwrap());
                prev["content"].as_array_mut().unwrap().extend(blocks);
            }
            _ => out.push(wire),
        }
    }
    Ok(out.into())
}

fn wire_message(msg: &Message, with_thinking: bool) -> anyhow::Result<serde_json::Value> {
    if msg.role == Role::Tool {
        let mut content = Vec::new();
        for part in &msg.contents {
            content.push(match part {
                Part::Value { value } if value.is_object() => {
                    json!({"json": serde_json::Value::from(value.clone())})
                }
                Part::Value { value } => json!({"text": value_text(value)}),
                Part::Function { .. } => continue,
                Part::Text { text } if text.is_empty() => continue,
                part => wire_block(part)?,
            });
        }
        if content.is_empty() {
            content.push(json!({"text": "(no output)"}));
        }
        return Ok(json!({
            "role": "user",
            "content": [{"toolResult": {
                "toolUseId": msg.id.as_deref().unwrap_or_default(),
                "content": content,
            }}],
        }));
    }

    let mut content = Vec::new();
    if with_thinking && let Some(thinking) = msg.thinking.as_deref().filter(|t| !t.is_empty()) {
        let mut text = json!({"text": thinking});
        if let Some(signature) = &msg.signature {
            text["signature"] = signature.as_str().into();
        }
        content.push(json!({"reasoningContent": {"reasoningText": text}}));
    }
    for part in msg.contents.iter().chain(msg.tool_calls.iter().flatten()) {
        // Converse rejects empty text blocks.
        if !matches!(part, Part::Text { text } if text.is_empty()) {
            content.push(wire_block(part)?);
        }
    }
    let role = if msg.role == Role::Assistant {
        "assistant"
    } else {
        "user"
    };
    Ok(json!({"role": role, "content": content}))
}

/// A content block. Converse has no free-form JSON block outside tool results, and
/// fetches no image URL.
fn wire_block(part: &Part) -> anyhow::Result<serde_json::Value> {
    Ok(match part {
        Part::Text { text } => json!({"text": text}),
        Part::Value { value } => json!({"text": value_text(value)}),
        Part::Function { id, function } => json!({"toolUse": {
            "toolUseId": id,
            "name": function.name,
            "input": serde_json::Value::from(function.arguments.clone()),
        }}),
        Part::Image {
            image: PartImage::Embedded { mime_type, data },
        } => {
            let format = match mime_type.strip_prefix("image/").unwrap_or(mime_type) {
                "jpg" => "jpeg",
                other => other,
            };
            json!({"image": {"format": format, "source": {"bytes": data.base64()}}})
        }
        Part::Image {
            image: PartImage::Url { .. },
        } => anyhow::bail!("Bedrock takes no image URLs; embed the image instead"),
    })
}

fn wire_tool(tool: &ToolDesc) -> serde_json::Value {
    let mut spec = json!({
        "name": tool.name,
        "inputSchema": {"json": serde_json::Value::from(tool.parameters.clone())},
    });
    // Converse rejects an empty description.
    if let Some(description) = tool.description.as_deref().filter(|d| !d.is_empty()) {
        spec["description"] = description.into();
    }
    json!({"toolSpec": spec})
}

/// Bedrock's OpenAI models answer an image inside `toolResult` with a 400, so for them
/// [`lift_tool_result_images`] moves it out. An id names the vendor as `<vendor>.…` or
/// `<geo>.<vendor>.…`; an inference-profile ARN names none.
fn takes_images_outside_tool_results(model: &str) -> bool {
    model.split('.').take(2).any(|s| s == "openai")
}

/// Moves the images out of each user turn's `toolResult` blocks to the end of that turn,
/// each after a line naming its tool call, and leaves a note in the result in its place.
fn lift_tool_result_images(messages: &mut serde_json::Value) {
    for msg in messages.as_array_mut().into_iter().flatten() {
        if msg["role"] != "user" {
            continue;
        }
        let Some(blocks) = msg["content"].as_array_mut() else {
            continue;
        };
        let mut lifted = Vec::new();
        for block in blocks.iter_mut() {
            let result = &mut block["toolResult"];
            let Some(content) = result["content"].as_array_mut() else {
                continue;
            };
            let (images, mut rest): (Vec<_>, Vec<_>) = std::mem::take(content)
                .into_iter()
                .partition(|c| c["image"].is_object());
            if !images.is_empty() {
                rest.push(json!({"text": "(the image is attached after the tool results)"}));
                lifted.push(
                    json!({"text": format!("Image from tool call {}:", result["toolUseId"])}),
                );
                lifted.extend(images);
            }
            result["content"] = rest.into();
        }
        blocks.append(&mut lifted);
    }
}

/// `ConverseStream` events, each framed as `{"<eventType>": body}`.
struct BedrockEvents;

impl EventParser for BedrockEvents {
    fn drain(&self, buf: &mut Vec<u8>) -> anyhow::Result<Vec<String>> {
        eventstream_drain(buf)
    }

    fn flush(&self, buf: &[u8]) -> Vec<String> {
        eventstream_flush(buf)
    }

    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let event: serde_json::Value = serde_json::from_str(data)?;
        let Some((kind, body)) = event.as_object().and_then(|o| o.iter().next()) else {
            anyhow::bail!("Converse stream event is not a single-key object: {data}");
        };
        let mut out = MessageDeltaOutput::new();
        match kind.as_str() {
            "messageStart" => out.delta = MessageDelta::new().with_role(Role::Assistant),
            "contentBlockStart" => {
                let tool = &body["start"]["toolUse"];
                if !tool.is_object() {
                    return Ok(None);
                }
                out.delta.tool_calls = vec![PartDelta::Function {
                    id: tool["toolUseId"].as_str().map(str::to_owned),
                    function: PartDeltaFunction::WithStringArgs {
                        name: tool["name"].as_str().unwrap_or_default().to_owned(),
                        arguments: String::new(),
                    },
                }];
            }
            "contentBlockDelta" => {
                let delta = &body["delta"];
                if let Some(text) = delta["text"].as_str() {
                    out.delta.contents = vec![PartDelta::Text {
                        text: text.to_owned(),
                    }];
                } else if delta["reasoningContent"].is_object() {
                    let reasoning = &delta["reasoningContent"];
                    out.delta.thinking = reasoning["text"].as_str().map(str::to_owned);
                    out.delta.signature = reasoning["signature"].as_str().map(str::to_owned);
                } else if let Some(input) = delta["toolUse"]["input"].as_str() {
                    out.delta.tool_calls = vec![PartDelta::Function {
                        id: None,
                        function: PartDeltaFunction::WithStringArgs {
                            name: String::new(),
                            arguments: input.to_owned(),
                        },
                    }];
                } else {
                    return Ok(None);
                }
            }
            "messageStop" => {
                out.finish_reason = body["stopReason"].as_str().map(parse_stop_reason);
            }
            "metadata" => out.usage = parse_usage(&body["usage"]),
            // `contentBlockStop`, and event types added later.
            _ => return Ok(None),
        }
        Ok(Some(out))
    }
}

/// A whole Converse response.
fn parse_response(response: &serde_json::Value) -> anyhow::Result<MessageOutput> {
    let mut contents = Vec::new();
    let mut thinking: Option<String> = None;
    let mut signature = None;
    let mut tool_calls = Vec::new();
    for block in response["output"]["message"]["content"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if let Some(text) = block["text"].as_str() {
            contents.push(Part::text(text));
        } else if block["toolUse"].is_object() {
            let tool = &block["toolUse"];
            tool_calls.push(Part::Function {
                id: tool["toolUseId"].as_str().unwrap_or_default().to_owned(),
                function: PartFunction {
                    name: tool["name"].as_str().unwrap_or_default().to_owned(),
                    arguments: Value::from(tool["input"].clone()),
                },
            });
        } else if block["reasoningContent"]["reasoningText"].is_object() {
            let reasoning = &block["reasoningContent"]["reasoningText"];
            thinking
                .get_or_insert_default()
                .push_str(reasoning["text"].as_str().unwrap_or_default());
            signature = reasoning["signature"]
                .as_str()
                .map(str::to_owned)
                .or(signature);
        }
    }
    let finish_reason = response["stopReason"]
        .as_str()
        .map(parse_stop_reason)
        .ok_or_else(|| anyhow::anyhow!("Converse response has no stopReason: {response}"))?;
    Ok(assistant_output(
        contents,
        thinking,
        signature,
        tool_calls,
        finish_reason,
        parse_usage(&response["usage"]),
    ))
}

fn parse_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop {},
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length {},
        "tool_use" => FinishReason::ToolCall {},
        // `guardrail_intervened`, `content_filtered`, `malformed_*`.
        other => FinishReason::Refusal {
            reason: other.to_owned(),
        },
    }
}

fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    usage.is_object().then(|| TokenUsage {
        input_tokens: tokens(&usage["inputTokens"]).unwrap_or(0),
        output_tokens: tokens(&usage["outputTokens"]).unwrap_or(0),
        cache_creation_input_tokens: tokens(&usage["cacheWriteInputTokens"]),
        cache_read_input_tokens: tokens(&usage["cacheReadInputTokens"]),
    })
}
