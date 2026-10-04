use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{
    EventParser, assistant_output, bearer, close_objects, last_user_index, rate_limit_only,
    request_json, request_stream, secret_header, system_text, tokens, value_text,
};
use crate::{
    datatype::Value,
    experimental::model::{Credential, LangModelInference, find_credential},
    message::{
        FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    tool::ToolDesc,
};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Sent when [`ClaudeOption::max_tokens`] is `None`. A non-streaming request that runs
/// toward this cap can take long enough to hit an HTTP timeout; stream those, or set a
/// lower cap.
const DEFAULT_MAX_TOKENS: u64 = 64000;

/// Request options for [`Claude`]. `None` leaves a field out of the request, so the API's
/// default applies. What a model accepts differs by generation and is not checked here: a
/// field the model rejects surfaces as an API error.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClaudeOption {
    /// Output token cap per response. Required by the API, so `None` means a default is
    /// filled in at request time.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// How the model thinks; sent as `thinking`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ClaudeThinking>,

    /// Thinking depth and overall token spend; sent as `output_config.effort`. The API
    /// default is `high`, except `medium` on Claude Opus 5.5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ClaudeEffort>,

    /// Constrains the response to a JSON schema; sent as `output_config.format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ClaudeOutputFormat>,

    /// Strings that end the response when generated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// Whether and which tools the model must call; sent as `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ClaudeToolChoice>,
}

/// The `thinking` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeThinking {
    /// The model decides when and how much to think. The default on current models.
    Adaptive {
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<ClaudeThinkingDisplay>,
    },
    /// A fixed thinking budget. Only for models before Claude 4.6, such as Haiku 4.5.
    Enabled { budget_tokens: u64 },
    /// No thinking. Rejected by Claude Fable 5/5.1, Opus 5.5 and Sonnet 5.5.
    Disabled,
    /// Thinking off except between tool calls; Claude Sonnet 5.5's way to turn it off.
    BetweenTools,
}

/// What thinking blocks carry back. Thinking is billed the same under every setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaudeThinkingDisplay {
    /// A readable summary of the reasoning.
    Summarized,
    /// Empty thinking text; the default on current models.
    Omitted,
    /// Progress notes between tool calls only (beta).
    Updates,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaudeEffort {
    Low,
    Medium,
    High,
    /// From Claude Opus 4.7 on.
    Xhigh,
    Max,
}

/// The `output_config.format` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeOutputFormat {
    JsonSchema { schema: Value },
}

/// The `tool_choice` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClaudeToolChoice {
    /// The model decides; the API default when tools are given.
    Auto {
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        disable_parallel_tool_use: bool,
    },
    /// Some tool must be called. Rejected by Claude Fable 5.1, Opus 5.5 and Sonnet 5.5.
    Any {
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        disable_parallel_tool_use: bool,
    },
    /// The named tool must be called. Rejected by the same models as `Any`.
    Tool {
        name: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        disable_parallel_tool_use: bool,
    },
    /// No tool may be called.
    None,
}

/// A Claude model on the Claude API.
#[derive(Clone, Debug)]
pub struct Claude {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `anthropic` credential each request uses.
    provider: String,
    option: ClaudeOption,
}

impl Claude {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: ClaudeOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: ClaudeOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &ClaudeOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut ClaudeOption {
        &mut self.option
    }

    /// The Messages API body. System messages are joined into the top-level `system`;
    /// `effort` and `format` go under `output_config`.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        stream: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut body = json!({
            "model": self.model,
            "max_tokens": option.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "messages": wire_messages(messages),
            "stream": stream,
        });
        let fields = body.as_object_mut().unwrap();

        let system = system_text(messages);
        if !system.is_empty() {
            fields.insert("system".into(), system.into());
        }
        if !tools.is_empty() {
            fields.insert("tools".into(), tools.iter().map(wire_tool).collect());
        }
        if let Some(thinking) = &option.thinking {
            fields.insert("thinking".into(), serde_json::to_value(thinking)?);
        }
        if let Some(tool_choice) = &option.tool_choice {
            fields.insert("tool_choice".into(), serde_json::to_value(tool_choice)?);
        }
        if !option.stop_sequences.is_empty() {
            fields.insert("stop_sequences".into(), json!(option.stop_sequences));
        }

        let mut output_config = serde_json::Map::new();
        if let Some(effort) = option.effort {
            output_config.insert("effort".into(), serde_json::to_value(effort)?);
        }
        if let Some(ClaudeOutputFormat::JsonSchema { schema }) = &option.format {
            // Structured outputs need `additionalProperties: false` on every object.
            let schema = close_objects(&schema.clone().into());
            output_config.insert(
                "format".into(),
                json!({"type": "json_schema", "schema": schema}),
            );
        }
        if !output_config.is_empty() {
            fields.insert("output_config".into(), output_config.into());
        }
        Ok(body)
    }

    /// Credentials, API version, and for a stream the event-stream `accept`. An API key
    /// goes in `x-api-key`; an OAuth token as a bearer token with the OAuth beta header.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let (name, credential) = match find_credential(&self.provider, "anthropic")? {
            Credential::ApiKey(key) => (HeaderName::from_static("x-api-key"), secret_header(&key)?),
            Credential::OAuthToken(token) => {
                headers.insert(
                    "anthropic-beta",
                    HeaderValue::from_static("oauth-2025-04-20"),
                );
                (AUTHORIZATION, bearer(&token)?)
            }
            Credential::Bedrock { .. } => {
                anyhow::bail!("the Claude API takes an API key or OAuth token; use Bedrock")
            }
        };
        headers.insert(name, credential);
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        #[cfg(target_arch = "wasm32")]
        headers.insert(
            "anthropic-dangerous-direct-browser-access",
            HeaderValue::from_static("true"),
        );
        Ok(headers)
    }
}

impl LangModelInference for Claude {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let request = self
            .request_headers(true)
            .and_then(|headers| Ok((headers, self.request_body(messages, tools, true)?)));
        request_stream(
            MESSAGES_URL.to_owned(),
            request,
            rate_limit_only,
            ClaudeEvents,
        )
    }

    fn infer<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let headers = self.request_headers(false)?;
            let body = self.request_body(messages, tools, false)?;
            parse_response(&request_json(MESSAGES_URL, headers, &body, rate_limit_only).await?)
        })
    }
}

/// `messages` for the Messages API. System messages are left out, as they go in the
/// top-level `system`. Thinking is replayed only for assistant turns after the last user
/// message, as the API requires.
fn wire_messages(messages: &[Message]) -> serde_json::Value {
    let last_user = last_user_index(messages);
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role != Role::System)
        .map(|(i, m)| wire_message(m, i > last_user))
        .collect()
}

/// A tool message becomes a user turn holding one `tool_result`; the others keep their
/// role, with thinking first, then contents, then tool calls.
fn wire_message(msg: &Message, with_thinking: bool) -> serde_json::Value {
    if msg.role == Role::Tool {
        let content: Vec<_> = msg
            .contents
            .iter()
            .filter_map(|part| match part {
                Part::Value { value } => Some(json!({"type": "text", "text": value_text(value)})),
                Part::Function { .. } => None,
                part => Some(wire_part(part)),
            })
            .collect();
        return json!({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": msg.id.as_deref().unwrap_or_default(),
                "content": content,
            }],
        });
    }

    let mut content = Vec::new();
    if with_thinking && let Some(thinking) = msg.thinking.as_deref().filter(|t| !t.is_empty()) {
        let mut block = json!({"type": "thinking", "thinking": thinking});
        if let Some(signature) = &msg.signature {
            block["signature"] = signature.as_str().into();
        }
        content.push(block);
    }
    content.extend(msg.contents.iter().map(wire_part));
    content.extend(msg.tool_calls.iter().flatten().map(wire_part));
    json!({"role": msg.role.to_string(), "content": content})
}

fn wire_part(part: &Part) -> serde_json::Value {
    match part {
        Part::Text { text } => json!({"type": "text", "text": text}),
        Part::Function { id, function } => json!({
            "type": "tool_use",
            "id": id,
            "name": function.name,
            "input": serde_json::Value::from(function.arguments.clone()),
        }),
        Part::Value { value } => value.clone().into(),
        Part::Image {
            image: PartImage::Embedded { mime_type, data },
        } => json!({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data.base64()},
        }),
        Part::Image {
            image: PartImage::Url { url },
        } => json!({"type": "image", "source": {"type": "url", "url": url}}),
    }
}

fn wire_tool(tool: &ToolDesc) -> serde_json::Value {
    let mut wire = json!({
        "name": tool.name,
        "input_schema": serde_json::Value::from(tool.parameters.clone()),
    });
    if let Some(description) = &tool.description {
        wire["description"] = description.as_str().into();
    }
    wire
}

/// The Messages API's server-sent events. Each event stands alone: a tool call's first
/// delta carries its id and name, and later argument fragments merge into it.
struct ClaudeEvents;

impl EventParser for ClaudeEvents {
    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let event: serde_json::Value = serde_json::from_str(data)?;
        let mut out = MessageDeltaOutput::new();
        match event["type"].as_str().unwrap_or_default() {
            "message_start" => {
                out.delta = MessageDelta::new().with_role(Role::Assistant);
                out.usage = parse_usage(&event["message"]["usage"]);
            }
            "content_block_start" => {
                let block = &event["content_block"];
                match block["type"].as_str() {
                    Some("tool_use") => {
                        out.delta.tool_calls = vec![PartDelta::Function {
                            id: block["id"].as_str().map(str::to_owned),
                            function: PartDeltaFunction::WithStringArgs {
                                name: block["name"].as_str().unwrap_or_default().to_owned(),
                                arguments: String::new(),
                            },
                        }];
                    }
                    _ => return Ok(None),
                }
            }
            "content_block_delta" => {
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        out.delta.contents = vec![PartDelta::Text {
                            text: delta["text"].as_str().unwrap_or_default().to_owned(),
                        }];
                    }
                    Some("thinking_delta") => {
                        out.delta.thinking = delta["thinking"].as_str().map(str::to_owned);
                    }
                    Some("signature_delta") => {
                        out.delta.signature = delta["signature"].as_str().map(str::to_owned);
                    }
                    Some("input_json_delta") => {
                        out.delta.tool_calls = vec![PartDelta::Function {
                            id: None,
                            function: PartDeltaFunction::WithStringArgs {
                                name: String::new(),
                                arguments: delta["partial_json"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_owned(),
                            },
                        }];
                    }
                    _ => return Ok(None),
                }
            }
            "message_delta" => {
                out.finish_reason = event["delta"]["stop_reason"]
                    .as_str()
                    .map(parse_stop_reason);
                out.usage = parse_usage(&event["usage"]);
            }
            "error" => anyhow::bail!(
                "Claude stream error ({}): {}",
                event["error"]["type"].as_str().unwrap_or("unknown"),
                event["error"]["message"].as_str().unwrap_or("(no message)"),
            ),
            // `ping`, `content_block_stop`, `message_stop`, and event types added later.
            _ => return Ok(None),
        }
        Ok(Some(out))
    }
}

/// A whole Messages API response. Thinking blocks concatenate, keeping the last signature;
/// redacted thinking is dropped.
fn parse_response(response: &serde_json::Value) -> anyhow::Result<MessageOutput> {
    let mut contents = Vec::new();
    let mut thinking: Option<String> = None;
    let mut signature = None;
    let mut tool_calls = Vec::new();
    for block in response["content"].as_array().into_iter().flatten() {
        match block["type"].as_str() {
            Some("text") => contents.push(Part::text(block["text"].as_str().unwrap_or_default())),
            Some("thinking") => {
                thinking
                    .get_or_insert_default()
                    .push_str(block["thinking"].as_str().unwrap_or_default());
                signature = block["signature"].as_str().map(str::to_owned).or(signature);
            }
            Some("tool_use") => tool_calls.push(Part::Function {
                id: block["id"].as_str().unwrap_or_default().to_owned(),
                function: PartFunction {
                    name: block["name"].as_str().unwrap_or_default().to_owned(),
                    arguments: block["input"].clone().into(),
                },
            }),
            _ => {}
        }
    }
    let finish_reason = response["stop_reason"]
        .as_str()
        .map(parse_stop_reason)
        .ok_or_else(|| anyhow::anyhow!("Claude response has no stop_reason: {response}"))?;
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
        "end_turn" | "pause_turn" | "stop_sequence" => FinishReason::Stop {},
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length {},
        "tool_use" => FinishReason::ToolCall {},
        other => FinishReason::Refusal {
            reason: other.to_owned(),
        },
    }
}

fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    usage.is_object().then(|| TokenUsage {
        input_tokens: tokens(&usage["input_tokens"]).unwrap_or(0),
        output_tokens: tokens(&usage["output_tokens"]).unwrap_or(0),
        cache_creation_input_tokens: tokens(&usage["cache_creation_input_tokens"]),
        cache_read_input_tokens: tokens(&usage["cache_read_input_tokens"]),
    })
}
