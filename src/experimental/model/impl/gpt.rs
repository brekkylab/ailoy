use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{
    EventParser, assistant_output, bearer, close_objects, last_user_index, request_json,
    request_stream, system_text, tokens, value_text,
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

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";

/// Request options for [`Gpt`]. `None` leaves a field out of the request, so the API's
/// default applies. What a model accepts differs by family and is not checked here: a
/// field the model rejects (e.g. `temperature` on a reasoning model) surfaces as an API
/// error.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GptOption {
    /// Output token cap per response, reasoning tokens included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// How a reasoning model thinks; sent as `reasoning`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<GptReasoning>,

    /// Constrains the response to a JSON schema; sent as `text.format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<GptOutputFormat>,

    /// Whether and which tools the model must call; sent as `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<GptToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// The `reasoning` request field, in its wire shape.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GptReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<GptEffort>,

    /// Whether reasoning comes back as a summary; none does unless asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<GptReasoningSummary>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptEffort {
    /// No reasoning; from GPT-5.1 on.
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

/// The `text.format` request field. Always sent strict, under the name `response`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GptOutputFormat {
    JsonSchema { schema: Value },
}

/// The `tool_choice` request field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GptToolChoice {
    /// The model decides; the API default when tools are given.
    Auto,
    /// Some tool must be called.
    Required,
    /// No tool may be called.
    None,
    /// The named function must be called.
    Function { name: String },
}

impl GptToolChoice {
    fn to_wire(&self) -> serde_json::Value {
        match self {
            Self::Auto => "auto".into(),
            Self::Required => "required".into(),
            Self::None => "none".into(),
            Self::Function { name } => serde_json::json!({"type": "function", "name": name}),
        }
    }
}

/// A GPT model on the OpenAI Responses API.
#[derive(Clone, Debug)]
pub struct Gpt {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `openai` entry each request uses.
    provider: String,
    option: GptOption,
}

impl Gpt {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: GptOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: GptOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &GptOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut GptOption {
        &mut self.option
    }

    /// The Responses API body. System messages are joined into `instructions`.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        stream: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut body = json!({
            "model": self.model,
            "input": wire_input(messages),
            "stream": stream,
        });
        let fields = body.as_object_mut().unwrap();

        let instructions = system_text(messages);
        if !instructions.is_empty() {
            fields.insert("instructions".into(), instructions.into());
        }
        if !tools.is_empty() {
            fields.insert("tools".into(), tools.iter().map(wire_tool).collect());
        }
        if let Some(max_output_tokens) = option.max_output_tokens {
            fields.insert("max_output_tokens".into(), max_output_tokens.into());
        }
        if let Some(temperature) = option.temperature {
            fields.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = option.top_p {
            fields.insert("top_p".into(), top_p.into());
        }
        if let Some(reasoning) = &option.reasoning {
            fields.insert("reasoning".into(), serde_json::to_value(reasoning)?);
        }
        if let Some(tool_choice) = &option.tool_choice {
            fields.insert("tool_choice".into(), tool_choice.to_wire());
        }
        if let Some(parallel_tool_calls) = option.parallel_tool_calls {
            fields.insert("parallel_tool_calls".into(), parallel_tool_calls.into());
        }
        if let Some(GptOutputFormat::JsonSchema { schema }) = &option.format {
            // Strict mode needs `additionalProperties: false` on every object.
            fields.insert(
                "text".into(),
                json!({"format": {
                    "type": "json_schema",
                    "name": "response",
                    "strict": true,
                    "schema": close_objects(&schema.clone().into()),
                }}),
            );
        }
        Ok(body)
    }

    /// The API key as a bearer token, and for a stream the event-stream `accept`.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let ProviderEntry::ApiKey(key) = find_entry(&self.provider, "openai")? else {
            anyhow::bail!("the OpenAI API takes an API key");
        };
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer(&key)?);
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        Ok(headers)
    }
}

impl LangModelInference for Gpt {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let request = self
            .request_headers(true)
            .and_then(|headers| Ok((headers, self.request_body(messages, tools, true)?)));
        request_stream(
            RESPONSES_URL.to_owned(),
            request,
            is_permanent_quota,
            GptEvents,
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
            parse_response(&request_json(RESPONSES_URL, headers, &body, is_permanent_quota).await?)
        })
    }
}

/// `input` items for the Responses API. System messages are left out, as they go in
/// `instructions`. Thinking is replayed, as a reasoning summary, only for assistant turns
/// after the last user message.
fn wire_input(messages: &[Message]) -> serde_json::Value {
    let last_user = last_user_index(messages);
    messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role != Role::System)
        .flat_map(|(i, m)| wire_items(m, i > last_user))
        .collect()
}

/// A tool message becomes one `function_call_output`. Any other is its reasoning, a
/// message holding its contents, and one `function_call` per tool call, each only if
/// present.
fn wire_items(msg: &Message, with_thinking: bool) -> Vec<serde_json::Value> {
    if msg.role == Role::Tool {
        let output: Vec<_> = msg
            .contents
            .iter()
            .filter(|p| !p.is_function())
            .map(|p| wire_content(p, "input_text"))
            .collect();
        return vec![json!({
            "type": "function_call_output",
            "call_id": msg.id.as_deref().unwrap_or_default(),
            "output": output,
        })];
    }

    let mut items = Vec::new();
    if with_thinking && let Some(thinking) = msg.thinking.as_deref().filter(|t| !t.is_empty()) {
        items.push(json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": thinking}],
        }));
    }
    let text_type = if msg.role == Role::Assistant {
        "output_text"
    } else {
        "input_text"
    };
    let content: Vec<_> = msg
        .contents
        .iter()
        .filter(|p| !p.is_function())
        .map(|p| wire_content(p, text_type))
        .collect();
    if !content.is_empty() {
        items.push(json!({"role": msg.role.to_string(), "content": content}));
    }
    items.extend(msg.tool_calls.iter().flatten().filter_map(|p| match p {
        Part::Function { id, function } => Some(json!({
            "type": "function_call",
            "call_id": id,
            "name": function.name,
            "arguments": serde_json::to_string(&function.arguments).unwrap_or_default(),
        })),
        _ => None,
    }));
    items
}

/// A content part; text (and a value, as text) is typed `text_type`.
fn wire_content(part: &Part, text_type: &str) -> serde_json::Value {
    match part {
        Part::Text { text } => json!({"type": text_type, "text": text}),
        Part::Value { value } => json!({"type": text_type, "text": value_text(value)}),
        Part::Image { image } => {
            let url = match image {
                PartImage::Embedded { mime_type, data } => {
                    format!("data:{mime_type};base64,{}", data.base64())
                }
                PartImage::Url { url } => url.clone(),
            };
            json!({"type": "input_image", "image_url": url})
        }
        Part::Function { .. } => unreachable!("function parts are filtered out"),
    }
}

fn wire_tool(tool: &ToolDesc) -> serde_json::Value {
    let mut wire = json!({
        "type": "function",
        "name": tool.name,
        "parameters": serde_json::Value::from(tool.parameters.clone()),
    });
    if let Some(description) = &tool.description {
        wire["description"] = description.as_str().into();
    }
    wire
}

/// `insufficient_quota` never clears by waiting; other 429s are rate limits.
fn is_permanent_quota(body: &str) -> bool {
    let Ok(body) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    body["error"]["type"] == "insufficient_quota" || body["error"]["code"] == "insufficient_quota"
}

/// The Responses API's server-sent events. A function call is taken whole from its
/// `output_item.done`, so its argument deltas are skipped; the terminal response gives the
/// finish reason and usage.
struct GptEvents;

impl EventParser for GptEvents {
    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let event: serde_json::Value = serde_json::from_str(data)?;
        // Every delta carries the role: a reasoning model cut off mid-reasoning sends only
        // reasoning before the terminal event.
        let mut out = MessageDeltaOutput::new();
        out.delta = MessageDelta::new().with_role(Role::Assistant);
        match event["type"].as_str().unwrap_or_default() {
            "response.output_text.delta" => {
                out.delta.contents = vec![PartDelta::Text {
                    text: event["delta"].as_str().unwrap_or_default().to_owned(),
                }];
            }
            "response.reasoning_summary_text.delta" => {
                out.delta.thinking = event["delta"].as_str().map(str::to_owned);
            }
            "response.output_item.done" if event["item"]["type"] == "function_call" => {
                let item = &event["item"];
                out.delta.tool_calls = vec![PartDelta::Function {
                    id: item["call_id"].as_str().map(str::to_owned),
                    function: PartDeltaFunction::WithStringArgs {
                        name: item["name"].as_str().unwrap_or_default().to_owned(),
                        arguments: item["arguments"].as_str().unwrap_or_default().to_owned(),
                    },
                }];
            }
            "response.completed" | "response.incomplete" => {
                let response = &event["response"];
                out.finish_reason = Some(parse_finish_reason(response));
                out.usage = parse_usage(&response["usage"]);
            }
            "response.failed" => anyhow::bail!(
                "GPT response failed: {}",
                event["response"]["error"]["message"]
                    .as_str()
                    .unwrap_or("(no message)")
            ),
            "error" => anyhow::bail!(
                "GPT stream error: {}",
                event["message"]
                    .as_str()
                    .or(event["error"]["message"].as_str())
                    .unwrap_or("(no message)")
            ),
            // Lifecycle events, and deltas taken whole elsewhere.
            _ => return Ok(None),
        }
        Ok(Some(out))
    }
}

/// A whole Responses API response: message text, reasoning summaries and function calls,
/// in output order.
fn parse_response(response: &serde_json::Value) -> anyhow::Result<MessageOutput> {
    if response["status"] == "failed" {
        anyhow::bail!(
            "GPT response failed: {}",
            response["error"]["message"]
                .as_str()
                .unwrap_or("(no message)")
        );
    }
    let mut contents = Vec::new();
    let mut thinking: Option<String> = None;
    let mut tool_calls = Vec::new();
    for item in response["output"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") => {
                contents.extend(
                    item["content"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter(|c| c["type"] == "output_text")
                        .map(|c| Part::text(c["text"].as_str().unwrap_or_default())),
                );
            }
            Some("reasoning") => {
                for summary in item["summary"].as_array().into_iter().flatten() {
                    thinking
                        .get_or_insert_default()
                        .push_str(summary["text"].as_str().unwrap_or_default());
                }
            }
            Some("function_call") => {
                let arguments = item["arguments"].as_str().unwrap_or("{}");
                tool_calls.push(Part::Function {
                    id: item["call_id"].as_str().unwrap_or_default().to_owned(),
                    function: PartFunction {
                        name: item["name"].as_str().unwrap_or_default().to_owned(),
                        arguments: serde_json::from_str::<serde_json::Value>(arguments)?.into(),
                    },
                });
            }
            _ => {}
        }
    }
    Ok(assistant_output(
        contents,
        thinking,
        None,
        tool_calls,
        parse_finish_reason(response),
        parse_usage(&response["usage"]),
    ))
}

/// A response that completed with a function call in its output stopped for it.
fn parse_finish_reason(response: &serde_json::Value) -> FinishReason {
    match response["status"].as_str() {
        Some("incomplete") => match response["incomplete_details"]["reason"].as_str() {
            Some("max_output_tokens") => FinishReason::Length {},
            reason => FinishReason::Refusal {
                reason: reason.unwrap_or("incomplete").to_owned(),
            },
        },
        _ if response["output"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|item| item["type"] == "function_call") =>
        {
            FinishReason::ToolCall {}
        }
        _ => FinishReason::Stop {},
    }
}

fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    usage.is_object().then(|| TokenUsage {
        input_tokens: tokens(&usage["input_tokens"]).unwrap_or(0),
        output_tokens: tokens(&usage["output_tokens"]).unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: tokens(&usage["input_tokens_details"]["cached_tokens"]),
    })
}
