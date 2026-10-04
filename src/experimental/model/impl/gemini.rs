use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{
    EventParser, assistant_output, bearer, request_json, request_stream, secret_header,
    system_text, tokens, value_text,
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

const MODELS_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Request options for [`Gemini`]; all but `tool_choice` go under `generationConfig`.
/// `None` leaves a field out of the request, so the API's default applies. What a model
/// accepts differs by generation and is not checked here: a field the model rejects
/// surfaces as an API error.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GeminiOption {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,

    /// Strings that end the response when generated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// How the model thinks; sent as `thinkingConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<GeminiThinking>,

    /// Constrains the response to a JSON schema; sent as `responseJsonSchema`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<GeminiOutputFormat>,

    /// Whether and which tools the model must call; sent as
    /// `toolConfig.functionCallingConfig`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<GeminiToolChoice>,
}

/// The `thinkingConfig` request field, in its wire shape. Gemini 2.x takes a budget,
/// Gemini 3 onwards a level.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiThinking {
    /// Thinking tokens; `0` turns thinking off and `-1` lets the model decide.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<GeminiThinkingLevel>,

    /// Whether thought summaries come back; none do unless asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

/// Sent as `responseMimeType: application/json` with `responseJsonSchema`, which takes
/// JSON Schema as-is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeminiOutputFormat {
    JsonSchema { schema: Value },
}

/// The function-calling mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeminiToolChoice {
    /// The model decides; the API default.
    Auto,
    /// Some function must be called, from `allowed_function_names` if non-empty.
    Any {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_function_names: Vec<String>,
    },
    /// No function may be called.
    None,
}

impl GeminiToolChoice {
    fn to_wire(&self) -> serde_json::Value {
        let config = match self {
            Self::Auto => serde_json::json!({"mode": "AUTO"}),
            Self::Any {
                allowed_function_names,
            } if !allowed_function_names.is_empty() => serde_json::json!({
                "mode": "ANY",
                "allowedFunctionNames": allowed_function_names,
            }),
            Self::Any { .. } => serde_json::json!({"mode": "ANY"}),
            Self::None => serde_json::json!({"mode": "NONE"}),
        };
        serde_json::json!({"functionCallingConfig": config})
    }
}

/// A Gemini model on the Gemini API.
#[derive(Clone, Debug)]
pub struct Gemini {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `gemini` credential each request uses.
    provider: String,
    option: GeminiOption,
}

impl Gemini {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: GeminiOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: GeminiOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &GeminiOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut GeminiOption {
        &mut self.option
    }

    /// Streaming is picked by the endpoint, not a body flag.
    fn request_url(&self, stream: bool) -> String {
        if stream {
            format!("{MODELS_URL}/{}:streamGenerateContent?alt=sse", self.model)
        } else {
            format!("{MODELS_URL}/{}:generateContent", self.model)
        }
    }

    /// The `generateContent` body. System messages are joined into `system_instruction`.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut body = json!({"contents": wire_contents(messages)?});
        let fields = body.as_object_mut().unwrap();

        let system = system_text(messages);
        if !system.is_empty() {
            fields.insert(
                "system_instruction".into(),
                json!({"parts": [{"text": system}]}),
            );
        }
        if !tools.is_empty() {
            let declarations: Vec<_> = tools.iter().map(wire_tool).collect();
            fields.insert(
                "tools".into(),
                json!([{"functionDeclarations": declarations}]),
            );
        }
        if let Some(tool_choice) = &option.tool_choice {
            fields.insert("toolConfig".into(), tool_choice.to_wire());
        }

        let mut config = serde_json::Map::new();
        if let Some(max_output_tokens) = option.max_output_tokens {
            config.insert("maxOutputTokens".into(), max_output_tokens.into());
        }
        if let Some(temperature) = option.temperature {
            config.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = option.top_p {
            config.insert("topP".into(), top_p.into());
        }
        if let Some(top_k) = option.top_k {
            config.insert("topK".into(), top_k.into());
        }
        if !option.stop_sequences.is_empty() {
            config.insert("stopSequences".into(), json!(option.stop_sequences));
        }
        if let Some(thinking) = &option.thinking {
            config.insert("thinkingConfig".into(), serde_json::to_value(thinking)?);
        }
        if let Some(GeminiOutputFormat::JsonSchema { schema }) = &option.format {
            config.insert("responseMimeType".into(), "application/json".into());
            config.insert(
                "responseJsonSchema".into(),
                serde_json::Value::from(schema.clone()),
            );
        }
        if !config.is_empty() {
            fields.insert("generationConfig".into(), config.into());
        }
        Ok(body)
    }

    /// An API key goes in `x-goog-api-key`; an OAuth token as a bearer token.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        match find_credential(&self.provider, "gemini")? {
            Credential::ApiKey(key) => headers.insert("x-goog-api-key", secret_header(&key)?),
            Credential::OAuthToken(token) => headers.insert(AUTHORIZATION, bearer(&token)?),
            Credential::Bedrock { .. } => {
                anyhow::bail!("the Gemini API takes an API key or OAuth token")
            }
        };
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        Ok(headers)
    }
}

impl LangModelInference for Gemini {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let request = self
            .request_headers(true)
            .and_then(|headers| Ok((headers, self.request_body(messages, tools)?)));
        request_stream(
            self.request_url(true),
            request,
            is_permanent_quota,
            GeminiEvents::default(),
        )
    }

    fn infer<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let headers = self.request_headers(false)?;
            let body = self.request_body(messages, tools)?;
            let url = self.request_url(false);
            parse_response(&request_json(&url, headers, &body, is_permanent_quota).await?)
        })
    }
}

/// `contents` for the Gemini API. System messages are left out, as they go in
/// `system_instruction`.
fn wire_contents(messages: &[Message]) -> anyhow::Result<serde_json::Value> {
    messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(wire_content)
        .collect()
}

/// A tool message becomes a user turn holding its `functionResponse`, with any images
/// beside it, as a REST function response has no room for them. The assistant is `model`.
///
/// Thought text is not sent back, only its signature, which goes on the first function
/// call or else on the last part, where the model put it.
fn wire_content(msg: &Message) -> anyhow::Result<serde_json::Value> {
    if msg.role == Role::Tool {
        // The call id is `<name>/<id>`; the response is matched by name.
        let id = msg.id.as_deref().unwrap_or_default();
        let name = id.split_once('/').map_or(id, |(name, _)| name);
        let mut results = Vec::new();
        let mut images = Vec::new();
        for part in &msg.contents {
            match part {
                Part::Text { text } => results.push(json!(text)),
                Part::Value { value } => results.push(value.clone().into()),
                Part::Image { image } => images.push(wire_image(image)?),
                Part::Function { .. } => {}
            }
        }
        let result = match results.len() {
            1 => results.pop().unwrap(),
            _ => results.into(),
        };
        let mut parts = vec![json!({
            "functionResponse": {"name": name, "response": {"result": result}},
        })];
        parts.extend(images);
        return Ok(json!({"role": "user", "parts": parts}));
    }

    let mut parts = Vec::new();
    for part in &msg.contents {
        parts.push(match part {
            Part::Text { text } => json!({"text": text}),
            Part::Value { value } => json!({"text": value_text(value)}),
            Part::Image { image } => wire_image(image)?,
            Part::Function { .. } => continue,
        });
    }
    let first_call = parts.len();
    parts.extend(msg.tool_calls.iter().flatten().filter_map(|p| match p {
        Part::Function { function, .. } => Some(json!({
            "functionCall": {
                "name": function.name,
                "args": serde_json::Value::from(function.arguments.clone()),
            },
        })),
        _ => None,
    }));
    if let Some(signature) = &msg.signature {
        let at = if first_call < parts.len() {
            Some(first_call)
        } else {
            parts.len().checked_sub(1)
        };
        if let Some(at) = at {
            parts[at]["thoughtSignature"] = signature.as_str().into();
        }
    }
    let role = if msg.role == Role::Assistant {
        "model"
    } else {
        "user"
    };
    Ok(json!({"role": role, "parts": parts}))
}

/// Embedded bytes, or a base64 `data:` URL, as `inline_data`. Gemini fetches no other
/// image URL.
fn wire_image(image: &PartImage) -> anyhow::Result<serde_json::Value> {
    let (mime_type, data) = match image {
        PartImage::Embedded { mime_type, data } => (mime_type.clone(), data.base64()),
        PartImage::Url { url } => {
            let (mime_type, data) = url
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(";base64,"))
                .ok_or_else(|| {
                    anyhow::anyhow!("Gemini takes no image URLs; embed the image instead")
                })?;
            (mime_type.to_owned(), data.to_owned())
        }
    };
    Ok(json!({"inline_data": {"mime_type": mime_type, "data": data}}))
}

fn wire_tool(tool: &ToolDesc) -> serde_json::Value {
    let mut wire = json!({
        "name": tool.name,
        "parametersJsonSchema": serde_json::Value::from(tool.parameters.clone()),
    });
    if let Some(description) = &tool.description {
        wire["description"] = description.as_str().into();
    }
    wire
}

/// `RESOURCE_EXHAUSTED` is both a rate limit and an exhausted quota; only a rate limit
/// says when to retry.
fn is_permanent_quota(body: &str) -> bool {
    let Ok(body) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let error = &body["error"];
    error["status"] == "RESOURCE_EXHAUSTED"
        && !error["details"].as_array().into_iter().flatten().any(|d| {
            d["@type"]
                .as_str()
                .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
        })
}

/// The stream's chunks, each a partial `GenerateContentResponse`. Gemini reports `STOP`
/// for a turn that called functions, so the parser remembers whether one came.
#[derive(Default)]
struct GeminiEvents {
    called: bool,
}

impl EventParser for GeminiEvents {
    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let chunk: serde_json::Value = serde_json::from_str(data)?;
        let candidate = &chunk["candidates"][0];
        let mut out = MessageDeltaOutput::new();
        // Every chunk carries the role: a finish-only one may come first.
        out.delta = MessageDelta::new().with_role(Role::Assistant);
        for part in candidate["content"]["parts"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(signature) = part["thoughtSignature"].as_str() {
                out.delta.signature = Some(signature.to_owned());
            }
            if let Some(text) = part["text"].as_str() {
                if part["thought"] == true {
                    out.delta.thinking.get_or_insert_default().push_str(text);
                } else {
                    out.delta.contents.push(PartDelta::Text {
                        text: text.to_owned(),
                    });
                }
            } else if part["functionCall"].is_object() {
                self.called = true;
                let (id, name, arguments) = parse_call(&part["functionCall"]);
                out.delta.tool_calls.push(PartDelta::Function {
                    id: Some(id),
                    function: PartDeltaFunction::WithParsedArgs { name, arguments },
                });
            }
        }
        out.finish_reason = parse_finish_reason(&chunk, self.called);
        out.usage = parse_usage(&chunk["usageMetadata"]);
        Ok(Some(out))
    }
}

/// A whole `GenerateContentResponse`, from its first candidate.
fn parse_response(response: &serde_json::Value) -> anyhow::Result<MessageOutput> {
    let mut contents = Vec::new();
    let mut thinking: Option<String> = None;
    let mut signature = None;
    let mut tool_calls = Vec::new();
    for part in response["candidates"][0]["content"]["parts"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if let Some(s) = part["thoughtSignature"].as_str() {
            signature = Some(s.to_owned());
        }
        if let Some(text) = part["text"].as_str() {
            if part["thought"] == true {
                thinking.get_or_insert_default().push_str(text);
            } else {
                contents.push(Part::text(text));
            }
        } else if part["functionCall"].is_object() {
            let (id, name, arguments) = parse_call(&part["functionCall"]);
            tool_calls.push(Part::Function {
                id,
                function: PartFunction { name, arguments },
            });
        }
    }
    let finish_reason = parse_finish_reason(response, !tool_calls.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Gemini response has no finish reason: {response}"))?;
    Ok(assistant_output(
        contents,
        thinking,
        signature,
        tool_calls,
        finish_reason,
        parse_usage(&response["usageMetadata"]),
    ))
}

/// A function call as `(id, name, args)`. The id is `<name>/<id>`, Gemini's own id when it
/// gives one, so a tool result can name the function it answers.
fn parse_call(call: &serde_json::Value) -> (String, String, Value) {
    let name = call["name"].as_str().unwrap_or_default().to_owned();
    let id = match call["id"].as_str() {
        Some(id) => format!("{name}/{id}"),
        None => format!(
            "{name}/call-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        ),
    };
    let arguments = match &call["args"] {
        serde_json::Value::Null => json!({}),
        args => args.clone(),
    };
    (id, name, arguments.into())
}

/// The first candidate's finish reason, with `STOP` read as a tool call when `called`. A
/// prompt blocked before any candidate is a refusal.
fn parse_finish_reason(response: &serde_json::Value, called: bool) -> Option<FinishReason> {
    if let Some(reason) = response["promptFeedback"]["blockReason"].as_str() {
        return Some(FinishReason::Refusal {
            reason: reason.to_owned(),
        });
    }
    Some(match response["candidates"][0]["finishReason"].as_str()? {
        "STOP" if called => FinishReason::ToolCall {},
        "STOP" => FinishReason::Stop {},
        "MAX_TOKENS" => FinishReason::Length {},
        other => FinishReason::Refusal {
            reason: other.to_owned(),
        },
    })
}

/// Thinking tokens are billed as output, so they count toward it.
fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    usage.is_object().then(|| TokenUsage {
        input_tokens: tokens(&usage["promptTokenCount"]).unwrap_or(0),
        output_tokens: tokens(&usage["candidatesTokenCount"]).unwrap_or(0)
            + tokens(&usage["thoughtsTokenCount"]).unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: tokens(&usage["cachedContentTokenCount"]),
    })
}
