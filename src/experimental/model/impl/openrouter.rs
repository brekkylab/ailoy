use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{bearer, chat, close_objects, rate_limit_only, request_json, request_stream};
use crate::{
    datatype::Value,
    experimental::model::{Credential, LangModelInference, find_credential},
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

const CHAT_COMPLETIONS_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Request options for [`OpenRouter`]. `None` leaves a field out of the request, so the
/// default applies. OpenRouter passes each field on to the routed model, which may ignore
/// or reject it: that is not checked here.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterOption {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u64>,

    /// Strings that end the response when generated; sent as `stop`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,

    /// OpenRouter's model-neutral thinking control; sent as `reasoning`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<OpenRouterReasoning>,

    /// Constrains the response to a JSON schema; sent as `response_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OpenRouterOutputFormat>,

    /// Whether and which tools the model must call; sent as `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenRouterToolChoice>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

/// The `reasoning` request field, in its wire shape. Set `effort` or `max_tokens`, not
/// both; OpenRouter maps either onto the routed model's own control.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct OpenRouterReasoning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<OpenRouterEffort>,

    /// A thinking-token budget, for models that take one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    /// Thinks without returning the reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,

    /// Turns thinking on at the model's default effort.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
}

/// Sent as a strict `response_format`, under the name `response`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenRouterOutputFormat {
    JsonSchema { schema: Value },
}

/// The `tool_choice` request field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenRouterToolChoice {
    /// The model decides; the default when tools are given.
    Auto,
    /// Some tool must be called.
    Required,
    /// No tool may be called.
    None,
    /// The named function must be called.
    Function { name: String },
}

impl OpenRouterToolChoice {
    fn to_wire(&self) -> serde_json::Value {
        match self {
            Self::Auto => "auto".into(),
            Self::Required => "required".into(),
            Self::None => "none".into(),
            Self::Function { name } => {
                serde_json::json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

/// A model routed by OpenRouter over its Chat Completions API. The model id is
/// OpenRouter's `<vendor>/<model>`, e.g. `anthropic/claude-sonnet-5`.
#[derive(Clone, Debug)]
pub struct OpenRouter {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `openrouter` credential each request uses.
    provider: String,
    option: OpenRouterOption,
}

/// Where thinking comes back, and goes back in.
const REASONING_FIELD: &str = "reasoning";

impl OpenRouter {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: OpenRouterOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: OpenRouterOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &OpenRouterOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut OpenRouterOption {
        &mut self.option
    }

    /// The Chat Completions body. System messages stay in `messages`.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        stream: bool,
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut body = json!({
            "model": self.model,
            "messages": chat::wire_messages(messages, REASONING_FIELD),
            "stream": stream,
        });
        let fields = body.as_object_mut().unwrap();

        if stream {
            fields.insert("stream_options".into(), json!({"include_usage": true}));
        }
        if !tools.is_empty() {
            fields.insert("tools".into(), tools.iter().map(chat::wire_tool).collect());
        }
        if let Some(max_tokens) = option.max_tokens {
            fields.insert("max_tokens".into(), max_tokens.into());
        }
        if let Some(temperature) = option.temperature {
            fields.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = option.top_p {
            fields.insert("top_p".into(), top_p.into());
        }
        if let Some(top_k) = option.top_k {
            fields.insert("top_k".into(), top_k.into());
        }
        if !option.stop.is_empty() {
            fields.insert("stop".into(), json!(option.stop));
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
        if let Some(OpenRouterOutputFormat::JsonSchema { schema }) = &option.format {
            fields.insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": true,
                        "schema": close_objects(&schema.clone().into()),
                    },
                }),
            );
        }
        Ok(body)
    }

    /// The API key as a bearer token, and for a stream the event-stream `accept`.
    /// OpenRouter's OAuth flow ends in an API key too.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let Credential::ApiKey(key) = find_credential(&self.provider, "openrouter")? else {
            anyhow::bail!("OpenRouter takes an API key");
        };
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer(&key)?);
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        Ok(headers)
    }
}

impl LangModelInference for OpenRouter {
    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let request = self
            .request_headers(true)
            .and_then(|headers| Ok((headers, self.request_body(messages, tools, true)?)));
        request_stream(
            CHAT_COMPLETIONS_URL.to_owned(),
            request,
            rate_limit_only,
            chat::ChatEvents::new(REASONING_FIELD),
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
            let response =
                request_json(CHAT_COMPLETIONS_URL, headers, &body, rate_limit_only).await?;
            chat::parse_response(&response, REASONING_FIELD)
        })
    }
}
