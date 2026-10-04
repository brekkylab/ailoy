use futures::{future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::utils::{bearer, chat, request_json, request_stream};
use crate::{
    experimental::model::{Credential, LangModelInference, find_credential},
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

const CHAT_COMPLETIONS_URL: &str = "https://api.moonshot.ai/v1/chat/completions";

/// Where thinking comes back, and goes back in.
const REASONING_FIELD: &str = "reasoning_content";

/// Request options for [`Kimi`]. `None` leaves a field out of the request, so the API's
/// default applies. What a model accepts differs by model and is not checked here: a
/// field the model rejects surfaces as an API error.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct KimiOption {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    /// Strings that end the response when generated; sent as `stop`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,

    /// Whether the model thinks; sent as `thinking`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<KimiThinking>,

    /// Constrains the response to JSON; sent as `response_format`. JSON schemas are not taken.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<KimiOutputFormat>,

    /// Whether and which tools the model must call; sent as `tool_choice`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<KimiToolChoice>,
}

/// The `thinking` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KimiThinking {
    Enabled,
    Disabled,
}

/// The `response_format` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KimiOutputFormat {
    /// Some JSON object; the prompt has to ask for JSON and describe its shape.
    JsonObject,
}

/// The `tool_choice` request field.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KimiToolChoice {
    /// The model decides; the default when tools are given.
    Auto,
    /// Some tool must be called.
    Required,
    /// No tool may be called.
    None,
    /// The named function must be called.
    Function { name: String },
}

impl KimiToolChoice {
    fn to_wire(&self) -> serde_json::Value {
        match self {
            Self::Auto => "auto".into(),
            Self::Required => "required".into(),
            Self::None => "none".into(),
            Self::Function { name } => {
                json!({"type": "function", "function": {"name": name}})
            }
        }
    }
}

/// A Kimi model on the Moonshot API, e.g. `kimi-k2.5`.
#[derive(Clone, Debug)]
pub struct Kimi {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `moonshot` credential each request uses.
    provider: String,
    option: KimiOption,
}

impl Kimi {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: KimiOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: KimiOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &KimiOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut KimiOption {
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
        if !option.stop.is_empty() {
            fields.insert("stop".into(), json!(option.stop));
        }
        if let Some(thinking) = &option.thinking {
            fields.insert("thinking".into(), serde_json::to_value(thinking)?);
        }
        if let Some(format) = &option.format {
            fields.insert("response_format".into(), serde_json::to_value(format)?);
        }
        if let Some(tool_choice) = &option.tool_choice {
            fields.insert("tool_choice".into(), tool_choice.to_wire());
        }
        Ok(body)
    }

    /// The API key as a bearer token, and for a stream the event-stream `accept`.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let Credential::ApiKey(key) = find_credential(&self.provider, "moonshot")? else {
            anyhow::bail!("The Moonshot API takes an API key");
        };
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer(&key)?);
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        Ok(headers)
    }
}

impl LangModelInference for Kimi {
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
            is_permanent_quota,
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
                request_json(CHAT_COMPLETIONS_URL, headers, &body, is_permanent_quota).await?;
            chat::parse_response(&response, REASONING_FIELD)
        })
    }
}

/// `exceeded_current_quota_error` waits on a recharge; other 429s are rate limits.
fn is_permanent_quota(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .is_ok_and(|body| body["error"]["type"] == "exceeded_current_quota_error")
}
