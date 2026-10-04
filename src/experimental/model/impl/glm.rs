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

const CHAT_COMPLETIONS_URL: &str = "https://api.z.ai/api/paas/v4/chat/completions";

/// Where thinking comes back, and goes back in.
const REASONING_FIELD: &str = "reasoning_content";

/// Request options for [`Glm`]. `None` leaves a field out of the request, so the API's
/// default applies. What a model accepts differs by model and is not checked here: a
/// field the model rejects surfaces as an API error.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GlmOption {
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
    pub thinking: Option<GlmThinking>,

    /// Constrains the response to JSON; sent as `response_format`. JSON schemas are not taken.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<GlmOutputFormat>,
}

/// The `thinking` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GlmThinking {
    Enabled {
        /// `false` keeps earlier turns' thinking in context ("preserved thinking"); it
        /// needs that thinking sent back, which is done for the current turn only.
        #[serde(skip_serializing_if = "Option::is_none")]
        clear_thinking: Option<bool>,
    },
    Disabled,
}

/// The `response_format` request field, in its wire shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GlmOutputFormat {
    /// Some JSON object; the prompt has to ask for JSON and describe its shape.
    JsonObject,
}

/// A GLM model on the Z.ai API, e.g. `glm-4.6`. The model always decides whether to call
/// a tool: the API takes no other `tool_choice`.
#[derive(Clone, Debug)]
pub struct Glm {
    model: String,
    /// Name of the [`ModelProvider`](crate::experimental::model::ModelProvider) in the
    /// registry whose `zai` credential each request uses.
    provider: String,
    option: GlmOption,
}

impl Glm {
    /// Uses the `"default"` provider; see [`with_provider`](Self::with_provider).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            provider: "default".to_owned(),
            option: GlmOption::default(),
        }
    }

    /// Takes credentials from the provider registered under `provider`, looked up on
    /// every request.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    pub fn with_option(mut self, option: GlmOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn get_option(&self) -> &GlmOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut GlmOption {
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
        Ok(body)
    }

    /// The API key as a bearer token, and for a stream the event-stream `accept`.
    fn request_headers(&self, stream: bool) -> anyhow::Result<HeaderMap> {
        let Credential::ApiKey(key) = find_credential(&self.provider, "zai")? else {
            anyhow::bail!("The Z.ai API takes an API key");
        };
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, bearer(&key)?);
        if stream {
            headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
        }
        Ok(headers)
    }
}

impl LangModelInference for Glm {
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

/// Error `1113`, insufficient balance, waits on a recharge; other 429s are rate limits.
fn is_permanent_quota(body: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(body)
        .is_ok_and(|body| body["error"]["code"] == "1113")
}
