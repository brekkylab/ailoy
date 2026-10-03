use futures::{StreamExt as _, future::BoxFuture, stream::BoxStream};
use reqwest::header::{ACCEPT, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    experimental::model::LangModelInference,
    lang_model::{
        r#impl::{
            api::{AnthropicMarshal, AnthropicUnmarshal, anthropic::marshal_messages},
            framing::Framing,
            response_format::ResponseSchemaMarshal as _,
        },
        stream_response,
    },
    message::{Delta as _, Marshal as _, Message, MessageDeltaOutput, MessageOutput, Part, Role},
    tool::ToolDesc,
};

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";

/// Sent when [`ClaudeOption::max_tokens`] is `None`. Every request streams, so a large cap
/// does not risk an HTTP timeout.
const DEFAULT_MAX_TOKENS: u64 = 64000;

/// Credentials for the Claude API. Both bill the Console organization they belong to; a
/// claude.ai subscription (Pro/Max) login is not one of them, as Anthropic does not allow
/// third-party products to offer it without approval.
#[derive(Clone)]
pub enum ClaudeAuth {
    /// A Console API key (`sk-ant-api...`), sent as `x-api-key`.
    ApiKey(String),
    /// An OAuth access token, e.g. from `ant auth print-credentials --access-token` after
    /// `ant auth login`, sent as a bearer token with the OAuth beta header.
    OAuthToken(String),
}

impl ClaudeAuth {
    /// `ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`: the order the official SDKs check them.
    pub fn from_env() -> anyhow::Result<Self> {
        let var = |name| std::env::var(name).ok().filter(|v: &String| !v.is_empty());
        if let Some(key) = var("ANTHROPIC_API_KEY") {
            Ok(Self::ApiKey(key))
        } else if let Some(token) = var("ANTHROPIC_AUTH_TOKEN") {
            Ok(Self::OAuthToken(token))
        } else {
            anyhow::bail!("neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set")
        }
    }

    /// The headers every request carries: credentials and the API version.
    pub(crate) fn headers(&self) -> anyhow::Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let (name, credential) = match self {
            Self::ApiKey(key) => ("x-api-key", key.clone()),
            Self::OAuthToken(token) => {
                headers.insert(
                    "anthropic-beta",
                    HeaderValue::from_static("oauth-2025-04-20"),
                );
                ("authorization", format!("Bearer {token}"))
            }
        };
        let mut credential = HeaderValue::from_str(&credential)?;
        credential.set_sensitive(true);
        headers.insert(name, credential);
        Ok(headers)
    }
}

impl std::fmt::Debug for ClaudeAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ApiKey(_) => "ApiKey(..)",
            Self::OAuthToken(_) => "OAuthToken(..)",
        })
    }
}

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
    auth: ClaudeAuth,
    option: ClaudeOption,
}

impl Claude {
    pub fn new(model: impl Into<String>, auth: ClaudeAuth) -> Self {
        Self {
            model: model.into(),
            auth,
            option: ClaudeOption::default(),
        }
    }

    pub fn with_option(mut self, option: ClaudeOption) -> Self {
        self.option = option;
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn get_option(&self) -> &ClaudeOption {
        &self.option
    }

    pub fn get_option_mut(&mut self) -> &mut ClaudeOption {
        &mut self.option
    }

    /// The streaming Messages API body. System messages are joined into the top-level
    /// `system`; `effort` and `format` go under `output_config`.
    fn request_body(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> anyhow::Result<serde_json::Value> {
        let option = &self.option;
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": option.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "messages": serde_json::Value::from(marshal_messages(messages)),
            "stream": true,
        });
        let fields = body.as_object_mut().unwrap();

        let system = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| &m.contents)
            .filter_map(|p| match p {
                Part::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        if !system.is_empty() {
            fields.insert("system".into(), system.into());
        }
        if !tools.is_empty() {
            fields.insert("tools".into(), AnthropicMarshal.marshal(tools).into());
        }
        if let Some(thinking) = &option.thinking {
            fields.insert("thinking".into(), serde_json::to_value(thinking)?);
        }
        if let Some(tool_choice) = &option.tool_choice {
            fields.insert("tool_choice".into(), serde_json::to_value(tool_choice)?);
        }
        if !option.stop_sequences.is_empty() {
            fields.insert(
                "stop_sequences".into(),
                serde_json::to_value(&option.stop_sequences)?,
            );
        }

        let mut output_config = serde_json::Map::new();
        if let Some(effort) = option.effort {
            output_config.insert("effort".into(), serde_json::to_value(effort)?);
        }
        if let Some(ClaudeOutputFormat::JsonSchema { schema }) = &option.format {
            // Structured outputs need `additionalProperties: false` on every object.
            let schema = serde_json::Value::from(AnthropicMarshal.marshal_response_schema(schema));
            output_config.insert(
                "format".into(),
                serde_json::json!({"type": "json_schema", "schema": schema}),
            );
        }
        if !output_config.is_empty() {
            fields.insert("output_config".into(), output_config.into());
        }
        Ok(body)
    }

    fn request_headers(&self) -> anyhow::Result<HeaderMap> {
        let mut headers = self.auth.headers()?;
        headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
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
        // Built up front so the stream borrows nothing from the caller.
        let request = self
            .request_headers()
            .and_then(|headers| Ok((headers, self.request_body(messages, tools)?)));
        match request {
            Ok((headers, body)) => stream_response(
                MESSAGES_URL.to_owned(),
                headers,
                body,
                Box::new(AnthropicUnmarshal),
                Framing::Sse,
            ),
            Err(e) => Box::pin(futures::stream::once(async move { Err(e) })),
        }
    }

    /// [`infer_stream`](Self::infer_stream) accumulated into one message, so a long
    /// response does not hit an HTTP timeout.
    fn infer<'a>(
        &'a self,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let mut stream = self.infer_stream(messages, tools);
            let mut output = MessageDeltaOutput::new();
            while let Some(delta) = stream.next().await {
                output = output.accumulate(delta?)?;
            }
            output.finish()
        })
    }
}
