use std::sync::Arc;

use ailoy_macros::maybe_send_sync;
use anyhow::anyhow;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use super::api::{APISpecification, RequestConfig, ServerEvent, drain_next_event};
use crate::{
    utils::{BoxFuture, BoxStream},
    value::{Message, MessageDeltaOutput, MessageOutput, Role, ToolDesc},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum ThinkEffort {
    #[default]
    Disable,
    Enable,
    Low,
    Medium,
    High,
}

/// Configuration parameters that control the behavior of language model inference.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LangModelInferConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think_effort: Option<ThinkEffort>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
}

#[maybe_send_sync]
pub trait LangModelInference {
    fn infer_delta<'a>(
        &'a self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: LangModelInferConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>>;

    fn infer<'a>(
        &'a self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: LangModelInferConfig,
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>>;
}

#[maybe_send_sync]
type MakeRequestFunc =
    dyn Fn(Vec<Message>, Vec<ToolDesc>, RequestConfig) -> reqwest::RequestBuilder;

#[maybe_send_sync]
type HandleEventFunc = dyn Fn(ServerEvent) -> MessageDeltaOutput;

#[maybe_send_sync]
type HandleResponseFunc = dyn Fn(serde_json::Value) -> anyhow::Result<MessageOutput>;

/// Builder for constructing a [`LangModel`] with a specific API provider.
///
/// Obtain via provider free functions: [`anthropic`], [`openai`], [`gemini`], [`grok`],
/// [`chat_completion`].
pub struct LangModelBuilder {
    spec: APISpecification,
    model: String,
    api_key: Option<String>,
    base_url: Option<String>,
}

impl LangModelBuilder {
    fn new(spec: APISpecification, model: impl Into<String>) -> Self {
        Self {
            spec,
            model: model.into(),
            api_key: None,
            base_url: None,
        }
    }

    /// Override the API key. If not set, the provider's default environment variable is used.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Override the base URL (useful for OpenAI-compatible endpoints).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the [`LangModel`].
    ///
    /// If no `api_key` was provided, the provider's default environment variable is read
    /// (e.g. `ANTHROPIC_API_KEY` for Anthropic). Returns an error if neither is available.
    pub fn build(self) -> anyhow::Result<LangModel> {
        let api_key = match self.api_key {
            Some(k) => k,
            None => std::env::var(self.spec.env_var()).map_err(|_| {
                anyhow::anyhow!(
                    "API key not provided and environment variable '{}' is not set",
                    self.spec.env_var()
                )
            })?,
        };
        let url = self
            .base_url
            .unwrap_or_else(|| self.spec.default_url().to_owned());
        Ok(LangModel::with_url(self.spec, self.model, api_key, url))
    }
}

#[derive(Clone)]
pub struct LangModel {
    name: String,
    make_request: Arc<MakeRequestFunc>,
    handle_event: Arc<HandleEventFunc>,
    handle_response: Arc<HandleResponseFunc>,
}

impl LangModel {
    /// Create a [`LangModelBuilder`] for Anthropic Claude.
    /// Reads `ANTHROPIC_API_KEY` from the environment if `.api_key()` is not called.
    pub fn anthropic(model: impl Into<String>) -> LangModelBuilder {
        LangModelBuilder::new(APISpecification::Claude, model)
    }

    /// Create a [`LangModelBuilder`] for the OpenAI Responses API.
    /// Reads `OPENAI_API_KEY` from the environment if `.api_key()` is not called.
    pub fn openai(model: impl Into<String>) -> LangModelBuilder {
        LangModelBuilder::new(APISpecification::OpenAI, model)
    }

    /// Create a [`LangModelBuilder`] for Google Gemini.
    /// Reads `GEMINI_API_KEY` from the environment if `.api_key()` is not called.
    pub fn gemini(model: impl Into<String>) -> LangModelBuilder {
        LangModelBuilder::new(APISpecification::Gemini, model)
    }

    /// Create a [`LangModelBuilder`] for xAI Grok.
    /// Reads `XAI_API_KEY` from the environment if `.api_key()` is not called.
    pub fn grok(model: impl Into<String>) -> LangModelBuilder {
        LangModelBuilder::new(APISpecification::Grok, model)
    }

    /// Create a [`LangModelBuilder`] for any OpenAI Chat Completions–compatible endpoint.
    /// Reads `OPENAI_API_KEY` from the environment if `.api_key()` is not called.
    /// Use `.base_url()` to point at a custom endpoint.
    pub fn chat_completion(model: impl Into<String>) -> LangModelBuilder {
        LangModelBuilder::new(APISpecification::ChatCompletion, model)
    }

    fn with_url(
        spec: APISpecification,
        model: impl Into<String>,
        api_key: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let api_key = api_key.into();
        let url = url.into();

        match spec {
            APISpecification::OpenAI | APISpecification::Responses => LangModel {
                name: model,
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::api::openai::make_request(&url, &api_key, msgs, tools, req)
                    },
                ),
                handle_event: Arc::new(super::api::openai::handle_event),
                handle_response: Arc::new(super::api::openai::handle_response),
            },
            APISpecification::Gemini => LangModel {
                name: model,
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::api::gemini::make_request(&url, &api_key, msgs, tools, req)
                    },
                ),
                handle_event: Arc::new(super::api::gemini::handle_event),
                handle_response: Arc::new(super::api::gemini::handle_response),
            },
            APISpecification::Claude => LangModel {
                name: model,
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::api::anthropic::make_request(&url, &api_key, msgs, tools, req)
                    },
                ),
                handle_event: Arc::new(super::api::anthropic::handle_event),
                handle_response: Arc::new(super::api::anthropic::handle_response),
            },
            APISpecification::ChatCompletion | APISpecification::Grok => LangModel {
                name: model,
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::api::chat_completion::make_request(&url, &api_key, msgs, tools, req)
                    },
                ),
                handle_event: Arc::new(super::api::chat_completion::handle_event),
                handle_response: Arc::new(super::api::chat_completion::handle_response),
            },
        }
    }
}

impl LangModel {
    fn extract_system_message(msgs: &mut Vec<Message>) -> anyhow::Result<Option<String>> {
        if let Some(msg) = msgs.first()
            && msg.role == Role::System
        {
            let system_msg = msgs.remove(0);
            let text = system_msg
                .contents
                .first()
                .ok_or_else(|| anyhow::anyhow!("System message has no content"))?
                .as_text()
                .ok_or_else(|| anyhow::anyhow!("System message first content is not text"))?
                .to_owned();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }
}

impl LangModelInference for LangModel {
    fn infer_delta<'a>(
        &'a self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: LangModelInferConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        let strm = async_stream::try_stream! {
            let mut msgs = msgs;
            let mut buf: Vec<u8> = Vec::with_capacity(8192);

            let system_message = Self::extract_system_message(&mut msgs)?;
            let req = RequestConfig {
                model: Some(self.name.clone()),
                system_message,
                stream: true,
                think_effort: config.think_effort.unwrap_or_default(),
                temperature: config.temperature,
                top_p: config.top_p,
                max_tokens: config.max_tokens,
            };

            let resp = (self.make_request)(msgs, tools, req).send().await?;

            if resp.status().is_success() {
                let mut strm = resp.bytes_stream();
                'outer: while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res?;
                    buf.extend_from_slice(&chunk);
                    while let Some(evt) = drain_next_event(&mut buf) {
                        let message_output = (self.handle_event)(evt);
                        match message_output.finish_reason {
                            None => { yield message_output; }
                            Some(_) => { yield message_output; break 'outer; }
                        }
                    }
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(anyhow!("Request failed: {} - {}", status, text))?
            }
        };
        Box::pin(strm)
    }

    fn infer<'a>(
        &'a self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: LangModelInferConfig,
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let mut msgs = msgs;

            let system_message = Self::extract_system_message(&mut msgs)?;
            let req = RequestConfig {
                model: Some(self.name.clone()),
                system_message,
                stream: false,
                think_effort: config.think_effort.unwrap_or_default(),
                temperature: config.temperature,
                top_p: config.top_p,
                max_tokens: config.max_tokens,
            };

            let resp = (self.make_request)(msgs, tools, req).send().await?;
            if resp.status().is_success() {
                let json: serde_json::Value = resp.json().await?;
                (self.handle_response)(json)
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(anyhow!("Request failed: {} - {}", status, text))
            }
        })
    }
}
