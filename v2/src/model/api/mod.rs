mod openai;

use async_stream::try_stream;
use futures::{StreamExt as _, stream::BoxStream};
use reqwest::RequestBuilder;
use std::{fmt::Debug, sync::Arc};

use crate::{
    model::LanguageModel,
    value::{Message, MessageDelta, MessageOutput, Part, ToolDescription},
};

#[derive(Clone)]
pub struct APILanguageModel {
    model_name: String,
    build_request: Arc<dyn Fn() -> RequestBuilder + Send + Sync + 'static>,
}

impl APILanguageModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> APILanguageModel {
        let model_name = model_name.into();
        let api_key: String = api_key.into();
        let build_request: Arc<dyn Fn() -> reqwest::RequestBuilder + Send + Sync + 'static> =
            if model_name.starts_with("gpt") || model_name.starts_with("o") {
                Arc::new(move || {
                    reqwest::Client::new()
                        .request(
                            reqwest::Method::POST,
                            "https://api.openai.com/v1/chat/completions",
                        )
                        .bearer_auth(api_key.clone())
                        .header("Content-Type", "application/json")
                })
            } else if model_name.starts_with("claude") {
                Arc::new(move || {
                    reqwest::Client::new()
                        .request(
                            reqwest::Method::POST,
                            "https://api.anthropic.com/v1/messages",
                        )
                        .header("x-api-key", api_key.clone())
                        .header("Content-Type", "application/json")
                })
            } else if model_name.starts_with("gemini") {
                todo!()
            } else if model_name.starts_with("grok") {
                todo!()
            } else {
                panic!()
            };

        APILanguageModel {
            model_name,
            build_request,
        }
    }
}

impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDescription>,
    ) -> BoxStream<'static, Result<MessageOutput, String>> {
        let model_name = self.model_name.clone();
        let mut body = serde_json::json!({"model": model_name, "messages": msgs});
        if !tools.is_empty() {
            body["tool_choice"] = serde_json::json!("auto");
            body["tools"] = serde_json::to_value(tools).unwrap();
        }

        let is_openai_like = model_name.starts_with("gpt") || model_name.starts_with("o");
        if is_openai_like {
            body["stream"] = serde_json::json!(true);
        }

        let req = (self.build_request)()
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .body(body.to_string())
            .send();

        todo!()
    }
}

impl Debug for APILanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("APILanguageModel")
            .field("model_name", &self.model_name)
            .finish()
    }
}
