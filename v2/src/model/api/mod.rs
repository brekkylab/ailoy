use async_stream::try_stream;
use futures::stream::BoxStream;
use reqwest::RequestBuilder;
use std::{fmt::Debug, sync::Arc};

use crate::{
    model::LanguageModel,
    value::{Message, MessageDelta, Part, ToolDescription},
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
    ) -> BoxStream<'static, Result<MessageDelta, String>> {
        let model_name = self.model_name.clone();
        let body = serde_json::json!({"model": model_name, "messages": msgs}).to_string();
        let request = (self.build_request)().body(body);
        Box::pin(try_stream! {
            let response = request.send().await;
            let resp_str = response.map_err(|e| format!("Request failed: {}", e.to_string()))?.text().await.map_err(|e| e.to_string())?;
            let resp_json: serde_json::Value = serde_json::from_str(&resp_str).map_err(|e| format!("JSON conversion failed: {}", e.to_string()))?;
            let text = resp_json.pointer("/choices/0/message/content").and_then(|x| x.as_str()).unwrap_or("");
            println!("{}", text);
            yield MessageDelta::new_assistant_content(Part::new_text(text));
        })
    }
}

impl Debug for APILanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("APILanguageModel")
            .field("model_name", &self.model_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    const OPENAI_API_KEY: &str = "";

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Role};

        let model = Arc::new(APILanguageModel::new("gpt-5", OPENAI_API_KEY));
        let msgs = vec![
            Message::with_content(Role::System, Part::new_text("You are an assistant.")),
            Message::with_content(Role::User, Part::new_text("Hi what's your name?")),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
        println!("{:?}", agg.finalize());
    }
}
