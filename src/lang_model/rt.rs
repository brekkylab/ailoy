use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use url::Url;

use super::{LangModelAPISchema, LangModelProviderElem};
use crate::{
    datatype::Value,
    lang_model::r#impl::api,
    message::{Delta as _, Marshaled, Message, MessageOutput, Unmarshal as _},
    tool::ToolDesc,
};

/// Runtime
pub struct LangModel {
    model: String,
    provider: LangModelProviderElem,
}

pub(crate) struct LangModelRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDesc],
    pub url: &'a Url,
    pub api_key: &'a Option<String>,
    pub max_tokens: Option<u64>,
}

impl LangModel {
    pub fn new(model: String, provider: LangModelProviderElem) -> Self {
        Self { model, provider }
    }

    pub fn model_id(&self) -> &str {
        &self.model
    }

    pub async fn run(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> anyhow::Result<MessageOutput> {
        match &self.provider {
            LangModelProviderElem::API {
                schema,
                url,
                api_key,
                max_tokens,
            } => {
                // Create request
                let req = LangModelRequest {
                    model: &self.model,
                    messages,
                    tools,
                    url,
                    api_key,
                    max_tokens: *max_tokens,
                };

                let req = match schema {
                    LangModelAPISchema::Anthropic => Value::from(Marshaled::<
                        LangModelRequest,
                        api::AnthropicMarshal,
                    >::new(&req)),
                    LangModelAPISchema::ChatCompletion => {
                        Value::from(
                            Marshaled::<LangModelRequest, api::ChatCompletionMarshal>::new(&req),
                        )
                    }
                    LangModelAPISchema::Gemini => {
                        Value::from(Marshaled::<LangModelRequest, api::GeminiMarshal>::new(&req))
                    }
                    LangModelAPISchema::OpenAI => {
                        Value::from(Marshaled::<LangModelRequest, api::OpenAIMarshal>::new(&req))
                    }
                };

                let req = req.as_object().ok_or(anyhow::anyhow!("Invalid Marshal"))?;

                // Build url
                let url = req
                    .get("url")
                    .ok_or(anyhow::anyhow!("No URL in marshaled request"))?
                    .as_str()
                    .ok_or(anyhow::anyhow!("Invalid URL"))?;

                // Build headers
                let headers = req
                    .get("header")
                    .ok_or(anyhow::anyhow!("No headers in marshaled request"))?;
                let mut header_map = HeaderMap::new();
                if let Some(header_obj) = headers.as_object() {
                    for (key, value) in header_obj.iter() {
                        if let Some(val_str) = value.as_str() {
                            header_map.insert(
                                HeaderName::from_bytes(key.as_bytes())?,
                                HeaderValue::from_str(val_str)?,
                            );
                        }
                    }
                }

                // Build body
                let body = req
                    .get("body")
                    .ok_or(anyhow::anyhow!("No body in marshaled request"))?;
                let body: serde_json::Value = body.clone().into();

                // Send request with retry on 429 (rate limit)
                let client = reqwest::Client::new();
                const MAX_RETRIES: u32 = 3;
                let (status, response_text) = {
                    let mut last_status = None;
                    let mut last_text = None;
                    for attempt in 0..=MAX_RETRIES {
                        let response = client
                            .post(url)
                            .headers(header_map.clone())
                            .json(&body)
                            .send()
                            .await?;
                        let s = response.status();
                        if s.as_u16() == 429 && attempt < MAX_RETRIES {
                            const MAX_WAIT_SECS: u64 = 10;
                            let wait_secs = response
                                .headers()
                                .get("retry-after")
                                .and_then(|v| v.to_str().ok())
                                .and_then(|v| v.parse::<u64>().ok())
                                .unwrap_or(1u64 << attempt)
                                .min(MAX_WAIT_SECS);
                            let text = response.text().await?;
                            log::warn!(
                                "Rate limited (429). Retrying after {}s (attempt {}/{}): {}",
                                wait_secs,
                                attempt + 1,
                                MAX_RETRIES,
                                text
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
                            last_status = Some(s);
                            last_text = Some(text);
                            continue;
                        }
                        let text = response.text().await?;
                        last_status = Some(s);
                        last_text = Some(text);
                        break;
                    }
                    (last_status.unwrap(), last_text.unwrap())
                };

                // Check response status
                if !status.is_success() {
                    anyhow::bail!("API request failed with status {status}: {response_text}");
                }

                let response_value: Value =
                    serde_json::from_str::<serde_json::Value>(&response_text)?.into();

                // Unmarshal
                let delta_output = match schema {
                    LangModelAPISchema::Anthropic => {
                        api::AnthropicUnmarshal::default().unmarshal(response_value)?
                    }
                    LangModelAPISchema::ChatCompletion => {
                        api::ChatCompletionUnmarshal::default().unmarshal(response_value)?
                    }
                    LangModelAPISchema::Gemini => {
                        api::GeminiUnmarshal::default().unmarshal(response_value)?
                    }
                    LangModelAPISchema::OpenAI => {
                        api::OpenAIUnmarshal::default().unmarshal(response_value)?
                    }
                };

                delta_output.finish()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lang_model::LangModelProvider,
        message::{FinishReason, Part, Role},
        to_value,
        tool::{ToolDesc, ToolDescBuilder},
    };

    fn openai_chat_completion(model: &str, api_key: String) -> LangModel {
        LangModel::new(
            model.to_string(),
            LangModelProvider::chat_completion(
                "https://api.openai.com/v1/chat/completions",
                Some(api_key),
            )
            .unwrap(),
        )
    }

    /// Verifies that the POST request is sent and response is parsed.
    #[tokio::test]
    async fn test_run_lang_model_api_simple() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = openai_chat_completion("gpt-5.4-mini", api_key);
        let messages = vec![Message::new(Role::User).with_contents([Part::text("Hi")])];
        let tools: Vec<ToolDesc> = vec![];

        let resp = model.run(&messages, &tools).await.unwrap();
        assert!(
            !resp.message.contents.is_empty(),
            "Expected at least one message content"
        );
        assert!(
            resp.message.contents.first().unwrap().is_text(),
            "Expected a text type content"
        );
    }

    /// Verifies that the API returns tool calls when tools are provided.
    #[tokio::test]
    async fn test_run_lang_model_api_tool_call() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = openai_chat_completion("gpt-5.4-mini", api_key);
        let messages = vec![
            Message::new(Role::User)
                .with_contents([Part::text("What is the current temperature in Seoul?")]),
        ];
        let tools = vec![
            ToolDescBuilder::new("get_temperature")
                .description("Get the current temperature for a given city")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }))
                .build(),
        ];

        let resp = model.run(&messages, &tools).await.unwrap();

        // The model should respond with a tool call
        assert_eq!(resp.finish_reason, FinishReason::ToolCall {});
        let tool_calls = resp
            .message
            .tool_calls
            .as_ref()
            .expect("Expected tool_calls in response");
        assert!(!tool_calls.is_empty(), "Expected at least one tool call");

        let (_, name, arguments) = tool_calls[0]
            .as_function()
            .expect("Expected function part in tool call");
        assert_eq!(name, "get_temperature");
        assert!(
            arguments.pointer("/location").is_some(),
            "Expected 'location' argument in tool call"
        );
    }

    /// Verifies that token usage is populated in the response from the real API.
    #[tokio::test]
    async fn test_run_returns_usage() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");
        let model = openai_chat_completion("gpt-5.4-mini", api_key);
        let messages = vec![Message::new(Role::User).with_contents([Part::text("Hi")])];

        let resp = model.run(&messages, &[]).await.unwrap();
        let usage = resp
            .usage
            .expect("usage must be present in real API response");
        assert!(usage.input_tokens > 0, "input_tokens must be > 0");
        assert!(usage.output_tokens > 0, "output_tokens must be > 0");
    }

    /// Verifies that the runtime retries on 429 and succeeds when the server recovers.
    /// Uses an axum mock server that returns 429 for the first two requests, then 200.
    #[tokio::test]
    async fn test_run_retries_on_429() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, body::Body, response::Response, routing::post};

        let call_count = Arc::new(Mutex::new(0u32));

        let count = call_count.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let count = count.clone();
                async move {
                    let mut n = count.lock().unwrap();
                    *n += 1;
                    let current = *n;
                    drop(n);

                    if current <= 2 {
                        // Return 429 with retry-after: 0 so the sleep is instant.
                        Response::builder()
                            .status(429)
                            .header("retry-after", "0")
                            .body(Body::from(r#"{"error":"rate limited"}"#))
                            .unwrap()
                    } else {
                        Response::builder()
                            .status(200)
                            .header("content-type", "application/json")
                            .body(Body::from(r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#))
                            .unwrap()
                    }
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let model = LangModel::new(
            "test-model".to_string(),
            LangModelProvider::chat_completion(&format!("http://{}/", addr), None).unwrap(),
        );
        let messages = vec![Message::new(Role::User).with_contents([Part::text("hi")])];

        let resp = model.run(&messages, &[]).await.unwrap();

        assert_eq!(
            *call_count.lock().unwrap(),
            3,
            "should have made 3 total attempts (2x 429 retried + 1 success)"
        );
        assert!(!resp.message.contents.is_empty());
    }
}
