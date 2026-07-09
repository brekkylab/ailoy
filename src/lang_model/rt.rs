use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use super::LangModelProviderElem;
use crate::{
    datatype::Value,
    lang_model::{LangModelOptions, get_lm_providers},
    message::{Delta as _, Marshal as _, Message, MessageOutput, Unmarshal as _},
    tool::ToolDesc,
};

/// Runtime
pub struct LangModel {
    model: String,
    provider: LangModelProviderElem,
}

pub(super) struct LangModelRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDesc],
    pub provider: &'a LangModelProviderElem,
    pub options: &'a LangModelOptions,
}

impl LangModel {
    /// Resolve `model` against the `"default"` entry of
    /// [`get_lm_providers`](crate::lang_model::get_lm_providers).  Convenience
    /// over [`try_from_provider`](Self::try_from_provider).
    ///
    /// Returns an error if the `"default"` provider is missing or has no
    /// entry matching `model`.
    pub fn try_new(model: String) -> anyhow::Result<Self> {
        Self::try_from_provider(model, "default")
    }

    /// Resolve `model` against the [`LangModelProvider`](super::LangModelProvider)
    /// registered under `provider` in
    /// [`get_lm_providers`](crate::lang_model::get_lm_providers).
    ///
    /// `model` is the spec-side name (e.g. `"openai/gpt-4o"`) used to look up
    /// the registered pattern; the stored API-side id has any `provider/`
    /// prefix stripped (e.g. `"gpt-4o"`) so it matches what the upstream
    /// endpoint expects.
    ///
    /// Returns an error if `provider` is not registered, or if no entry
    /// inside it matches `model` (with the usual exact-then-glob lookup).
    pub fn try_from_provider(model: String, provider: impl AsRef<str>) -> anyhow::Result<Self> {
        let provider_name = provider.as_ref();
        let registry = get_lm_providers();
        let lmp = registry.get(provider_name).ok_or_else(|| {
            anyhow::anyhow!("lang_model_provider '{}' not registered", provider_name)
        })?;
        let provider_elem = lmp
            .get(&model)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no entry for model '{}' in lang_model_provider '{}'",
                    model,
                    provider_name
                )
            })?
            .clone();
        let api_model_id = lmp.resolve_model_id(&model)?;
        Ok(Self {
            model: api_model_id,
            provider: provider_elem,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model
    }

    pub async fn run(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<MessageOutput> {
        let provider = &self.provider;
        match provider {
            LangModelProviderElem::API { schema, .. } => {
                // Create request
                let req = LangModelRequest {
                    model: &self.model,
                    messages,
                    tools,
                    provider,
                    options,
                };

                // Marshal req
                let req = schema.marshal(&req);
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
                            // Permanent quota/credit exhaustion never recovers; don't retry.
                            if schema.is_permanent_quota_error(&text) {
                                log::warn!("Quota exhausted (429), not retrying: {text}");
                                last_status = Some(s);
                                last_text = Some(text);
                                break;
                            }
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

                // Decode via the schema's unmarshal impl into a single delta.
                let delta_output = {
                    let response_value: Value =
                        serde_json::from_str::<serde_json::Value>(&response_text)?.into();
                    schema.unmarshal(response_value)?
                };

                // In non-streaming API, a single delta is the complete output, so finalize now.
                delta_output.finish()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        lang_model::{LangModelProvider, ResponseFormat, get_lm_providers_mut},
        message::{FinishReason, Part, Role},
        to_value,
        tool::{ToolDesc, ToolDescBuilder},
    };

    fn stored_schema(fmt: ResponseFormat) -> Value {
        match fmt {
            ResponseFormat::JsonSchema(s) => s,
        }
    }

    #[test]
    fn test_response_format_invalid_schema_rejected() {
        assert!(ResponseFormat::json_schema(to_value!({"type": 123})).is_err());
    }

    #[test]
    fn test_response_format_non_object_rejected() {
        assert!(ResponseFormat::json_schema(to_value!("not an object")).is_err());
    }

    #[test]
    fn test_response_format_stores_raw_schema() {
        let schema = to_value!({"type": "object", "properties": {"x": {"type": "string"}}});
        let fmt = ResponseFormat::json_schema(schema.clone()).unwrap();
        let stored = stored_schema(fmt);
        assert_eq!(stored, schema, "stored schema must equal input");
        assert!(
            stored.pointer("/additionalProperties").is_none(),
            "ResponseFormat must not mutate the schema at construction time"
        );
    }

    #[test]
    fn test_response_format_serde_round_trip() {
        let schema = to_value!({"type": "object", "properties": {"name": {"type": "string"}}});
        let original = ResponseFormat::json_schema(schema).unwrap();
        let json = serde_json::to_string(&original).expect("should serialize");
        let restored: ResponseFormat = serde_json::from_str(&json).expect("should deserialize");
        assert_eq!(original, restored);
    }

    /// Register a one-off [`LangModelProvider`] under a unique key in the
    /// global registry and build a [`LangModel`] from it via
    /// [`LangModel::try_from_provider`].  Test fixtures only.
    fn build_test_model(
        provider_name: &str,
        model: &str,
        elem: LangModelProviderElem,
    ) -> LangModel {
        let mut lmp = LangModelProvider::new();
        lmp.insert(model.into(), elem);
        get_lm_providers_mut().insert(provider_name.into(), lmp);
        LangModel::try_from_provider(model.to_string(), provider_name).unwrap()
    }

    fn openai_chat_completion(model: &str, api_key: String) -> LangModel {
        let elem = LangModelProvider::chat_completion(
            "https://api.openai.com/v1/chat/completions",
            Some(api_key),
        )
        .unwrap();
        build_test_model("test_openai_chat_completion", model, elem)
    }

    /// Verifies that the POST request is sent and response is parsed.
    #[tokio::test]
    async fn test_run_lang_model_api_simple() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = openai_chat_completion("gpt-5.4-mini", api_key);
        let messages = vec![Message::new(Role::User).with_contents([Part::text("Hi")])];
        let tools: Vec<ToolDesc> = vec![];

        let resp = model
            .run(&messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();
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

        let resp = model
            .run(&messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();

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

        let resp = model
            .run(&messages, &[], &LangModelOptions::default())
            .await
            .unwrap();
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

        let elem = LangModelProvider::chat_completion(&format!("http://{}/", addr), None).unwrap();
        let model = build_test_model("test_run_lang_model_mock", "test-model", elem);
        let messages = vec![Message::new(Role::User).with_contents([Part::text("hi")])];

        let resp = model
            .run(&messages, &[], &LangModelOptions::default())
            .await
            .unwrap();

        assert_eq!(
            *call_count.lock().unwrap(),
            3,
            "should have made 3 total attempts (2x 429 retried + 1 success)"
        );
        assert!(!resp.message.contents.is_empty());
    }
}
