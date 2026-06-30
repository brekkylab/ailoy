use futures::{StreamExt as _, stream::BoxStream};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use url::Url;

use super::{LangModelAPISchema, LangModelProviderElem};
use crate::{
    datatype::Value,
    lang_model::{LangModelOptions, r#impl::api},
    message::{Delta as _, Marshaled, Message, MessageDeltaOutput, MessageOutput},
    tool::ToolDesc,
};

/// Constrains the model's response to a specific JSON format.
///
/// Constructed via [`ResponseFormat::json_schema`], which validates the schema
/// against JSON Schema Draft 7 before storing it, then normalises it to satisfy
/// provider-specific requirements.  The stored schema is provider-agnostic;
/// each marshal converts it to the wire format expected by its API.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "schema", rename_all = "snake_case")]
pub enum ResponseFormat {
    JsonSchema(Value),
}

impl ResponseFormat {
    /// Validate `schema` against JSON Schema Draft 7.  Returns `Err` if the
    /// schema is structurally invalid (e.g. `"type": 123`).  The stored schema
    /// is the user's original; provider-specific transformations happen in each
    /// marshal via [`ResponseSchemaMarshal::marshal_response_schema`].
    pub fn json_schema(schema: Value) -> anyhow::Result<Self> {
        let serde_schema: serde_json::Value = schema.clone().into();
        jsonschema::validator_for(&serde_schema)
            .map_err(|e| anyhow::anyhow!("Invalid JSON schema: {}", e))?;
        Ok(Self::JsonSchema(schema))
    }
}

impl schemars::JsonSchema for ResponseFormat {
    fn schema_name() -> String {
        "ResponseFormat".into()
    }

    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        use schemars::schema::{InstanceType, ObjectValidation, SchemaObject, SingleOrVec};
        SchemaObject {
            instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Object))),
            object: Some(Box::new(ObjectValidation {
                required: ["type".to_owned()].into(),
                ..Default::default()
            })),
            ..Default::default()
        }
        .into()
    }
}

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
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub response_format: Option<&'a ResponseFormat>,
    /// When true, the marshal requests a streaming (SSE) response.
    pub stream: bool,
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
        options: &LangModelOptions,
    ) -> anyhow::Result<MessageOutput> {
        match &self.provider {
            LangModelProviderElem::API {
                schema,
                url,
                api_key,
            } => {
                // Create request (non-streaming)
                let req = LangModelRequest {
                    model: &self.model,
                    messages,
                    tools,
                    url,
                    api_key,
                    max_tokens: options.max_tokens,
                    temperature: options.temperature,
                    top_p: options.top_p,
                    top_k: options.top_k,
                    response_format: options.response_format.as_ref(),
                    stream: false,
                };
                let (url, header_map, body) = marshal_request(schema, &req)?;

                // Send with retry on 429, then read the whole response.
                let provider = api::provider_api(schema);
                let client = reqwest::Client::new();
                let response =
                    send_with_retry(&client, &url, header_map, &body, provider.as_ref()).await?;
                let response_text = response.text().await?;

                let response_value: Value =
                    serde_json::from_str::<serde_json::Value>(&response_text)?.into();

                // Unmarshal
                let delta_output = provider.unmarshal_response(response_value)?;

                delta_output.finish()
            }
        }
    }

    /// Streaming counterpart to [`run`](Self::run): requests an SSE response and
    /// yields one [`MessageDeltaOutput`] per incremental update. Callers
    /// accumulate the deltas (via [`Delta::accumulate`]) to build the final
    /// message. The stream ends when the response body ends (after the terminal
    /// SSE event), not at the first `finish_reason` — some providers send usage
    /// in a final chunk after it.
    pub fn run_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        // Own everything the stream touches so the returned future is 'static.
        let model = self.model.clone();
        let provider_elem = self.provider.clone();
        let messages = messages.to_vec();
        let tools = tools.to_vec();
        let options = options.clone();

        Box::pin(async_stream::try_stream! {
            let LangModelProviderElem::API { schema, url, api_key } = &provider_elem;

            // Create request (streaming)
            let req = LangModelRequest {
                model: &model,
                messages: &messages,
                tools: &tools,
                url,
                api_key,
                max_tokens: options.max_tokens,
                temperature: options.temperature,
                top_p: options.top_p,
                top_k: options.top_k,
                response_format: options.response_format.as_ref(),
                stream: true,
            };
            let (url, header_map, body) = marshal_request(schema, &req)?;

            let provider = api::provider_api(schema);
            let client = reqwest::Client::new();
            let response =
                send_with_retry(&client, &url, header_map, &body, provider.as_ref()).await?;

            // Read the SSE body chunk by chunk, framing complete events out of a
            // buffer (network chunks don't align with event boundaries). We drain
            // until the body ends rather than stopping at the first `finish_reason`
            // — some providers (ChatCompletion with `stream_options.include_usage`)
            // send the usage in a final chunk *after* the finish_reason one. The
            // server ends the body right after the terminal event, so this exits
            // promptly without waiting on the connection.
            let mut byte_stream = response.bytes_stream();
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = byte_stream.next().await {
                buf.extend_from_slice(&chunk?);
                while let Some(data) = drain_next_event(&mut buf) {
                    if data.is_empty() {
                        continue; // keep-alive / comment line
                    }
                    if let Some(output) = provider.unmarshal_event(&data)? {
                        yield output;
                    }
                }
            }

            // A well-behaved server terminates the final event with a blank
            // line, but some close the connection right after the last event
            // (no trailing blank line). `drain_next_event` only frames on that
            // blank line, so flush whatever remains as one last event — this is
            // the only copy of an EOF-terminated terminal event (e.g. Gemini's
            // finish_reason + usage chunk), which would otherwise be dropped.
            let data = extract_event_data(&buf);
            // Not a let-chain: the `try_stream!` macro rejects them (Rust 2024).
            #[allow(clippy::collapsible_if)]
            if !data.is_empty() {
                if let Some(output) = provider.unmarshal_event(&data)? {
                    yield output;
                }
            }
        })
    }
}

/// Marshals a [`LangModelRequest`] for `schema` into the wire `(url, headers,
/// body)` tuple shared by the blocking and streaming paths.
fn marshal_request(
    schema: &LangModelAPISchema,
    req: &LangModelRequest,
) -> anyhow::Result<(String, HeaderMap, serde_json::Value)> {
    let marshaled = match schema {
        LangModelAPISchema::Anthropic => {
            Value::from(Marshaled::<LangModelRequest, api::AnthropicMarshal>::new(req))
        }
        LangModelAPISchema::ChatCompletion => {
            Value::from(Marshaled::<LangModelRequest, api::ChatCompletionMarshal>::new(req))
        }
        LangModelAPISchema::Gemini => {
            Value::from(Marshaled::<LangModelRequest, api::GeminiMarshal>::new(req))
        }
        LangModelAPISchema::OpenAI => {
            Value::from(Marshaled::<LangModelRequest, api::OpenAIMarshal>::new(req))
        }
    };
    let obj = marshaled
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid Marshal"))?;

    let url = obj
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("No URL in marshaled request"))?
        .to_owned();

    let mut header_map = HeaderMap::new();
    if let Some(header_obj) = obj.get("header").and_then(|v| v.as_object()) {
        for (key, value) in header_obj.iter() {
            if let Some(val_str) = value.as_str() {
                header_map.insert(
                    HeaderName::from_bytes(key.as_bytes())?,
                    HeaderValue::from_str(val_str)?,
                );
            }
        }
    }

    let body = obj
        .get("body")
        .ok_or_else(|| anyhow::anyhow!("No body in marshaled request"))?;
    let body: serde_json::Value = body.clone().into();

    Ok((url, header_map, body))
}

/// POSTs the request, retrying transient 429s with backoff, and returns the
/// successful (2xx) response **unconsumed** so the caller decides whether to
/// read it whole (`run`) or stream it (`run_stream`). Bails on a non-2xx
/// response or exhausted retries.
async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
    provider: &(dyn api::ProviderApi + Send + Sync),
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    for attempt in 0..=MAX_RETRIES {
        let response = client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if status.as_u16() == 429 && attempt < MAX_RETRIES {
            let wait_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1u64 << attempt)
                .min(MAX_WAIT_SECS);
            let text = response.text().await?;
            // Permanent quota/credit exhaustion never recovers; don't retry.
            if provider.is_permanent_quota_error(&text) {
                log::warn!("Quota exhausted (429), not retrying: {text}");
                anyhow::bail!("API request failed with status {status}: {text}");
            }
            log::warn!(
                "Rate limited (429). Retrying after {}s (attempt {}/{}): {}",
                wait_secs,
                attempt + 1,
                MAX_RETRIES,
                text
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("API request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Drains the next complete SSE event from `buf`, returning its concatenated
/// `data:` payload. Returns `None` if no event (terminated by a blank line) is
/// fully buffered yet; the partial bytes stay in `buf` for the next chunk.
fn drain_next_event(buf: &mut Vec<u8>) -> Option<String> {
    // Events are separated by a blank line: "\n\n" (LF) or "\r\n\r\n" (CRLF).
    let (sep_pos, sep_len) = buf
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| (p, 2))
        .or_else(|| buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| (p, 4)))?;

    let raw: Vec<u8> = buf.drain(..sep_pos + sep_len).collect();
    Some(extract_event_data(&raw))
}

/// Extracts the concatenated `data:` payload from one raw SSE event's bytes.
/// SSE permits multiple `data:` lines per event; they are joined with newlines.
/// Non-`data:` lines (`event:`, `id:`, comments) are dropped — every provider
/// here carries its event type inside the `data:` JSON.
fn extract_event_data(raw: &[u8]) -> String {
    String::from_utf8_lossy(raw)
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|rest| rest.trim())
        .collect::<Vec<_>>()
        .join("\n")
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

    #[test]
    fn test_drain_next_event_frames_on_blank_line() {
        // Two complete LF-separated events plus a partial third still buffered.
        let mut buf = b"data: a\n\ndata: b\n\ndata: c".to_vec();
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("a"));
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("b"));
        // The unterminated tail is not framed; it stays for the next chunk.
        assert_eq!(drain_next_event(&mut buf), None);
        assert_eq!(buf, b"data: c");
    }

    #[test]
    fn test_drain_next_event_handles_crlf_and_multi_data() {
        // CRLF separators, and an event with two `data:` lines (joined by \n).
        let mut buf = b"data: x\r\ndata: y\r\n\r\nrest".to_vec();
        assert_eq!(drain_next_event(&mut buf).as_deref(), Some("x\ny"));
        assert_eq!(drain_next_event(&mut buf), None);
        assert_eq!(buf, b"rest");
    }

    #[test]
    fn test_extract_event_data_recovers_eof_terminated_event() {
        // The run_stream EOF flush relies on this: a final event left in the
        // buffer without a trailing blank line must still yield its payload.
        assert_eq!(extract_event_data(b"data: {\"k\":1}"), "{\"k\":1}");
        // Non-`data:` lines (comments / event:) are dropped.
        assert_eq!(extract_event_data(b": keep-alive"), "");
        assert_eq!(extract_event_data(b"event: done\ndata: tail"), "tail");
    }

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

        let model = LangModel::new(
            "test-model".to_string(),
            LangModelProvider::chat_completion(&format!("http://{}/", addr), None).unwrap(),
        );
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

    /// Verifies a permanent quota 429 (`insufficient_quota`) is not retried.
    #[tokio::test]
    async fn test_run_does_not_retry_on_permanent_429() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, body::Body, response::Response, routing::post};

        let call_count = Arc::new(Mutex::new(0u32));
        let count = call_count.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let count = count.clone();
                async move {
                    *count.lock().unwrap() += 1;
                    Response::builder()
                        .status(429)
                        .body(Body::from(
                            r#"{"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"You exceeded your current quota"}}"#,
                        ))
                        .unwrap()
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
            LangModelProviderElem::API {
                schema: LangModelAPISchema::OpenAI,
                url: format!("http://{}/", addr).parse().unwrap(),
                api_key: None,
            },
        );
        let messages = vec![Message::new(Role::User).with_contents([Part::text("hi")])];

        let result = model
            .run(&messages, &[], &LangModelOptions::default())
            .await;

        assert!(result.is_err(), "permanent quota 429 must fail");
        assert_eq!(
            *call_count.lock().unwrap(),
            1,
            "permanent quota 429 must not be retried (1 attempt only)"
        );
    }
}
