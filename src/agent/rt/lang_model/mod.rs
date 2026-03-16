mod api;

use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use url::Url;

use crate::{
    agent::{LangModelAPISchema, LangModelProvider},
    datatype::Value,
    message::{Delta as _, Marshaled, Message, MessageOutput, ToolDesc, Unmarshal as _},
};

/// Runtime
pub struct LangModelRuntime {
    model: String,
    provider: LangModelProvider,
}

struct LangModelRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDesc],
    pub url: &'a Url,
    pub api_key: &'a Option<String>,
}

impl LangModelRuntime {
    pub fn new(model: String, provider: LangModelProvider) -> Self {
        Self { model, provider }
    }

    pub async fn run(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> anyhow::Result<MessageOutput> {
        match &self.provider {
            LangModelProvider::API {
                schema,
                url,
                api_key,
            } => {
                // Create request
                let req = LangModelRequest {
                    model: &self.model,
                    messages,
                    tools,
                    url: &url,
                    api_key: &api_key,
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
                    LangModelAPISchema::OpenAI | LangModelAPISchema::Responses => {
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

                // Send request
                let client = reqwest::Client::new();
                let response = client
                    .post(url)
                    .headers(header_map)
                    .json(&body)
                    .send()
                    .await?;

                // Get request
                let status = response.status();
                let response_text = response.text().await?;
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
                    LangModelAPISchema::OpenAI | LangModelAPISchema::Responses => {
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
    use url::Url;

    use crate::{
        message::{FinishReason, Part, Role, ToolDescBuilder},
        to_value,
    };

    use super::*;

    /// Verifies that the POST request is sent and response is parsed.
    #[tokio::test]
    async fn test_run_lang_model_api_simple() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
        let lm = LangModelRuntime::new(
            "gpt-4".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::ChatCompletion,
                url,
                api_key: Some(api_key),
            },
        );
        let messages = vec![Message::new(Role::User).with_contents([Part::text("Hi")])];
        let tools: Vec<ToolDesc> = vec![];

        let resp = lm.run(&messages, &tools).await.unwrap();
        println!("{}", resp);
    }

    /// Verifies that the API returns tool calls when tools are provided.
    #[tokio::test]
    async fn test_run_lang_model_api_tool_call() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
        let lm = LangModelRuntime::new(
            "gpt-4".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::ChatCompletion,
                url,
                api_key: Some(api_key),
            },
        );
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

        let resp = lm.run(&messages, &tools).await.unwrap();

        println!("{}", resp);

        // The model should respond with a tool call
        assert_eq!(resp.finish_reason, FinishReason::ToolCall {});
        let tool_calls = resp
            .message
            .tool_calls
            .as_ref()
            .expect("Expected tool_calls in response");
        assert!(!tool_calls.is_empty(), "Expected at least one tool call");

        let (id, name, arguments) = tool_calls[0]
            .as_function()
            .expect("Expected function part in tool call");
        assert!(id.is_some(), "Expected tool call to have an id");
        assert_eq!(name, "get_temperature");
        assert!(
            arguments.pointer("/location").is_some(),
            "Expected 'location' argument in tool call"
        );
    }
}
