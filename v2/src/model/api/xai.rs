use openai_sdk_rs::OpenAI;

use crate::{
    model::{
        api::openai_chat_completion::OpenAIChatCompletion,
        openai_chat_completion::{OpenAIGenerationConfig, OpenAIGenerationConfigBuilder},
    },
    value::FinishReason,
};

pub type XAIGenerationConfig = OpenAIGenerationConfig;
pub type XAIGenerationConfigBuilder = OpenAIGenerationConfigBuilder;

#[derive(Clone)]
pub struct XAILanguageModel {
    model_name: String,
    client: OpenAI,
    config: XAIGenerationConfig,
}

impl XAILanguageModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let client = OpenAI::builder()
            .base_url("https://api.x.ai/v1")
            .api_key(api_key.into())
            .build()
            .map_err(|e| format!("Failed to build XAI client: {}", e))
            .unwrap();

        Self {
            model_name: model_name.into(),
            client,
            config: OpenAIGenerationConfig::default(),
        }
    }
}

impl OpenAIChatCompletion for XAILanguageModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn client(&self) -> &OpenAI {
        &self.client
    }

    fn config(&self) -> &OpenAIGenerationConfig {
        &self.config
    }

    fn parse_finish_reason(finish_reason: &str) -> Result<FinishReason, String> {
        match finish_reason {
            "stop" => Ok(FinishReason::Stop),
            "length" => Ok(FinishReason::Length),
            "tool_calls" => Ok(FinishReason::ToolCalls),
            "content_filter" => Ok(FinishReason::ContentFilter),
            "end_turn" => Ok(FinishReason::Stop),
            _ => return Err(format!("Unknown finish reason: {}", finish_reason)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;
    use crate::{
        model::LanguageModel,
        utils::log,
        value::{Message, MessageAggregator, Part, Role, ToolDesc},
    };

    static XAI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("XAI_API_KEY")
            .expect("Environment variable 'XAI_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn xai_infer_with_thinking() {
        let mut xai = XAILanguageModel::new("grok-3-mini", *XAI_API_KEY);

        let msgs = vec![
            Message::with_role(Role::System).with_contents(vec![Part::Text(
                "You are a helpful mathematics assistant.".to_owned(),
            )]),
            Message::with_role(Role::User).with_contents(vec![Part::Text(
                "What is the sum of the first 50 prime numbers?".to_owned(),
            )]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = xai.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn xai_infer_tool_call() {
        use serde_json::json;

        let mut xai = XAILanguageModel::new("grok-3", *XAI_API_KEY);

        let tools = vec![
            ToolDesc::new(
                "temperature".into(),
                "Get current temperature".into(),
                json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name"
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit of temperature",
                            "enum": ["Celsius", "Fahrenheit"]
                        }
                    },
                    "required": ["location", "unit"]
                }),
                Some(json!({
                    "type": "number",
                    "description": "Null if the given city name is unavailable.",
                    "nullable": true,
                })),
            )
            .unwrap(),
        ];
        let mut msgs = vec![Message::with_role(Role::User).with_contents([Part::Text(
            "How much hot currently in Dubai? Answer in Celsius.".to_owned(),
        )])];
        let mut agg = MessageAggregator::new();
        let mut assistant_msg: Option<Message> = None;
        {
            let mut strm = xai.run(msgs.clone(), tools.clone());
            while let Some(delta_opt) = strm.next().await {
                let delta = delta_opt.unwrap();
                log::debug(format!("{:?}", delta).as_str());
                if let Some(msg) = agg.update(delta) {
                    log::info(format!("{:?}", msg).as_str());
                    assistant_msg = Some(msg);
                }
            }
        }
        // This should be tool call message
        let assistant_msg = assistant_msg.unwrap();
        msgs.push(assistant_msg.clone());

        // Append a fake tool call result message
        let mut tool_result_msg = Message::with_role(Role::Tool)
            .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
        if let Part::Function { id, .. } = assistant_msg.tool_calls[0].clone() {
            tool_result_msg = tool_result_msg.with_tool_call_id(id);
        }
        msgs.push(tool_result_msg);

        let mut strm = xai.run(msgs, tools);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                // Final message shuold say something like "Dubai is 38.5°C"
                log::info(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn xai_infer_with_image() {
        use base64::Engine;

        let client = reqwest::Client::new();
        let test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jensen_Huang_%28cropped%29.jpg/250px-Jensen_Huang_%28cropped%29.jpg";
        let response = client.get(test_image_url).header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36").send().await.unwrap();
        let image_bytes = response.bytes().await.unwrap();
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let mut xai = XAILanguageModel::new("grok-4", *XAI_API_KEY);

        let msgs = vec![
            Message::with_role(Role::User).with_contents(vec![Part::ImageData {
                data: image_base64,
                mime_type: "image/jpeg".into(),
            }]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("What is shown in this image?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = xai.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta));
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg));
            }
        }
    }
}
