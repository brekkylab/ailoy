use openai_sdk_rs::OpenAI;

use crate::model::api::openai_chat_completion::OpenAIChatCompletion;
pub use crate::model::api::openai_chat_completion::{
    OpenAIGenerationConfig, OpenAIGenerationConfigBuilder,
};

#[derive(Clone)]
pub struct OpenAILanguageModel {
    model_name: String,
    client: OpenAI,
    config: OpenAIGenerationConfig,
}

impl OpenAILanguageModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let client = OpenAI::builder()
            .api_key(api_key.into())
            .build()
            .map_err(|e| format!("Failed to build OpenAI client: {}", e))
            .unwrap();

        Self {
            model_name: model_name.into(),
            client,
            config: OpenAIGenerationConfig::default(),
        }
    }

    pub fn with_config(mut self, config: OpenAIGenerationConfig) -> Self {
        self.config = config;
        self
    }
}

impl OpenAIChatCompletion for OpenAILanguageModel {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn client(&self) -> &OpenAI {
        &self.client
    }

    fn config(&self) -> &OpenAIGenerationConfig {
        &self.config
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

    static OPENAI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("OPENAI_API_KEY")
            .expect("Environment variable 'OPENAI_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn openai_infer_with_thinking() {
        let mut model = OpenAILanguageModel::new("o3-mini", *OPENAI_API_KEY);

        let msgs = vec![
            Message::with_role(Role::System).with_contents(vec![Part::Text(
                "You are a helpful mathematics assistant.".to_owned(),
            )]),
            Message::with_role(Role::User).with_contents(vec![Part::Text(
                "What is the sum of the first 50 prime numbers?".to_owned(),
            )]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn openai_infer_tool_call() {
        use super::*;
        use crate::value::{MessageAggregator, ToolDescArg};

        let mut model = OpenAILanguageModel::new("gpt-4.1", *OPENAI_API_KEY);
        let tools = vec![ToolDesc::new(
            "temperature",
            "Get current temperature",
            ToolDescArg::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescArg::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescArg::new_string()
                            .with_enum(["Celcius", "Fernheit"])
                            .with_desc("The unit of temperature"),
                    ),
                ],
                ["location", "unit"],
            ),
            Some(
                ToolDescArg::new_number().with_desc("Null if the given city name is unavailable."),
            ),
        )];
        let mut msgs = vec![Message::with_role(Role::User).with_contents([Part::Text(
            "How much hot currently in Dubai? Answer in Celcius.".to_owned(),
        )])];
        let mut agg = MessageAggregator::new();
        let mut assistant_msg: Option<Message> = None;
        {
            let mut strm = model.run(msgs.clone(), tools.clone());
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
        let tool_call_id = if let Part::Function { id, .. } = assistant_msg.tool_calls[0].clone() {
            Some(id)
        } else {
            None
        };
        let tool_result_msg = Message::with_role(Role::Tool("temperature".into(), tool_call_id))
            .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
        msgs.push(tool_result_msg);

        let mut strm = model.run(msgs, tools);
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
    async fn openai_infer_with_image() {
        use base64::Engine;

        use super::*;
        use crate::value::MessageAggregator;

        let client = reqwest::Client::new();
        let test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jensen_Huang_%28cropped%29.jpg/250px-Jensen_Huang_%28cropped%29.jpg";
        let response = client.get(test_image_url).header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36").send().await.unwrap();
        let image_bytes = response.bytes().await.unwrap();
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let mut model = OpenAILanguageModel::new("gpt-4.1", *OPENAI_API_KEY);
        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents(vec![Part::ImageData(image_base64, "image/jpeg".into())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("What is shown in this image?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta));
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg));
            }
        }
    }

    #[multi_platform_test]
    async fn openai_infer_structured_output() {
        use serde_json::json;

        use super::OpenAIGenerationConfigBuilder;
        use crate::{
            model::api::openai_chat_completion::{
                OpenAIResponseFormat, OpenAIResponseFormatJSONSchema,
            },
            value::MessageAggregator,
        };

        let json_schema = serde_json::from_value::<OpenAIResponseFormatJSONSchema>(json!({
            "name": "summarize-content",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A brief summary of the content."
                    }
                },
                "required": [
                    "summary"
                ],
                "additionalProperties": false
            }
        }))
        .unwrap();
        let config = OpenAIGenerationConfigBuilder::default()
            .response_format(Some(OpenAIResponseFormat::JsonSchema { json_schema }))
            .build()
            .unwrap();

        let mut model = OpenAILanguageModel::new("gpt-4.1", *OPENAI_API_KEY).with_config(config);

        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("What is Artificial Intelligence?".into())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta));
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg));
            }
        }
    }
}
