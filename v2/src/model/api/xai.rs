use openai_sdk_rs::OpenAI;

use crate::model::api::openai_chat_completion::OpenAIChatCompletion;
use crate::value::FinishReason;

#[derive(Clone)]
pub struct XAILanguageModel {
    model_name: String,
    client: OpenAI,
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
    use super::*;
    use crate::model::LanguageModel;
    use crate::multi_platform_test;
    use crate::utils::log;
    use crate::value::{Message, MessageAggregator, Part, Role, ToolDesc};
    use futures::StreamExt;

    const XAI_API_KEY: &str = env!("XAI_API_KEY");

    multi_platform_test! {
        async fn xai_infer_with_thinking() {
            let xai = std::sync::Arc::new(
                XAILanguageModel::new("grok-3-mini", XAI_API_KEY)
            );

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
    }

    multi_platform_test! {
        async fn xai_infer_tool_call() {
            use super::*;
            use crate::value::{MessageAggregator, ToolDescArg};

            let xai = std::sync::Arc::new(
                XAILanguageModel::new("grok-3", XAI_API_KEY)
            );

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
            let mut strm = xai.clone().run(msgs.clone(), tools.clone());
            let mut assistant_msg: Option<Message> = None;
            while let Some(delta_opt) = strm.next().await {
                let delta = delta_opt.unwrap();
                log::debug(format!("{:?}", delta).as_str());
                if let Some(msg) = agg.update(delta) {
                    log::info(format!("{:?}", msg).as_str());
                    assistant_msg = Some(msg);
                }
            }
            // This should be tool call message
            let assistant_msg = assistant_msg.unwrap();
            msgs.push(assistant_msg.clone());

            // Append a fake tool call result message
            let tool_call_id = if let Part::Function { id, .. } = assistant_msg.tool_calls[0].clone() {
                Some(id)
            } else {None};
            let tool_result_msg = Message::with_role(Role::Tool("temperature".into(), tool_call_id))
                .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
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
    }

    multi_platform_test! {
        async fn xai_infer_with_image() {
            use super::*;
            use crate::value::MessageAggregator;

            use base64::Engine;

            let client = reqwest::Client::new();
            let test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jensen_Huang_%28cropped%29.jpg/250px-Jensen_Huang_%28cropped%29.jpg";
            let response = client.get(test_image_url).header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36").send().await.unwrap();
            let image_bytes = response.bytes().await.unwrap();
            let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

            let xai = std::sync::Arc::new(
                XAILanguageModel::new("grok-4", XAI_API_KEY)
            );

            let msgs = vec![
                Message::with_role(Role::User)
                    .with_contents(vec![
                        Part::ImageData(image_base64, "image/jpeg".into())
                    ]),
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
}
