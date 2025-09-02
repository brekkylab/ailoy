use futures::stream::StreamExt;
use gemini_rust::{
    Content as GeminiContent, FunctionCall as GeminiFunctionCall,
    FunctionDeclaration as GeminiFunctionDeclaration,
    FunctionParameters as GeminiFunctionParameters, Gemini,
    GenerationResponse as GeminiGenerationResponse, Message as GeminiMessage, Role as GeminiRole,
    Tool as GeminiTool,
};
pub use gemini_rust::{
    GenerationConfig as GeminiGenerationConfig, ThinkingConfig as GeminiThinkingConfig,
};

use crate::{
    model::LanguageModel,
    utils::BoxStream,
    value::{FinishReason, Message, MessageOutput, Part, Role, ToolDesc},
};

#[derive(Clone)]
pub struct GeminiLanguageModel {
    model_name: String,
    inner: Gemini,
    config: GeminiGenerationConfig,
}

impl GeminiLanguageModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model_name: String = model_name.into();
        let inner = Gemini::with_model(api_key.into(), format!("models/{}", model_name.clone()));
        Self {
            model_name,
            inner,
            config: GeminiGenerationConfig::default(),
        }
    }

    pub fn with_config(&mut self, config: GeminiGenerationConfig) -> Self {
        self.config = config;
        self.clone()
    }
}

fn gemini_response_to_ailoy(response: &GeminiGenerationResponse) -> Result<MessageOutput, String> {
    let candidate = response.candidates[0].clone();

    // https://ai.google.dev/api/generate-content#FinishReason
    let finish_reason = match candidate.finish_reason {
        Some(finish_reason) => match finish_reason.as_str() {
            "STOP" => {
                if candidate.content.parts.len() > 0 {
                    // Gemini does not provide "ToolCalls" finish reason explicitly,
                    // so we infer it by checking if there's FunctionCall part.
                    let part = candidate.content.parts[0].clone();
                    match part {
                        gemini_rust::Part::FunctionCall { .. } => Some(FinishReason::ToolCalls),
                        _ => Some(FinishReason::Stop),
                    }
                } else {
                    Some(FinishReason::Stop)
                }
            }
            "MAX_TOKENS" => {
                if !response.thoughts().is_empty() {
                    // Gemini can return MAX_TOKENS finish reason if it was thinking and it has been finished.
                    // In this case, the finish reason should not be set, and the generation should be continued.
                    None
                } else {
                    Some(FinishReason::Length)
                }
            }
            "SAFETY" | "RECITATION" | "LANGUAGE" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII"
            | "IMAGE_SAFETY" => Some(FinishReason::ContentFilter),
            _ => Some(FinishReason::Stop),
        },
        None => None,
    };

    let mut message = Message::new();
    message.role = Some(Role::Assistant);
    for part in candidate.content.parts.iter() {
        match part {
            gemini_rust::Part::Text { text, thought } => {
                if thought.is_some_and(|b| b) {
                    message.reasoning = text.clone();
                } else {
                    message.contents.push(Part::Text(text.clone()));
                }
            }
            gemini_rust::Part::FunctionCall { function_call } => {
                message.tool_calls.push(Part::new_function(
                    "", // Gemini does not return tool call id
                    function_call.name.clone(),
                    function_call.args.to_string(),
                ));
            }
            gemini_rust::Part::InlineData { .. } => {
                todo!("Gemini outputs other than text is not supported")
            }
            gemini_rust::Part::FunctionResponse { .. } => {
                panic!("Function Response cannot be returned from model")
            }
        }
    }

    let output = MessageOutput {
        delta: message,
        finish_reason,
    };

    Ok(output)
}

impl LanguageModel for GeminiLanguageModel {
    fn run<'a>(
        self: &'a mut Self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        let mut content = self.inner.generate_content();

        // Parse messages
        for msg in msgs.iter() {
            match &msg.role {
                Some(role) => match role {
                    Role::System => {
                        content = content.with_system_prompt(msg.contents[0].to_string().unwrap());
                    }
                    Role::User => {
                        let part = msg.contents[0].clone();
                        match part {
                            Part::Text(text) => {
                                content = content.with_user_message(text);
                            }
                            Part::ImageData(base64, mime_type) => {
                                content = content.with_message(GeminiMessage {
                                    role: GeminiRole::User,
                                    content: GeminiContent::inline_data(mime_type, base64),
                                });
                            }
                            Part::ImageURL(_) => {
                                panic!("Gemini does not support public image URL input")
                            }
                            _ => panic!("This part cannot belong to user"),
                        }
                    }
                    Role::Assistant => {
                        if msg.contents.len() > 0 {
                            let part = msg.contents[0].clone();
                            match part {
                                Part::Text(text) => {
                                    content = content.with_model_message(text);
                                }
                                _ => panic!("This part cannot belong to assistant contents"),
                            }
                        } else if msg.tool_calls.len() > 0 {
                            let tc = msg.tool_calls[0].clone();
                            match tc {
                                Part::Function {
                                    name, arguments, ..
                                } => {
                                    let function_call =
                                        GeminiContent::function_call(GeminiFunctionCall {
                                            name: name,
                                            args: serde_json::from_str(arguments.as_str()).unwrap(),
                                        });
                                    content = content.with_message(GeminiMessage {
                                        content: function_call,
                                        role: GeminiRole::Model,
                                    });
                                }
                                Part::FunctionString(_) => {
                                    panic!("Function call should be in a completed form")
                                }
                                _ => panic!("This part cannot belong to assistant tool_calls"),
                            }
                        } else {
                            panic!("Assistant message does not have any content")
                        }
                    }
                    Role::Tool(tool_name, _) => {
                        content = content.with_function_response(
                            tool_name,
                            serde_json::from_str(&msg.contents[0].to_string().unwrap()).unwrap(),
                        );
                    }
                },
                None => {
                    panic!("Message role should not be None");
                }
            }
        }

        // Parse tools
        if tools.len() > 0 {
            content = content.with_tool(GeminiTool::with_functions(
                tools
                    .iter()
                    .map(|tool| {
                        let tool_params = serde_json::to_value(&tool.parameters).unwrap();
                        let function_params =
                            serde_json::from_value::<GeminiFunctionParameters>(tool_params)
                                .unwrap();
                        GeminiFunctionDeclaration::new(
                            tool.name.clone(),
                            tool.description.clone(),
                            function_params,
                        )
                    })
                    .collect(),
            ));
        }

        // Set generation config
        content = content.with_generation_config(self.config.clone());

        let strm = async_stream::try_stream! {
            let mut stream = content.execute_stream().await.unwrap();
            while let Some(resp) = stream.next().await {
                let resp = resp.map_err(|e| e.to_string()).unwrap();
                let output = gemini_response_to_ailoy(&resp)?;
                yield output;
            }
        };
        Box::pin(strm)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::log;
    use ailoy_macros::multi_platform_test;
    use std::sync::LazyLock;

    static GEMINI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("GEMINI_API_KEY")
            .expect("Environment variable 'GEMINI_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn gemini_infer_with_thinking() {
        use super::*;
        use crate::value::MessageAggregator;

        let mut gemini_config = GeminiGenerationConfig::default();
        gemini_config.max_output_tokens = Some(2048);
        gemini_config.thinking_config =
            Some(GeminiThinkingConfig::default().with_thoughts_included(true));
        let gemini = Arc::new(
            GeminiLanguageModel::new("gemini-2.5-flash", *GEMINI_API_KEY)
                .with_config(gemini_config),
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
        let mut strm = gemini.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log::debug(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn gemini_infer_tool_call() {
        use super::*;
        use crate::value::{MessageAggregator, ToolDescArg};

        let gemini = Arc::new(GeminiLanguageModel::new(
            "gemini-2.5-flash",
            *GEMINI_API_KEY,
        ));

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
            let mut strm = gemini.run(msgs.clone(), tools.clone());
            while let Some(delta_opt) = strm.next().await {
                let delta = delta_opt.unwrap();
                log::debug(format!("{:?}", delta).as_str());
                if let Some(msg) = agg.update(delta) {
                    log::debug(format!("{:?}", msg).as_str());
                    assistant_msg = Some(msg);
                }
            }
        }
        // This should be tool call message
        let assistant_msg = assistant_msg.unwrap();
        msgs.push(assistant_msg.clone());

        // Append a fake tool call result message
        let tool_result_msg = Message::with_role(Role::Tool("temperature".into(), None))
            .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
        msgs.push(tool_result_msg);

        let mut strm = gemini.run(msgs, tools);
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
    async fn gemini_infer_with_image() {
        use super::*;
        use crate::value::MessageAggregator;

        use base64::Engine;

        let client = reqwest::Client::new();
        let test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jensen_Huang_%28cropped%29.jpg/250px-Jensen_Huang_%28cropped%29.jpg";
        let response = client.get(test_image_url).header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36").send().await.unwrap();
        let image_bytes = response.bytes().await.unwrap();
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let gemini = Arc::new(GeminiLanguageModel::new(
            "gemini-2.5-flash",
            *GEMINI_API_KEY,
        ));

        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents(vec![Part::ImageData(image_base64, "image/jpeg".into())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("What is shown in this image?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = gemini.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log::debug(format!("{:?}", delta));
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg));
            }
        }
    }
}
