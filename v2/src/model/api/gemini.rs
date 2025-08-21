use std::sync::Arc;

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
                    "",
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
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'static, Result<MessageOutput, String>> {
        let mut content = self.inner.generate_content();

        // Parse messages
        for msg in msgs.iter() {
            match &msg.role {
                Some(role) => match role {
                    Role::System => {
                        content = content.with_system_prompt(msg.contents[0].as_str().unwrap());
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
                    Role::Tool(tool_name) => {
                        content = content.with_function_response(
                            tool_name,
                            serde_json::from_str(&msg.contents[0].as_str().unwrap()).unwrap(),
                        );
                    }
                },
                None => {
                    panic!("Message role should not be None");
                }
            }
        }

        // Parse tools
        content = content.with_tool(GeminiTool::with_functions(
            tools
                .iter()
                .map(|tool| {
                    let tool_params = serde_json::to_value(&tool.parameters).unwrap();
                    let function_params =
                        serde_json::from_value::<GeminiFunctionParameters>(tool_params).unwrap();
                    GeminiFunctionDeclaration::new(
                        tool.name.clone(),
                        tool.description.clone(),
                        function_params,
                    )
                })
                .collect(),
        ));

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
    use std::sync::LazyLock;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen::prelude::*;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::*;

    #[cfg(not(target_arch = "wasm32"))]
    static GEMINI_API_KEY: LazyLock<String> = LazyLock::new(|| {
        dotenv::dotenv().ok();
        std::env::var("GEMINI_API_KEY").unwrap_or_else(|_| {
            eprintln!("Warning: GEMINI_API_KEY is not set");
            "".to_string()
        })
    });

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn log(s: &str) {
        println!("{}", s);
    }

    #[cfg(target_arch = "wasm32")]
    static GEMINI_API_KEY: LazyLock<String> = LazyLock::new(|| "".to_string());

    async fn _gemini_infer_with_thinking() {
        use super::*;
        use crate::value::MessageAggregator;

        let mut gemini_config = GeminiGenerationConfig::default();
        gemini_config.max_output_tokens = Some(2048);
        gemini_config.thinking_config =
            Some(GeminiThinkingConfig::default().with_thoughts_included(true));
        let gemini = Arc::new(
            GeminiLanguageModel::new("gemini-2.5-flash", GEMINI_API_KEY.as_str())
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
            log(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log(format!("{:?}", msg).as_str());
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn gemini_infer_with_thinking() {
        _gemini_infer_with_thinking().await
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    async fn gemini_infer_with_thinking() {
        _gemini_infer_with_thinking().await
    }

    async fn _gemini_infer_tool_call() {
        use super::*;
        use crate::value::{MessageAggregator, ToolDescArg};

        let gemini = Arc::new(GeminiLanguageModel::new(
            "gemini-2.5-flash",
            GEMINI_API_KEY.as_str(),
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
        let mut strm = gemini.clone().run(msgs.clone(), tools.clone());
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log(format!("{:?}", msg).as_str());
                assistant_msg = Some(msg);
            }
        }
        // This should be tool call message
        let assistant_msg = assistant_msg.unwrap();
        msgs.push(assistant_msg);

        // Append a fake tool call result message
        let tool_result_msg = Message::with_role(Role::Tool("temperature".into()))
            .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
        msgs.push(tool_result_msg);

        let mut strm = gemini.run(msgs, tools);
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            log(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log(format!("{:?}", msg).as_str());
                assistant_msg = Some(msg);
            }
        }
        // Final message shuold say something like "Dubai is 38.5°C"
        let assistant_msg = assistant_msg.unwrap();
        log(format!("Final Answer: {:?}", assistant_msg).as_str());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn gemini_infer_tool_call() {
        _gemini_infer_tool_call().await
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    async fn gemini_infer_tool_call() {
        _gemini_infer_tool_call().await
    }

    async fn _gemini_infer_with_image() {
        use super::*;
        use crate::value::MessageAggregator;

        use base64::Engine;

        #[cfg(target_arch = "wasm32")]
        let test_image_url = "https://newsroom.haas.berkeley.edu/wp-content/uploads/2023/02/jensen-huang-headshot2_thmb-300x246.jpg";
        #[cfg(not(target_arch = "wasm32"))]
        let test_image_url =
            "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg?w=385";

        let response = reqwest::get(test_image_url).await.unwrap();
        println!("response: {:?}", response);
        let image_bytes = response.bytes().await.unwrap();
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let gemini = Arc::new(GeminiLanguageModel::new(
            "gemini-2.5-flash",
            GEMINI_API_KEY.as_str(),
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
            log(format!("{:?}", delta).as_str());
            if let Some(msg) = agg.update(delta) {
                log(format!("{:?}", msg).as_str());
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn gemini_infer_with_image() {
        _gemini_infer_with_image().await
    }

    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen_test]
    async fn gemini_infer_with_image() {
        _gemini_infer_with_image().await
    }
}
