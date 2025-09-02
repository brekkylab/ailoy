use std::collections::HashMap;

use anthropic_ai_sdk::{
    client::{AnthropicClient, AnthropicClientBuilder},
    types::message::{
        ContentBlock as AnthropicContentBlock, ContentBlockDelta as AnthropicContentBlockDelta,
        CreateMessageParams, Message as AnthropicMessage, MessageClient, MessageError,
        Metadata as AnthropicMetadata, RequiredMessageParams, Role as AnthropicRole,
        StopReason as AnthropicStopReason, StreamEvent as AnthropicStreamEvent,
        Thinking as AnthropicThinking, ThinkingType as AnthropicThinkingType,
        Tool as AnthropicTool, ToolChoice as _AnthropicToolChoice,
    },
};
use futures::StreamExt;

use crate::{
    model::LanguageModel,
    utils::{BoxStream, log},
    value::{FinishReason, Message, MessageOutput, Part, Role},
};

#[derive(Clone)]
pub struct AnthropicGenerationConfig {
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub tool_choice: Option<AnthropicToolChoice>,
    pub thinking: Option<AnthropicThinkingConfig>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Clone)]
pub enum AnthropicToolChoice {
    Auto,
    Any,
    Tool { name: String },
    None,
}

#[derive(Clone)]
pub struct AnthropicThinkingConfig {
    pub enabled: bool,
    pub budget_tokens: usize,
}

impl Default for AnthropicGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: None,
            stop_sequences: None,
            top_k: None,
            top_p: None,
            tool_choice: None,
            thinking: None,
            metadata: None,
        }
    }
}

impl Default for AnthropicThinkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            budget_tokens: 1024,
        }
    }
}

#[derive(Clone)]
pub struct AnthropicLanguageModel {
    model_name: String,
    inner: AnthropicClient,
    config: AnthropicGenerationConfig,
}

impl AnthropicLanguageModel {
    pub fn new(model_name: impl Into<String>, api_key: impl Into<String>) -> Self {
        let client = AnthropicClientBuilder::new(api_key, "2023-06-01")
            .build::<MessageError>()
            .unwrap();
        Self {
            model_name: model_name.into(),
            inner: client,
            config: AnthropicGenerationConfig::default(),
        }
    }

    pub fn with_config(&mut self, config: AnthropicGenerationConfig) -> Self {
        self.config = config;
        self.clone()
    }
}

fn anthropic_stream_event_to_ailoy(
    evt: AnthropicStreamEvent,
) -> Result<Option<MessageOutput>, String> {
    match evt {
        AnthropicStreamEvent::MessageStart { message } => {
            log::debug(format!(
                "[Anthropic] new message started (id: {})",
                message.id
            ));
            Ok(Some(
                MessageOutput::new().with_delta(Message::with_role(Role::Assistant)),
            ))
        }
        AnthropicStreamEvent::ContentBlockStart {
            index,
            content_block,
        } => {
            log::debug(format!(
                "[Anthropic] content block (idx: {}) start: {:?}",
                index, content_block
            ));
            match content_block {
                AnthropicContentBlock::Text { text } => Ok(Some(
                    MessageOutput::new()
                        .with_delta(Message::new().with_contents(vec![Part::Text(text)])),
                )),
                AnthropicContentBlock::Image { source } => Ok(Some(
                    MessageOutput::new().with_delta(
                        Message::new()
                            .with_contents(vec![Part::ImageData(source.data, source.media_type)]),
                    ),
                )),
                AnthropicContentBlock::ToolUse { id, name, .. } => {
                    // tool_use.input always starts with "{}".
                    // The arguments will be eventually assembled by following content block deltas,
                    // so we ignore the first arguments string.
                    Ok(Some(MessageOutput::new().with_delta(
                        Message::new().with_tool_calls(vec![Part::Function {
                            id,
                            name,
                            arguments: "".into(),
                        }]),
                    )))
                }
                AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                } => {
                    // This happens on "server_tool_use".
                    // There's no way to figure out the name of server tool, so fill it as empty.
                    Ok(Some(
                        MessageOutput::new().with_delta(
                            Message::with_role(Role::Tool("".into(), Some(tool_use_id)))
                                .with_contents(vec![Part::Text(content)]),
                        ),
                    ))
                }
                AnthropicContentBlock::Thinking { thinking, .. } => Ok(Some(
                    MessageOutput::new().with_delta(Message::new().with_reasoning(thinking)),
                )),
                AnthropicContentBlock::RedactedThinking { .. } => {
                    log::warn("[Anthropic] Redacted thinking is not supported");
                    Ok(None)
                }
            }
        }
        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
            log::debug(format!(
                "[Anthropic] content block (idx: {}) delta: {:?}",
                index, delta
            ));
            match delta {
                AnthropicContentBlockDelta::TextDelta { text } => Ok(Some(
                    MessageOutput::new()
                        .with_delta(Message::new().with_contents(vec![Part::Text(text)])),
                )),
                AnthropicContentBlockDelta::InputJsonDelta { partial_json } => {
                    Ok(Some(MessageOutput::new().with_delta(
                        Message::new().with_tool_calls(vec![Part::Function {
                            id: "".into(),
                            name: "".into(),
                            arguments: partial_json,
                        }]),
                    )))
                }
                AnthropicContentBlockDelta::ThinkingDelta { thinking } => Ok(Some(
                    MessageOutput::new().with_delta(Message::new().with_reasoning(thinking)),
                )),
                AnthropicContentBlockDelta::SignatureDelta { .. } => {
                    log::warn("[Anthropic] Thinking signature delta is ignored");
                    Ok(None)
                }
            }
        }
        AnthropicStreamEvent::ContentBlockStop { index } => {
            log::debug(format!("[Anthropic] content block (idx: {}) stop", index));
            Ok(None)
        }
        AnthropicStreamEvent::MessageDelta { delta, .. } => {
            log::debug(format!("[Anthropic] message delta: {:?}", delta));
            let stop_reason = delta
                .stop_reason
                .expect("MessageDelta should have stop_reason");
            match stop_reason {
                AnthropicStopReason::EndTurn => Ok(Some(
                    MessageOutput::new().with_finish_reason(FinishReason::Stop),
                )),
                AnthropicStopReason::MaxTokens => Ok(Some(
                    MessageOutput::new().with_finish_reason(FinishReason::Length),
                )),
                AnthropicStopReason::StopSequence => Ok(Some(
                    MessageOutput::new().with_finish_reason(FinishReason::Stop),
                )),
                AnthropicStopReason::ToolUse => Ok(Some(
                    MessageOutput::new().with_finish_reason(FinishReason::ToolCalls),
                )),
                AnthropicStopReason::Refusal => Ok(Some(
                    MessageOutput::new().with_finish_reason(FinishReason::ContentFilter),
                )),
            }
        }
        AnthropicStreamEvent::MessageStop => {
            log::debug("[Anthropic] Message stop");
            Ok(None)
        }
        AnthropicStreamEvent::Ping => {
            log::debug("[Anthropic] Ping");
            Ok(None)
        }
        AnthropicStreamEvent::Error { error } => {
            Err(format!("type: {}, message: {}", error.type_, error.message))
        }
    }
}

impl LanguageModel for AnthropicLanguageModel {
    fn run(
        self: std::sync::Arc<Self>,
        msgs: Vec<crate::value::Message>,
        tools: Vec<crate::value::ToolDesc>,
    ) -> BoxStream<'static, Result<crate::value::MessageOutput, String>> {
        let mut params = CreateMessageParams::new(RequiredMessageParams {
            model: self.model_name.clone(),
            messages: vec![],
            max_tokens: self.config.max_tokens,
        });

        // Parse messages
        for msg in msgs {
            match &msg.role {
                Some(role) => match role {
                    Role::System => {
                        // Anthropic does not consider system message as a general message.
                        // It's rather considered as one of the generation config.
                        let system_message = msg.contents[0].to_string().unwrap();
                        params.system = Some(system_message.to_string());
                        continue;
                    }
                    Role::User => {
                        let mut content_blocks: Vec<AnthropicContentBlock> = vec![];
                        for part in msg.contents.iter() {
                            match part {
                                Part::Text(text) => {
                                    content_blocks.push(AnthropicContentBlock::text(text));
                                }
                                Part::ImageData(base64, mime_type) => {
                                    content_blocks.push(AnthropicContentBlock::image(
                                        "base64".to_string(),
                                        mime_type,
                                        base64,
                                    ));
                                }
                                Part::ImageURL(_) => {
                                    // let mime_type =
                                    //     mime_infer::from_path(url).first().unwrap().to_string();
                                    // let block = AnthropicContentBlock::Image {
                                    //     source: AnthropicImageSource {
                                    //         type_: "url".into(),
                                    //         media_type: mime_type,
                                    //         data: url,
                                    //     },
                                    // };
                                    // content_blocks.push(block);
                                    todo!(
                                        "Anthropic indeed supports image url input, but anthropic-ai-sdk currently does not support it."
                                    )
                                }
                                _ => panic!("This part cannot belong to user"),
                            }
                        }
                        params.messages.push(AnthropicMessage::new_blocks(
                            AnthropicRole::User,
                            content_blocks,
                        ));
                    }
                    Role::Assistant => {
                        let mut content_blocks: Vec<AnthropicContentBlock> = vec![];
                        for part in msg.contents.iter() {
                            match part {
                                Part::Text(text) => {
                                    content_blocks.push(AnthropicContentBlock::text(text));
                                }
                                _ => panic!("This part cannot belong to assistant contents"),
                            }
                        }
                        for tc in msg.tool_calls.into_iter() {
                            match tc {
                                Part::Function {
                                    id,
                                    name,
                                    arguments,
                                } => {
                                    content_blocks.push(AnthropicContentBlock::ToolUse {
                                        id,
                                        name,
                                        input: serde_json::from_str(arguments.as_str()).unwrap(),
                                    });
                                }
                                Part::FunctionString(_) => {
                                    panic!("Function call should be in a completed form")
                                }
                                _ => panic!("This part cannot belong to assistant tool_calls"),
                            }
                        }
                        params.messages.push(AnthropicMessage::new_blocks(
                            AnthropicRole::Assistant,
                            content_blocks,
                        ));
                    }
                    Role::Tool(_, tool_call_id) => {
                        let mut content_blocks: Vec<AnthropicContentBlock> = vec![];
                        let tool_call_id = tool_call_id.clone().unwrap();
                        for part in msg.contents.iter() {
                            match part {
                                Part::Text(text) => {
                                    content_blocks.push(AnthropicContentBlock::ToolResult {
                                        tool_use_id: tool_call_id.clone(),
                                        content: text.to_string(),
                                    });
                                }
                                _ => {
                                    panic!("Tool results other than text are not supported")
                                }
                            }
                        }
                        params.messages.push(AnthropicMessage::new_blocks(
                            AnthropicRole::User,
                            content_blocks,
                        ));
                    }
                },
                None => panic!("Message role should not be None"),
            }
        }

        // Parse tools
        if tools.len() > 0 {
            params.tools = Some(
                tools
                    .into_iter()
                    .map(|tool| AnthropicTool {
                        name: tool.name,
                        description: Some(tool.description),
                        input_schema: serde_json::from_value(
                            serde_json::to_value(tool.parameters).unwrap(),
                        )
                        .unwrap(),
                    })
                    .collect::<Vec<AnthropicTool>>(),
            );
        }

        // Set generation configs
        if let Some(temperature) = self.config.temperature {
            params.temperature = Some(temperature);
        }
        if let Some(stop_sequences) = &self.config.stop_sequences {
            params.stop_sequences = Some(stop_sequences.clone());
        }
        if let Some(top_k) = self.config.top_k {
            params.top_k = Some(top_k);
        }
        if let Some(top_p) = self.config.top_p {
            params.top_p = Some(top_p);
        }
        if let Some(tool_choice) = &self.config.tool_choice {
            match tool_choice {
                AnthropicToolChoice::Auto => {
                    params.tool_choice = Some(_AnthropicToolChoice::Auto);
                }
                AnthropicToolChoice::Any => {
                    params.tool_choice = Some(_AnthropicToolChoice::Any);
                }
                AnthropicToolChoice::Tool { name } => {
                    params.tool_choice = Some(_AnthropicToolChoice::Tool { name: name.clone() });
                }
                AnthropicToolChoice::None => {
                    params.tool_choice = Some(_AnthropicToolChoice::None);
                }
            }
        }
        if let Some(thinking) = self.config.thinking.clone()
            && thinking.enabled
        {
            params.thinking = Some(AnthropicThinking {
                budget_tokens: thinking.budget_tokens,
                type_: AnthropicThinkingType::Enabled,
            });
        }
        if let Some(metadata) = &self.config.metadata {
            params.metadata = Some(AnthropicMetadata {
                fields: metadata.clone(),
            });
        }

        // Always streaming
        params = params.with_stream(true);

        let strm = async_stream::try_stream! {
            match self.inner.create_message_streaming(&params).await {
                Ok(mut stream) => {
                    while let Some(result) = stream.next().await {
                        let resp: AnthropicStreamEvent = result.map_err(|e| e.to_string()).unwrap();
                        match anthropic_stream_event_to_ailoy(resp) {
                            Ok(output) => {
                                if let Some(output) = output {
                                    yield output;
                                } else {
                                    continue;
                                }
                            },
                            Err(e) => {
                                panic!("Error occurred during streaming: {}", e)
                            }
                        }
                    }
                }
                Err(e) => {
                    panic!("Failed to send request: {}", e)
                }
            }
        };
        Box::pin(strm)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::value::{MessageAggregator, ToolDesc, ToolDescArg};
    use ailoy_macros::multi_platform_test;
    use std::sync::LazyLock;

    static ANTHROPIC_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("ANTHROPIC_API_KEY")
            .expect("Environment variable 'ANTHROPIC_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn anthropic_infer_with_thinking() {
        let mut config = AnthropicGenerationConfig::default();
        config.max_tokens = 2048;
        config.thinking = Some(AnthropicThinkingConfig::default());
        let anthropic = Arc::new(
            AnthropicLanguageModel::new("claude-sonnet-4-20250514", *ANTHROPIC_API_KEY)
                .with_config(config),
        );

        let msgs = vec![
            Message::with_role(Role::System).with_contents(vec![Part::Text(
                "You are a helpful mathematics assistant".to_owned(),
            )]),
            Message::with_role(Role::User).with_contents(vec![Part::Text(
                "What is the sum of the first 50 prime numbers?".to_owned(),
            )]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = anthropic.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn anthropic_infer_tool_call() {
        let anthropic = Arc::new(AnthropicLanguageModel::new(
            "claude-sonnet-4-20250514",
            *ANTHROPIC_API_KEY,
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
        let mut strm = anthropic.clone().run(msgs.clone(), tools.clone());
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
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
        } else {
            None
        };
        let tool_result_msg = Message::with_role(Role::Tool("temperature".into(), tool_call_id))
            .with_contents(vec![Part::Text("{\"temperature\": 38.5}".into())]);
        msgs.push(tool_result_msg);

        let mut strm = anthropic.run(msgs, tools);
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                // Final message shuold say something like "Dubai is 38.5°C"
                log::info(format!("{:?}", msg).as_str());
            }
        }
    }

    #[multi_platform_test]
    async fn anthropic_infer_with_image() {
        use base64::Engine;

        let client = reqwest::Client::new();
        let test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Jensen_Huang_%28cropped%29.jpg/250px-Jensen_Huang_%28cropped%29.jpg";
        let response = client.get(test_image_url).header(reqwest::header::USER_AGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36").send().await.unwrap();
        let image_bytes = response.bytes().await.unwrap();
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let anthropic = Arc::new(AnthropicLanguageModel::new(
            "claude-sonnet-4-20250514",
            *ANTHROPIC_API_KEY,
        ));

        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents(vec![Part::ImageData(image_base64, "image/jpeg".into())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("What is shown in this image?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = anthropic.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                log::info(format!("{:?}", msg));
            }
        }
    }
}
