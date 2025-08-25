use futures::StreamExt;
use openai_sdk_rs::{
    OpenAI,
    types::chat::{
        ChatCompletionChunk as OpenAIChatCompletionChunk, ChatCompletionRequest,
        ChatMessage as OpenAIChatMessage, ContentBlock as OpenAIContentBlock,
        Function as OpenAIFunction, Role as OpenAIRole, Tool as OpenAITool,
        ToolCall as OpenAIToolCall, ToolCallFunction as OpenAIToolCallFunction,
        ToolType as OpenAIToolType,
    },
};

use crate::{
    model::LanguageModel,
    utils::BoxStream,
    value::{FinishReason, Message, MessageOutput, Part, Role, ToolDesc},
};

pub trait OpenAIChatCompletion: LanguageModel {
    fn model_name(&self) -> &str;

    fn client(&self) -> &OpenAI;

    /// Build the chat completion request - can be overridden for service-specific behavior
    fn build_request(
        &self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> Result<ChatCompletionRequest, String> {
        let mut request = ChatCompletionRequest::default();
        request.model = self.model_name().to_string();

        // Apply base configurations
        // if let Some(max_tokens) = self.config().max_tokens {
        //     request.max_tokens = Some(max_tokens);
        // }
        // if let Some(temperature) = self.config().temperature {
        //     request.temperature = Some(temperature);
        // }

        // Process messages
        for msg in msgs {
            let message = self.build_message(msg)?;
            request.messages.push(message);
        }

        // Add tools
        if !tools.is_empty() {
            request.tools = Some(self.build_tools(tools)?);
        }

        Ok(request)
    }

    /// Build a single message - can be overridden for custom message handling
    fn build_message(&self, msg: Message) -> Result<OpenAIChatMessage, String> {
        match &msg.role {
            Some(role) => match role {
                Role::System => {
                    let content = msg.contents[0]
                        .to_string()
                        .expect("The system message should be exist in contents");
                    Ok(OpenAIChatMessage::system(content))
                }
                Role::User => {
                    let content_blocks = self.build_user_content(msg.contents)?;
                    Ok(OpenAIChatMessage::new_blocks(
                        OpenAIRole::User,
                        content_blocks,
                    ))
                }
                Role::Assistant => self.build_assistant_message(msg),
                Role::Tool(_, tool_call_id) => self.build_tool_message(msg.clone(), tool_call_id),
            },
            None => Err("Message role cannot be None".to_string()),
        }
    }

    /// Build user content blocks - can be overridden for service-specific content handling
    fn build_user_content(&self, contents: Vec<Part>) -> Result<Vec<OpenAIContentBlock>, String> {
        let mut blocks = Vec::new();

        for part in contents {
            match part {
                Part::Text(text) => {
                    blocks.push(OpenAIContentBlock::text(text));
                }
                Part::ImageURL(url) => {
                    blocks.push(OpenAIContentBlock::image(url, None));
                }
                Part::ImageData(_, _) => {
                    let base64_url = part
                        .to_string()
                        .expect("The base64 data url should not be None");
                    blocks.push(OpenAIContentBlock::image(base64_url, None));
                }
                _ => return Err("Invalid content type for user message".to_string()),
            }
        }

        Ok(blocks)
    }

    /// Build assistant message - can be overridden
    fn build_assistant_message(&self, msg: Message) -> Result<OpenAIChatMessage, String> {
        let mut content_blocks = Vec::new();

        for part in msg.contents {
            match part {
                Part::Text(text) => content_blocks.push(OpenAIContentBlock::text(text)),
                _ => return Err("Invalid content type for assistant message".to_string()),
            }
        }

        let mut message = OpenAIChatMessage::new_blocks(OpenAIRole::Assistant, content_blocks);

        if !msg.tool_calls.is_empty() {
            message.tool_calls = Some(self.build_tool_calls(msg.tool_calls)?);
        }

        Ok(message)
    }

    /// Build tool message - can be overridden
    fn build_tool_message(
        &self,
        msg: Message,
        tool_call_id: &Option<String>,
    ) -> Result<OpenAIChatMessage, String> {
        let tool_call_id = tool_call_id.as_ref().ok_or("Tool call ID required")?;

        let mut content_blocks = Vec::new();
        for part in msg.contents {
            match part {
                Part::Text(text) => content_blocks.push(OpenAIContentBlock::Text { text }),
                _ => return Err("Only text supported for tool results".to_string()),
            }
        }

        Ok(
            OpenAIChatMessage::new_blocks(OpenAIRole::Tool, content_blocks)
                .tool_call_id(tool_call_id.clone()),
        )
    }

    /// Build tool calls - can be overridden
    fn build_tool_calls(&self, tool_calls: Vec<Part>) -> Result<Vec<OpenAIToolCall>, String> {
        let mut calls = Vec::new();

        for tc in tool_calls {
            match tc {
                Part::Function {
                    id,
                    name,
                    arguments,
                } => {
                    calls.push(OpenAIToolCall {
                        id,
                        r#type: "function".to_string(),
                        function: OpenAIToolCallFunction { name, arguments },
                    });
                }
                _ => return Err("Invalid tool call format".to_string()),
            }
        }

        Ok(calls)
    }

    /// Build tools - can be overridden
    fn build_tools(&self, tools: Vec<ToolDesc>) -> Result<Vec<OpenAITool>, String> {
        tools
            .into_iter()
            .map(|tool| {
                let parameters = serde_json::from_value(
                    serde_json::to_value(tool.parameters)
                        .map_err(|e| format!("Serialize error: {}", e))?,
                )
                .map_err(|e| format!("Deserialize error: {}", e))?;

                Ok(OpenAITool {
                    r#type: OpenAIToolType::Function,
                    function: OpenAIFunction {
                        name: tool.name,
                        description: Some(tool.description),
                        parameters,
                    },
                })
            })
            .collect()
    }

    /// Parse response chunk - can be overridden for service-specific response handling
    fn parse_chunk(chunk: OpenAIChatCompletionChunk) -> Result<Option<MessageOutput>, String> {
        let choice = chunk.choices.first().ok_or("Empty chunk choices")?;

        let mut output = MessageOutput::new();
        let delta = &choice.delta;

        // Handle role
        if let Some(role) = &delta.role {
            output.delta.role = match role {
                OpenAIRole::System => Some(Role::System),
                OpenAIRole::User => Some(Role::User),
                OpenAIRole::Assistant => Some(Role::Assistant),
                OpenAIRole::Tool => return Err("Tool role not allowed in response".to_string()),
            };
        }

        // Handle content
        if let Some(content) = &delta.content {
            output.delta.contents.push(Part::Text(content.clone()));
        }

        // Handle reasoning (if supported)
        if let Some(reasoning_content) = &delta.reasoning_content {
            output.delta.reasoning = reasoning_content.clone();
        }

        // Handle tool calls
        if let Some(tool_calls) = &delta.tool_calls {
            output.delta.tool_calls = tool_calls
                .iter()
                .map(|tc| {
                    let tool_call_id = tc.id.clone().unwrap_or("".into());
                    let (tool_name, tool_args) = if let Some(function) = tc.function.clone() {
                        (
                            function.name.unwrap_or("".into()),
                            function.arguments.unwrap_or("".into()),
                        )
                    } else {
                        ("".into(), "".into())
                    };
                    Part::new_function(tool_call_id, tool_name, tool_args)
                })
                .collect::<Vec<Part>>();
        }

        // Handle finish reason
        if let Some(finish_reason) = &choice.finish_reason {
            output.finish_reason = Some(Self::parse_finish_reason(finish_reason.as_str())?);
        }

        Ok(Some(output))
    }

    /// Parse finish reason - can be overridden for service-specific finish reasons
    fn parse_finish_reason(finish_reason: &str) -> Result<FinishReason, String> {
        match finish_reason {
            "stop" => Ok(FinishReason::Stop),
            "length" => Ok(FinishReason::Length),
            "tool_calls" => Ok(FinishReason::ToolCalls),
            "content_filter" => Ok(FinishReason::ContentFilter),
            _ => return Err(format!("Unknown finish reason: {}", finish_reason)),
        }
    }
}

impl<T> LanguageModel for T
where
    T: OpenAIChatCompletion + Clone,
{
    fn run(
        self: std::sync::Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'static, Result<MessageOutput, String>> {
        let request = match self.build_request(msgs, tools) {
            Ok(req) => req,
            Err(e) => {
                return Box::pin(async_stream::try_stream! {
                    yield Err(e)?;
                });
            }
        };

        let client = self.client().clone();

        let strm = async_stream::try_stream! {
            let mut stream = client.chat_completion_stream(request).await
                .map_err(|e| format!("Stream creation failed: {}", e))?;

            while let Some(result) = stream.next().await {
                let chunk = result.map_err(|e| format!("Stream error: {}", e))?;
                let output = Self::parse_chunk(chunk)?;
                if let Some(output) = output {
                    yield output;
                }
            }
        };

        Box::pin(strm)
    }
}
