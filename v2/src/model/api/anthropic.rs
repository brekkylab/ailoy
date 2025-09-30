use crate::{
    model::{APIModel, sse::ServerSentEvent},
    value::{
        AnthropicMarshal, AnthropicUnmarshal, FinishReason, LMConfig, Marshaled, Message,
        MessageDelta, MessageOutput, ToolDesc, Unmarshaled,
    },
};

pub fn make_request(
    api_model: &APIModel,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: LMConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(&Marshaled::<_, AnthropicMarshal>::new(&config));

    body["messages"] = serde_json::json!(&Marshaled::<_, AnthropicMarshal>::new(&msgs));
    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!({"type": "auto"});
        body["tools"] = serde_json::json!(
            tools
                .iter()
                .map(|v| Marshaled::<_, AnthropicMarshal>::new(v))
                .collect::<Vec<_>>()
        );
    }

    reqwest::Client::new()
        .request(reqwest::Method::POST, api_model.endpoint())
        .header("x-api-key", api_model.api_key.clone())
        .header("anthropic-version", "2023-06-01")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub fn handle_event(evt: ServerSentEvent) -> MessageOutput {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };

    let finish_reason = val
        .pointer("/delta/stop_reason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "end_turn" => FinishReason::Stop,
            "pause_turn" => FinishReason::Stop, // consider same as "end_turn"
            "max_tokens" => FinishReason::Length,
            "tool_use" => FinishReason::ToolCall,
            "refusal" => {
                FinishReason::Refusal("Model output violated Anthropic's safety policy.".to_owned())
            }
            reason => FinishReason::Refusal(format!("reason: {}", reason)),
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal(_)) => MessageDelta::default(),
        _ => serde_json::from_value::<Unmarshaled<_, AnthropicUnmarshal>>(val.clone())
            .ok()
            .map(|decoded| decoded.get())
            .unwrap_or_default(),
    };

    MessageOutput {
        delta,
        finish_reason,
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        debug,
        model::{APIProvider, LanguageModel as _, sse::SSELanguageModel},
        to_value,
        value::{Delta, LMConfigBuilder, Part, Role, ToolDescBuilder},
    };

    const ANTHROPIC_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        let model = SSELanguageModel::new(APIModel::new(
            APIProvider::Anthropic,
            "claude-3-haiku-20240307",
            ANTHROPIC_API_KEY,
        ));

        let msgs =
            vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])];
        let config = LMConfigBuilder::new()
            .system_message("You are a helpful assistant.")
            .stream(true)
            .build();
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.run(msgs, Vec::new(), config);
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        let model = SSELanguageModel::new(APIModel::new(
            APIProvider::Anthropic,
            "claude-3-haiku-20240307",
            ANTHROPIC_API_KEY,
        ));
        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
        ];
        let config = LMConfigBuilder::new().stream(true).build();
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::ToolCall));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!(
                "{:?}",
                message.tool_calls.first().and_then(|f| f.as_function())
            );
            message.tool_calls.len() > 0
                && message
                    .tool_calls
                    .first()
                    .and_then(|f| f.as_function())
                    .map(|f| f.1 == "temperature")
                    .unwrap_or(false)
        }));
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_response() {
        let model = SSELanguageModel::new(APIModel::new(
            APIProvider::Anthropic,
            "claude-3-haiku-20240307",
            ANTHROPIC_API_KEY,
        ));
        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let config = LMConfigBuilder::new()
            .stream(true)
            .system_message("You are a helpful assistant.")
            .build();
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "toolu_01KjM9aTHwxL8zLQTKcj2yY8",
                "temperature",
                to_value!({"location": "Dubai", "unit": "fahrenheit"}),
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "toolu_01A8fw3xe1Rxe2eahjevvFbE",
                "temperature",
                to_value!({"location": "Dubai", "unit": "celsius"}),
            )]),
            Message::new(Role::Tool)
                .with_id("toolu_01KjM9aTHwxL8zLQTKcj2yY8")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
            Message::new(Role::Tool)
                .with_id("toolu_01A8fw3xe1Rxe2eahjevvFbE")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 30, "unit": "celsius"}),
                }]),
        ];
        let mut strm = model.run(msgs, tools, config);
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }
}
