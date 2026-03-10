#![cfg(test)]

use futures::StreamExt as _;

use crate::ToolDescBuilder;
use crate::{
    lang_model::language_model::{LangModelInferConfig, LangModelInference},
    message::{Delta, FinishReason, Message, MessageDelta, Part, Role, ToolDesc},
    to_value,
};

pub(crate) fn temperature_tool() -> ToolDesc {
    ToolDescBuilder::new("temperature")
        .description("Get current temperature")
        .parameters(to_value!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "unit": {
                    "type": "string",
                    "description": "The unit of temperature",
                    "enum": ["celsius", "fahrenheit"]
                }
            }
        }))
        .build()
}

pub fn simple_chat_msgs() -> Vec<Message> {
    vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])]
}

pub fn simple_chat_msgs_with_system() -> Vec<Message> {
    vec![
        Message::new(Role::System).with_contents([Part::text("You are a helpful assistant.")]),
        Message::new(Role::User).with_contents([Part::text("Hi what's your name?")]),
    ]
}

pub fn tool_call_msgs() -> Vec<Message> {
    vec![Message::new(Role::User).with_contents([Part::text("How much hot currently in Dubai?")])]
}

pub fn tool_call_msgs_with_system() -> Vec<Message> {
    vec![
        Message::new(Role::System).with_contents([Part::text("You are a helpful assistant.")]),
        Message::new(Role::User).with_contents([Part::text(
            "How much hot currently in Dubai? Answer in Celsius",
        )]),
    ]
}

pub async fn assert_simple_chat_streaming(model: &mut impl LangModelInference, msgs: Vec<Message>) {
    let mut assistant_msg = MessageDelta::new();
    let mut strm = model.infer_delta(msgs, Vec::new(), LangModelInferConfig::default());
    let mut finish_reason = None;
    while let Some(output_opt) = strm.next().await {
        let output = output_opt.unwrap();
        assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
        finish_reason = output.finish_reason;
    }
    assert_eq!(finish_reason, Some(FinishReason::Stop {}));
    assert!(
        assistant_msg
            .finish()
            .is_ok_and(|message| message.contents.len() > 0)
    );
}

pub async fn assert_simple_chat(model: &mut impl LangModelInference, msgs: Vec<Message>) {
    let output = model
        .infer(msgs, Vec::new(), LangModelInferConfig::default())
        .await
        .unwrap();
    assert_eq!(output.finish_reason, FinishReason::Stop {});
    assert!(output.message.contents.len() > 0);
}

pub async fn assert_tool_call_streaming(model: &mut impl LangModelInference, msgs: Vec<Message>) {
    let tools = vec![temperature_tool()];
    let mut strm = model.infer_delta(msgs, tools, LangModelInferConfig::default());
    let mut assistant_msg = MessageDelta::default();
    let mut finish_reason = None;
    while let Some(output_opt) = strm.next().await {
        let output = output_opt.unwrap();
        assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
        finish_reason = output.finish_reason;
    }
    assert_eq!(finish_reason, Some(FinishReason::ToolCall {}));
    assert!(assistant_msg.finish().is_ok_and(|message| {
        if let Some(tool_calls) = message.tool_calls {
            tool_calls
                .first()
                .and_then(|f| f.as_function())
                .map(|f| f.1 == "temperature")
                .unwrap_or(false)
        } else {
            false
        }
    }));
}

pub async fn assert_tool_call(model: &mut impl LangModelInference, msgs: Vec<Message>) {
    let tools = vec![temperature_tool()];
    let output = model
        .infer(msgs, tools, LangModelInferConfig::default())
        .await
        .unwrap();
    assert_eq!(output.finish_reason, FinishReason::ToolCall {});
    let tool_calls = output.message.tool_calls.unwrap();
    assert!(
        tool_calls
            .first()
            .and_then(|f| f.as_function())
            .map(|f| f.1 == "temperature")
            .unwrap_or(false)
    );
}

pub async fn assert_tool_response_streaming(
    model: &mut impl LangModelInference,
    msgs: Vec<Message>,
) {
    let tools = vec![temperature_tool()];
    let mut strm = model.infer_delta(msgs, tools, LangModelInferConfig::default());
    let mut assistant_msg = MessageDelta::default();
    let mut finish_reason = None;
    while let Some(output_opt) = strm.next().await {
        let output = output_opt.unwrap();
        assistant_msg = assistant_msg.accumulate(output.delta).unwrap();
        finish_reason = output.finish_reason;
    }
    assert_eq!(finish_reason, Some(FinishReason::Stop {}));
    assert!(
        assistant_msg
            .finish()
            .is_ok_and(|message| message.contents.len() > 0)
    );
}

pub async fn assert_tool_response(model: &mut impl LangModelInference, msgs: Vec<Message>) {
    let tools = vec![temperature_tool()];
    let output = model
        .infer(msgs, tools, LangModelInferConfig::default())
        .await
        .unwrap();
    assert_eq!(output.finish_reason, FinishReason::Stop {});
    assert!(output.message.contents.len() > 0);
}
