use anyhow::{Context, bail};

use crate::{
    model::{ServerEvent, ThinkEffort, api::RequestConfig},
    to_value,
    value::{
        FinishReason, Marshal, Marshaled, Message, MessageDelta, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, ToolDesc, Unmarshal, Unmarshaled, Value,
    },
};

#[derive(Clone, Debug, Default)]
struct AnthropicMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => to_value!({"type": "text", "text": text}),
            Part::Function {
                id,
                function: PartFunction { name, arguments },
            } => {
                let mut value =
                    to_value!({"type": "tool_use", "name": name, "input": arguments.clone()});
                if let Some(id) = id {
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("id".into(), id.into());
                };
                value
            }
            Part::Value { value } => {
                to_value!(serde_json::to_string(&value).unwrap())
            }
            Part::Image { image } => {
                let b64 = match image {
                    PartImage::Binary { .. } => image.base64().unwrap(),
                    PartImage::Url { .. } => panic!("Claude does not support image url inputs"),
                };
                to_value!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    }
                })
            }
        }
    };

    if msg.role == Role::Tool {
        return to_value!(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.id.clone().expect("Tool call id must exist."),
                        "content": part_to_value(&msg.contents[0])
                    }
                ]
            }
        );
    }

    // Collecting contents
    let mut contents = Vec::<Value>::new();
    if let Some(thinking) = &msg.thinking
        && !thinking.is_empty()
        && include_thinking
    {
        let mut part = to_value!({"type": "thinking", "thinking": thinking});
        if let Some(sig) = &msg.signature {
            part.as_object_mut()
                .unwrap()
                .insert("signature".into(), sig.into());
        }
        contents.push(part);
    }
    contents.extend(msg.contents.iter().map(part_to_value));
    contents.extend(
        msg.tool_calls
            .clone()
            .unwrap_or(vec![])
            .iter()
            .map(part_to_value),
    );

    // Final message object with role and collected parts
    to_value!({"role": msg.role.to_string(), "content": contents})
}

impl Marshal<Message> for AnthropicMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        marshal_message(msg, true)
    }
}

impl Marshal<Vec<Message>> for AnthropicMarshal {
    fn marshal(&mut self, msgs: &Vec<Message>) -> Value {
        let last_user_index = msgs
            .iter()
            .rposition(|m| m.role == Role::User)
            .unwrap_or_else(|| msgs.len());
        Value::array(
            msgs.iter()
                .enumerate()
                .map(|(i, msg)| marshal_message(msg, i > last_user_index))
                .collect::<Vec<_>>(),
        )
    }
}

impl Marshal<ToolDesc> for AnthropicMarshal {
    fn marshal(&mut self, item: &ToolDesc) -> Value {
        if let Some(desc) = &item.description {
            to_value!({
                "name": &item.name,
                "description": desc,
                "input_schema": item.parameters.clone()
            })
        } else {
            to_value!({
                "name": &item.name,
                "input_schema": item.parameters.clone()
            })
        }
    }
}

enum AnthropicModelType {
    Opus,
    Sonnet,
    Haiku,
}

impl Marshal<RequestConfig> for AnthropicMarshal {
    fn marshal(&mut self, config: &RequestConfig) -> Value {
        let Some(model) = &config.model else {
            panic!("Cannot marshal `Config` without `model`.");
        };

        let model_type = if model.contains("opus") {
            AnthropicModelType::Opus
        } else if model.contains("sonnet") {
            AnthropicModelType::Sonnet
        } else if model.contains("haiku") {
            AnthropicModelType::Haiku
        } else {
            panic!("Unsupported model.");
        };

        let is_reasoning_model = match model_type {
            AnthropicModelType::Opus | AnthropicModelType::Sonnet => true,
            AnthropicModelType::Haiku => false,
        };

        let budget_tokens = match (&config.think_effort, &model_type) {
            (_, AnthropicModelType::Haiku) => 0,
            (ThinkEffort::Disable, _) => 0,
            (ThinkEffort::Enable, _) => 8192,
            (ThinkEffort::Low, _) => 1024,
            (ThinkEffort::Medium, _) => 8192,
            (ThinkEffort::High, _) => 24576,
        };

        let reasoning = if budget_tokens != 0 {
            to_value!({"type": "enabled", "budget_tokens": budget_tokens as i64})
        } else {
            Value::Null
        };

        let system = if let Some(system_message) = &config.system_message {
            to_value!(system_message)
        } else {
            Value::Null
        };

        let max_tokens = if let Some(max_tokens) = &config.max_tokens {
            Value::integer(*max_tokens as i64)
        } else {
            Value::integer(match &model_type {
                AnthropicModelType::Opus => 32000,
                AnthropicModelType::Sonnet => 64000,
                AnthropicModelType::Haiku => {
                    if model.starts_with("claude-3-5-haiku") {
                        8192
                    } else if model.starts_with("claude-3-haiku") {
                        4096
                    } else {
                        64000
                    }
                }
            })
        };

        let temperature = if let Some(temperature) = &config.temperature
            && !is_reasoning_model
        {
            to_value!(*temperature)
        } else {
            Value::Null
        };

        let top_p = if let Some(top_p) = &config.top_p
            && !is_reasoning_model
        {
            to_value!(*top_p)
        } else {
            Value::Null
        };

        let stream = config.stream;

        let mut body = to_value!({
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "reasoning": reasoning,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        });
        body.as_object_mut()
            .unwrap()
            .retain(|_key, value| !value.is_null());
        body
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicUnmarshal;

impl Unmarshal<MessageDelta> for AnthropicUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDelta> {
        const STREAM_TYPES: &[&str] = &[
            "ping",
            "message_start",
            "message_delta",
            "message_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
        ];

        let mut rv = MessageDelta::default();
        let ty = if let Some(ty) = val.pointer("/type") {
            ty.as_str().unwrap_or_default()
        } else {
            ""
        };
        let streamed = STREAM_TYPES.contains(&ty);

        if streamed {
            match ty {
                "ping" => {
                    // r#"{"type":"ping"}"#,
                }
                "message_start" => {
                    // r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
                    if let Some(r) = val.pointer_as::<String>("/message/role") {
                        let v: Role = match r.as_str() {
                            "system" => Role::System,
                            "user" => Role::User,
                            "assistant" => Role::Assistant,
                            "tool" => Role::Tool,
                            other => bail!("Unknown role: {other}"),
                        };
                        rv.role = Some(v);
                    }
                }
                "message_delta" => {
                    // r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
                }
                "message_stop" => {
                    // r#"{"type":"message_stop"}"#,
                }
                "content_block_start" => {
                    // r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
                    // r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                    // r#"{"type":"content_block_start","content_block":{"type":"tool_use","id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","input":{}}}"#,
                    let Some(ty) = val.pointer_as::<String>("/content_block/type") else {
                        bail!("Invalid content block type");
                    };
                    match ty.as_str() {
                        "text" => {
                            rv.contents.push(PartDelta::Text {
                                text: String::default(),
                            });
                        }
                        "thinking" => {}
                        "tool_use" => {
                            let id = val.pointer_as::<String>("/content_block/id").cloned();
                            let name = val
                                .pointer_as::<String>("/content_block/name")
                                .cloned()
                                .unwrap_or_default();
                            let arguments = if let Some(input) =
                                val.pointer_as::<String>("/content_block/input")
                            {
                                input.clone()
                            } else {
                                "".to_owned()
                            };
                            rv.tool_calls.push(PartDelta::Function {
                                id,
                                function: PartDeltaFunction::WithStringArgs { name, arguments },
                            });
                        }
                        _ => {}
                    };
                }
                "content_block_delta" => {
                    if let Some(text) = val.pointer_as::<String>("/delta/text") {
                        rv.contents.push(PartDelta::Text { text: text.into() });
                    }
                    if let Some(text) = val.pointer_as::<String>("/delta/thinking") {
                        rv.thinking = Some(rv.thinking.unwrap_or("".into()) + text);
                    }
                    if let Some(text) = val.pointer_as::<String>("/delta/signature") {
                        rv.signature = Some(text.to_owned());
                    }
                    if let Some(args) = val.pointer_as::<String>("/delta/partial_json") {
                        rv.tool_calls.push(PartDelta::Function {
                            id: None,
                            function: PartDeltaFunction::WithStringArgs {
                                name: String::new(),
                                arguments: args.to_owned(),
                            },
                        });
                    }
                }
                "content_block_stop" => {
                    // r#"{"type":"content_block_stop","index":0}"#,
                    // r#"{"type":"content_block_stop","index":1}"#,
                }
                _ => {
                    bail!("Invalid stream message type");
                }
            }
            return Ok(rv);
        }

        // not streamed below

        let root = val.as_object().context("Root should be an object")?;

        // Parse role
        if let Some(r) = root.get("role") {
            let s = r.as_str().context("Role should be a string")?;
            let v = match s {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                other => bail!("Unknown role: {other}"),
            };
            rv.role = Some(v);
        }

        // Parse contents
        if let Some(contents) = root.get("content")
            && !contents.is_null()
        {
            if let Some(text) = contents.as_str() {
                // Contents can be a single string
                rv.contents.push(PartDelta::Text { text: text.into() });
            } else if let Some(contents) = contents.as_array() {
                // In case of part vector
                for content in contents {
                    let Some(content) = content.as_object() else {
                        bail!("Invalid part");
                    };
                    if let Some(text) = content.get("text") {
                        let Some(text) = text.as_str() else {
                            bail!("Invalid content part");
                        };
                        rv.contents.push(PartDelta::Text { text: text.into() });
                    } else if let Some(thinking) = content.get("thinking")
                        && let Some(signature) = content.get("signature")
                    {
                        let Some(thinking) = thinking.as_str() else {
                            bail!("Invalid thinking content");
                        };
                        let Some(signature) = signature.as_str() else {
                            bail!("Invalid signature content");
                        };
                        rv.signature = Some(signature.to_owned());
                        rv.thinking = Some(thinking.into());
                    } else if let Some(ty) = content.get("type")
                        && ty.as_str() == Some("tool_use")
                        && let Some(id) = content.get("id")
                        && let Some(name) = content.get("name")
                        && let Some(input) = content.get("input")
                    {
                        rv.tool_calls.push(PartDelta::Function {
                            id: id.as_str().and_then(|id| Some(id.to_owned())),
                            function: PartDeltaFunction::WithStringArgs {
                                name: name.as_str().unwrap_or_default().to_owned(),
                                arguments: serde_json::to_string(input).unwrap_or_default(),
                            },
                        });
                    } else {
                        bail!("Invalid part");
                    }
                }
            } else {
                bail!("Invalid content");
            }
        }

        Ok(rv)
    }
}

pub(super) fn make_request(
    url: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: RequestConfig,
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

    let builder = reqwest::Client::new()
        .request(reqwest::Method::POST, url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string());

    #[cfg(target_arch = "wasm32")]
    let builder = builder.header("anthropic-dangerous-direct-browser-access", "true");

    builder
}

pub(crate) fn handle_event(evt: ServerEvent) -> MessageOutput {
    let Ok(val) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };

    let finish_reason = val
        .pointer("/delta/stop_reason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "end_turn" => FinishReason::Stop {},
            "pause_turn" => FinishReason::Stop {}, // consider same as "end_turn"
            "max_tokens" => FinishReason::Length {},
            "tool_use" => FinishReason::ToolCall {},
            "refusal" => FinishReason::Refusal {
                reason: "Model output violated Anthropic's safety policy.".to_owned(),
            },
            reason => FinishReason::Refusal {
                reason: format!("reason: {}", reason),
            },
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal { .. }) => MessageDelta::default(),
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
mod dialect_tests {
    use super::*;
    use crate::value::{Delta, Marshaled, Message, Role};

    #[test]
    pub fn serialize_text() {
        let msg = Message::new(Role::User).with_contents([
            Part::text("Explain me about Riemann hypothesis."),
            Part::text("How cold brew is different from the normal coffee?"),
        ]);
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis."},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_messages_with_reasonings() {
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Hello there."), Part::text("How are you?")]),
            Message::new(Role::Assistant)
                .with_thinking_signature("This is reasoning text would be vanished.", "")
                .with_contents([Part::text("I'm fine, thank you. And you?")]),
            Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
            Message::new(Role::Assistant)
                .with_thinking_signature(
                    "This is reasoning text would be remaining.",
                    "Ev4MCkYIBxgCKkDl5A",
                )
                .with_contents([Part::text("Is there anything I can help with?")]),
        ];
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"text","text":"Hello there."},{"type":"text","text":"How are you?"}]},{"role":"assistant","content":[{"type":"text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"text","text":"I'm okay."}]},{"role":"assistant","content":[{"type":"thinking","thinking":"This is reasoning text would be remaining.","signature":"Ev4MCkYIBxgCKkDl5A"},{"type":"text","text":"Is there anything I can help with?"}]}]"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let msg = Message::new(Role::Assistant).with_tool_calls([
            Part::function_with_id(
                "funcid_123456",
                "temperature",
                Value::object([("unit", "celsius")]),
            ),
            Part::function_with_id(
                "funcid_7890ab",
                "temperature",
                Value::object([("unit", "fahrenheit")]),
            ),
        ]);
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","content":[{"type":"tool_use","name":"temperature","input":{"unit":"celsius"},"id":"funcid_123456"},{"type":"tool_use","name":"temperature","input":{"unit":"fahrenheit"},"id":"funcid_7890ab"}]}"#,
        );
    }

    #[test]
    pub fn serialize_tool_response() {
        let msgs = vec![
            Message::new(Role::Tool)
                .with_id("funcid_123456")
                .with_contents(vec![Part::Value {
                    value: to_value!({"temperature": 30, "unit": "celsius"}),
                }]),
            Message::new(Role::Tool)
                .with_id("funcid_7890ab")
                .with_contents(vec![Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
        ];
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"tool_result","tool_use_id":"funcid_123456","content":"{\"temperature\":30,\"unit\":\"celsius\"}"}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"funcid_7890ab","content":"{\"temperature\":86,\"unit\":\"fahrenheit\"}"}]}]"#
        );
    }

    #[test]
    pub fn serialize_image() {
        let raw_pixels: Vec<u8> = vec![
            10, 20, 30, // First row
            40, 50, 60, // Second row
            70, 80, 90, // Third row
        ];
        let msg = Message::new(Role::User).with_contents([
            Part::text("What you can see in this image?"),
            Part::image_binary(3, 3, "grayscale", raw_pixels).unwrap(),
        ]);
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#
        );
    }

    #[test]
    pub fn serialize_config() {
        let config = RequestConfig {
            model: Some("claude-sonnet-4-5".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"model":"claude-sonnet-4-5","max_tokens":1024,"system":"You are a helpful assistant.","reasoning":{"type":"enabled","budget_tokens":8192},"stream":true}"#
        );

        let config = RequestConfig {
            model: Some("claude-3-haiku-20240307".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"model":"claude-3-haiku-20240307","max_tokens":1024,"system":"You are a helpful assistant.","stream":true,"temperature":0.6,"top_p":0.9}"#
        );
    }

    #[test]
    pub fn deserialize_text() {
        let inputs = [
            r#"{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role": "assistant"}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_text_stream() {
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world!"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_text_with_reasoning() {
        let inputs = [
            r#"{"role": "assistant","content":[{"type": "thinking","thinking":"**Answering a simple question**\n\nUser is saying hello.","signature":"Ev4MCkYIBxgCKkDl5A"},{"type":"text","text":"Hello world!"}]}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.thinking,
            Some("**Answering a simple question**\n\nUser is saying hello.".into())
        );
        assert_eq!(delta.signature.unwrap(), "Ev4MCkYIBxgCKkDl5A");
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_text_with_reasoning_stream() {
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"**Answering a simple question**\n\n"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"User is saying hello."}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"Ev4MCkYIBxgCKkDl5A"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world!"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.thinking,
            Some("**Answering a simple question**\n\nUser is saying hello.".into())
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_tool_call() {
        let inputs = [
            r#"{"role": "assistant","content":[{"type":"tool_use","id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","input":{"location":"Paris, France"}}]}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.tool_calls.len(), 1);
        let tool_call = delta.tool_calls.pop().unwrap();
        let (id, name, args) = tool_call.to_function().unwrap();
        assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }

    #[test]
    pub fn deserialize_tool_call_stream() {
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","content_block":{"type":"tool_use","id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","input":{}}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\""}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"location"}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\":\""}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"Paris"}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":","}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":" France"}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\"}"}}"#,
            r#"{"type":"content_block_stop"}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut u = AnthropicUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.tool_calls.len(), 1);
        let tool_call = delta.tool_calls.pop().unwrap();
        let (id, name, args) = tool_call.to_function().unwrap();
        assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }
}

#[cfg(test)]
mod api_tests {
    use std::sync::LazyLock;

    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;
    use crate::{
        debug,
        model::{
            APISpecification, InferenceConfig, LangModelInference as _, api::StreamAPILangModel,
        },
        to_value,
        value::{Delta, Part, Role, ToolDescBuilder},
    };

    static ANTHROPIC_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("ANTHROPIC_API_KEY")
            .expect("Environment variable 'ANTHROPIC_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn infer_simple_chat() {
        let mut model = StreamAPILangModel::new(
            APISpecification::Claude,
            "claude-3-haiku-20240307",
            *ANTHROPIC_API_KEY,
        );

        let msgs =
            vec![Message::new(Role::User).with_contents([Part::text("Hi what's your name?")])];
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.infer(msgs, Vec::new(), Vec::new(), InferenceConfig::default());
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }

    #[multi_platform_test]
    async fn infer_tool_call() {
        use crate::model::InferenceConfig;

        let mut model = StreamAPILangModel::new(
            APISpecification::Claude,
            "claude-3-haiku-20240307",
            *ANTHROPIC_API_KEY,
        );
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
        let mut strm = model.infer(msgs, tools, Vec::new(), InferenceConfig::default());
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::ToolCall {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            if let Some(tool_calls) = message.tool_calls {
                debug!("{:?}", tool_calls.first().and_then(|f| f.as_function()));
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

    #[multi_platform_test]
    async fn infer_tool_response() {
        use crate::model::InferenceConfig;

        let mut model = StreamAPILangModel::new(
            APISpecification::Claude,
            "claude-3-haiku-20240307",
            *ANTHROPIC_API_KEY,
        );
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
        let mut strm = model.infer(msgs, tools, Vec::new(), InferenceConfig::default());
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop {}));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }
}
