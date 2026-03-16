use crate::{
    agent::rt::lang_model::LangModelRequest,
    datatype::Value,
    message::{
        FinishReason, Marshal, Message, MessageDelta, MessageDeltaOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, ToolDesc, Unmarshal,
    },
    to_value,
};

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Vec<Value> {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => {
                let r#type = if msg.role == Role::Assistant {
                    "output_text"
                } else {
                    "input_text"
                };
                to_value!({"type": r#type, "text": text})
            }
            Part::Function {
                id,
                function: PartFunction { name, arguments },
            } => {
                let arguments_string = serde_json::to_string(arguments).unwrap();
                if let Some(id) = id {
                    to_value!({"type": "function_call", "call_id": id, "name": name, "arguments": arguments_string})
                } else {
                    to_value!({"type": "function_call", "name": name, "arguments": arguments_string})
                }
            }
            Part::Value { value } => {
                to_value!(serde_json::to_string(value).unwrap())
            }
            Part::Image { image } => {
                let url = match image {
                    PartImage::Embedded { mime_type, data } => {
                        format!("data:{};base64,{}", mime_type, data.base64())
                    }
                    PartImage::Url { url } => url.clone(),
                };
                to_value!({"type": "input_image", "image_url": {"url": url}})
            }
        }
    };

    if msg.role == Role::Tool {
        return vec![to_value!(
            {
                "type": "function_call_output",
                "call_id": msg.id.clone().expect("Tool call id must exist."),
                "output": part_to_value(&msg.contents[0])
            }
        )];
    }

    let mut rv = Vec::<Value>::new();
    if let Some(thinking) = &msg.thinking
        && !thinking.is_empty()
        && include_thinking
    {
        rv.push(
            to_value!({"type": "reasoning", "summary": [{"type": "summary_text", "text": thinking}]}),
        );
    }
    if !msg.contents.is_empty() {
        rv.push(to_value!({"role": msg.role.to_string(), "content": msg.contents.iter().map(part_to_value).collect::<Vec<_>>()}));
    }
    rv.extend(
        msg.tool_calls
            .clone()
            .unwrap_or(vec![])
            .iter()
            .map(part_to_value),
    );
    rv
}

fn marshal_messages(msgs: &[Message]) -> Value {
    let last_user_index = msgs
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or_else(|| msgs.len());
    Value::Array(
        msgs.iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .flat_map(|(i, msg)| marshal_message(msg, i > last_user_index))
            .collect::<Vec<_>>(),
    )
}

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        to_value!(marshal_message(msg, true))
    }
}

impl Marshal<ToolDesc> for OpenAIMarshal {
    fn marshal(&mut self, item: &ToolDesc) -> Value {
        if let Some(desc) = &item.description {
            to_value!({
                "type": "function",
                "name": &item.name,
                "description": desc,
                "parameters": item.parameters.clone()
            })
        } else {
            to_value!({
                "type": "function",
                "name": &item.name,
                "parameters": item.parameters.clone()
            })
        }
    }
}

impl<'a> Marshal<LangModelRequest<'a>> for OpenAIMarshal {
    fn marshal(&mut self, req: &LangModelRequest) -> Value {
        // Extract system instruction from system message if present
        let instructions = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.contents.first())
            .and_then(|p| {
                if let Part::Text { text } = p {
                    Some(Value::from(text.as_str()))
                } else {
                    None
                }
            });

        let input = marshal_messages(req.messages);

        let tools = if !req.tools.is_empty() {
            Value::Array(req.tools.iter().map(|t| self.marshal(t)).collect())
        } else {
            Value::Null
        };

        let url = req.url.to_string();

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = req.api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("Authorization".into(), format!("Bearer {}", api_key).into());
        }

        let mut body = to_value!({
            "model": req.model,
            "input": input,
        });
        if let Some(instructions) = instructions {
            body.as_object_mut()
                .unwrap()
                .insert("instructions".into(), instructions);
        }
        if !tools.is_null() {
            body.as_object_mut()
                .unwrap()
                .insert("tool_choice".into(), to_value!("auto"));
            body.as_object_mut().unwrap().insert("tools".into(), tools);
        }

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIUnmarshal;

impl Unmarshal<MessageDeltaOutput> for OpenAIUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let root = val
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Root should be an object"))?;

        // Parse finish reason from status
        let status = root
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed");

        let mut finish_reason = match status {
            "completed" => Some(FinishReason::Stop {}),
            "incomplete" => {
                let reason = root
                    .get("incomplete_details")
                    .and_then(|d| d.pointer("/reason"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Some(match reason {
                    "max_output_tokens" => FinishReason::Length {},
                    "content_filter" => FinishReason::Refusal {
                        reason: "Model output violated OpenAI's safety policy.".to_owned(),
                    },
                    other => FinishReason::Refusal {
                        reason: format!("reason: {}", other),
                    },
                })
            }
            _ => None,
        };

        // Parse output items
        let mut delta = MessageDelta::default();

        if let Some(output) = root.get("output")
            && let Some(items) = output.as_array()
        {
            for item in items {
                let Some(item_obj) = item.as_object() else {
                    continue;
                };
                let ty = item_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "message" => {
                        // Parse role
                        if delta.role.is_none() {
                            let role = item_obj
                                .get("role")
                                .and_then(|v| v.as_str())
                                .map(|s| match s {
                                    "assistant" => Role::Assistant,
                                    "user" => Role::User,
                                    _ => Role::Assistant,
                                })
                                .unwrap_or(Role::Assistant);
                            delta.role = Some(role);
                        }
                        // Parse content parts
                        if let Some(content) = item_obj.get("content")
                            && let Some(parts) = content.as_array()
                        {
                            for part in parts {
                                let part_ty =
                                    part.pointer("/type").and_then(|v| v.as_str()).unwrap_or("");
                                if part_ty == "output_text" || part_ty == "text" {
                                    if let Some(text) =
                                        part.pointer("/text").and_then(|v| v.as_str())
                                    {
                                        delta.contents.push(PartDelta::Text {
                                            text: text.to_owned(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    "function_call" => {
                        if delta.role.is_none() {
                            delta.role = Some(Role::Assistant);
                        }
                        let id = item_obj
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_owned());
                        let name = item_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let arguments = item_obj
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        delta.tool_calls.push(PartDelta::Function {
                            id,
                            function: PartDeltaFunction::WithStringArgs { name, arguments },
                        });
                    }
                    "reasoning" => {
                        // Parse summary text as thinking
                        if let Some(summary) = item_obj.get("summary")
                            && let Some(parts) = summary.as_array()
                        {
                            for part in parts {
                                if let Some(text) = part.pointer("/text").and_then(|v| v.as_str()) {
                                    delta.thinking =
                                        Some(delta.thinking.unwrap_or_default() + text);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Adjust finish reason for tool calls
        if !delta.tool_calls.is_empty()
            && finish_reason
                .clone()
                .is_some_and(|r| matches!(r, FinishReason::Stop {}))
        {
            finish_reason = Some(FinishReason::ToolCall {});
        }

        Ok(MessageDeltaOutput {
            delta,
            finish_reason,
        })
    }
}

// #[cfg(test)]
// mod dialect_tests {
//     use super::*;
//     use crate::{
//         datatype::Bytes,
//         message::{Marshaled, Message, Role},
//     };

//     #[test]
//     pub fn serialize_text() {
//         let msg = Message::new(Role::User)
//             .with_contents([Part::text("Explain me about Riemann hypothesis.")]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","content":[{"type":"input_text","text":"Explain me about Riemann hypothesis."}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_messages_with_thinkings() {
//         let msgs = vec![
//             Message::new(Role::User).with_contents([Part::text("Hello there.")]),
//             Message::new(Role::Assistant)
//                 .with_thinking_signature("This is thinking text would be vanished.", "")
//                 .with_contents([Part::text("I'm fine, thank you. And you?")]),
//             Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
//             Message::new(Role::Assistant)
//                 .with_thinking_signature(
//                     "This is thinking text would be remaining.",
//                     "Ev4MCkYIBxgCKkDl5A",
//                 )
//                 .with_contents([Part::text("Is there anything I can help with?")]),
//         ];
//         // Use marshal_messages directly to test position-aware thinking inclusion.
//         let marshaled = marshal_messages(&msgs);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","content":[{"type":"input_text","text":"Hello there."}]},{"role":"assistant","content":[{"type":"output_text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"input_text","text":"I'm okay."}]},{"type":"reasoning","summary":[{"type":"summary_text","text":"This is thinking text would be remaining."}]},{"role":"assistant","content":[{"type":"output_text","text":"Is there anything I can help with?"}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_function() {
//         let msg = Message::new(Role::Assistant).with_tool_calls([
//             Part::function_with_id(
//                 "funcid_123456",
//                 "temperature",
//                 Value::object([("unit", "celsius")]),
//             ),
//             Part::function_with_id(
//                 "funcid_7890ab",
//                 "temperature",
//                 Value::object([("unit", "fahrenheit")]),
//             ),
//         ]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"type":"function_call","call_id":"funcid_123456","name":"temperature","arguments":"{\"unit\":\"celsius\"}"},{"type":"function_call","call_id":"funcid_7890ab","name":"temperature","arguments":"{\"unit\":\"fahrenheit\"}"}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_tool_response() {
//         let msgs = vec![
//             Message::new(Role::Tool)
//                 .with_id("funcid_123456")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 30, "unit": "celsius"}),
//                 }]),
//             Message::new(Role::Tool)
//                 .with_id("funcid_7890ab")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
//                 }]),
//         ];
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msgs);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"type":"function_call_output","call_id":"funcid_123456","output":"{\"temperature\":30,\"unit\":\"celsius\"}"},{"type":"function_call_output","call_id":"funcid_7890ab","output":"{\"temperature\":86,\"unit\":\"fahrenheit\"}"}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_image() {
//         use base64::prelude::*;

//         let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII=";
//         let png_bytes = BASE64_STANDARD.decode(png_base64).unwrap();
//         let msg = Message::new(Role::User).with_contents([
//             Part::text("What you can see in this image?"),
//             Part::image_embedded("image/png".to_owned(), Bytes::from(png_bytes)).unwrap(),
//         ]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","content":[{"type":"input_text","text":"What you can see in this image?"},{"type":"input_image","image_url":{"url":"data:image/png;base64,"#.to_owned()
//                 + png_base64
//                 + r#""}}]}]"#,
//         );
//     }

//     #[test]
//     pub fn deserialize_text() {
//         let input = r#"{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world!"}]}]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Stop {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.role, Some(Role::Assistant));
//         assert_eq!(delta.contents.len(), 1);
//         let content = delta.contents.pop().unwrap();
//         assert_eq!(content.to_text().unwrap(), "Hello world!");
//     }

//     #[test]
//     pub fn deserialize_text_with_reasoning() {
//         let input = r#"{"status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"**Answering a simple question**\n\nUser is saying hello."}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world!"}]}]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Stop {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.role, Some(Role::Assistant));
//         assert_eq!(
//             delta.thinking,
//             Some("**Answering a simple question**\n\nUser is saying hello.".into())
//         );
//         assert_eq!(delta.contents.len(), 1);
//         let content = delta.contents.pop().unwrap();
//         assert_eq!(content.to_text().unwrap(), "Hello world!");
//     }

//     #[test]
//     pub fn deserialize_tool_call() {
//         let input = r#"{"status":"completed","output":[{"type":"function_call","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","arguments":"{\"location\":\"Paris, France\"}"}]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::ToolCall {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.tool_calls.len(), 1);
//         let tool_call = delta.tool_calls.pop().unwrap();
//         let (id, name, args) = tool_call.to_function().unwrap();
//         assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
//         assert_eq!(name, "get_weather");
//         assert_eq!(args, "{\"location\":\"Paris, France\"}");
//     }

//     #[test]
//     pub fn deserialize_incomplete() {
//         let input =
//             r#"{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Length {}));
//     }
// }
