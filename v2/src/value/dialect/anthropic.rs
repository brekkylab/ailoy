use base64::Engine;

use crate::{
    to_value,
    value::{Marshal, Message, MessageDelta, Part, PartDelta, Role, Unmarshal, Value},
};

#[derive(Clone, Debug, Default)]
pub struct AnthropicMarshal;

impl Marshal<Message> for AnthropicMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
        let mut contents = Value::array_empty();
        // In OpenAI Responses API, a message can contain only one tool call.
        let mut tool_calls = Value::array_empty();
        let mut refusal = Value::array_empty();

        // Encode each parts
        for part in &item.parts {
            match part {
                Part::TextReasoning { .. } => {
                    // ignore
                }
                Part::TextContent(s) => {
                    let value = Value::object([("type", "text"), ("text", s.as_str())]);
                    contents.as_array_mut().unwrap().push(value);
                }
                Part::ImageContent { .. } => {
                    // Get image
                    let img = part.as_image().unwrap();
                    // Write PNG string
                    let mut png_buf = Vec::new();
                    img.write_to(
                        &mut std::io::Cursor::new(&mut png_buf),
                        image::ImageFormat::Png,
                    )
                    .unwrap();
                    // base64 encoding
                    let encoded = base64::engine::general_purpose::STANDARD.encode(png_buf);
                    let value = Value::object([
                        ("type", Value::string("image")),
                        (
                            "source",
                            Value::object([
                                ("type", "base64"),
                                ("media_type", "image/png"),
                                ("data", encoded.as_str()),
                            ]),
                        ),
                    ]);
                    contents.as_array_mut().unwrap().push(value);
                }
                Part::FunctionToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let arguments_string = serde_json::to_string(arguments)
                        .map_err(|_| String::from("Invald function"))
                        .unwrap();
                    let mut value = to_value!({"type": "tool_use", "id": id.as_deref().unwrap(), "name": name, "input": arguments_string});
                    if let Some(id) = id {
                        value
                            .as_object_mut()
                            .unwrap()
                            .insert("id".into(), id.into());
                    };
                    tool_calls.as_array_mut().unwrap().push(value);
                }
                Part::TextRefusal(s) => {
                    let value = Value::object([("type", "text"), ("text", s.as_str())]);
                    refusal.as_array_mut().unwrap().push(value);
                }
            };
        }

        // Final message object with role and collected parts
        let mut rv = to_value!({"role": item.role.to_string()});
        if !contents.as_array().unwrap().is_empty() {
            rv.as_object_mut()
                .unwrap()
                .insert("content".into(), contents);
        }
        if !tool_calls.as_array().unwrap().is_empty() {
            rv.as_object_mut()
                .unwrap()
                .insert("content".into(), tool_calls);
        }
        if !refusal.as_array().unwrap().is_empty() {
            rv.as_object_mut()
                .unwrap()
                .insert("refusal".into(), refusal);
        }
        rv
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicUnmarshal;

impl Unmarshal<MessageDelta> for AnthropicUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
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
                    if let Some(r) = val.pointer_as::<String>("/message/role") {
                        // r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
                        let v = match r.as_str() {
                            "system" => Ok(Role::System),
                            "user" => Ok(Role::User),
                            "assistant" => Ok(Role::Assistant),
                            "tool" => Ok(Role::Tool),
                            other => Err(format!("Unknown role: {other}")),
                        }?;
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

                    let Some(ty) = val.pointer_as::<String>("/content_block/type") else {
                        return Err(String::from("Invalid content block type"));
                    };
                    let part_delta = match ty.as_str() {
                        "text" => PartDelta::TextContent("".to_owned()),
                        "thinking" => PartDelta::TextReasoning {
                            text: "".to_owned(),
                            signature: None,
                        },
                        "tool_use" => {
                            // r#"{"type":"content_block_start","content_block":{"type":"tool_use","id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","input":{}}}"#,
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

                            PartDelta::FunctionToolCall {
                                id,
                                name: name.clone(),
                                arguments: arguments,
                            }
                        }
                        _ => PartDelta::Null,
                    };
                    rv.parts.push(part_delta);
                }
                "content_block_delta" => {
                    let part_delta = match (
                        val.pointer_as::<String>("/delta/text"),
                        val.pointer_as::<String>("/delta/thinking"),
                        val.pointer_as::<String>("/delta/signature"),
                        val.pointer_as::<String>("/delta/partial_json"),
                    ) {
                        (Some(s), None, None, None) => PartDelta::TextContent(s.into()),
                        (None, Some(s), None, None) => PartDelta::TextReasoning {
                            text: s.into(),
                            signature: None,
                        },
                        (None, None, Some(s), None) => PartDelta::TextReasoning {
                            text: "".to_owned(),
                            signature: Some(s.to_owned()),
                        },
                        (None, Some(text), Some(sig), None) => PartDelta::TextReasoning {
                            text: text.to_owned(),
                            signature: Some(sig.to_owned()),
                        },
                        (None, None, None, Some(s)) => PartDelta::FunctionToolCall {
                            id: None,
                            name: "".to_owned(),
                            arguments: s.into(),
                        },
                        _ => PartDelta::Null,
                    };

                    rv.parts.push(part_delta);
                }
                "content_block_stop" => {
                    // r#"{"type":"content_block_stop","index":0}"#,
                    // r#"{"type":"content_block_stop","index":1}"#,
                }
                _ => {
                    return Err(String::from("Invalid stream message type"));
                }
            }
            return Ok(rv);
        }

        // not streamed below

        let root = val
            .as_object()
            .ok_or_else(|| String::from("Root should be an object"))?;

        // Parse role
        if let Some(r) = root.get("role") {
            let s = r
                .as_str()
                .ok_or_else(|| String::from("Role should be a string"))?;
            let v = match s {
                "system" => Ok(Role::System),
                "user" => Ok(Role::User),
                "assistant" => Ok(Role::Assistant),
                "tool" => Ok(Role::Tool),
                other => Err(format!("Unknown role: {other}")),
            }?;
            rv.role = Some(v);
        }

        // Parse contents
        if let Some(contents) = root.get("content")
            && !contents.is_null()
        {
            if let Some(text) = contents.as_str() {
                // Contents can be a single string
                rv.parts.push(PartDelta::TextContent(text.into()));
            } else if let Some(contents) = contents.as_array() {
                // In case of part vector
                for content in contents {
                    let Some(content) = content.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    if let Some(text) = content.get("text") {
                        let Some(text) = text.as_str() else {
                            return Err(String::from("Invalid content part"));
                        };
                        rv.parts.push(PartDelta::TextContent(text.into()));
                    } else if let Some(thinking) = content.get("thinking")
                        && let Some(signature) = content.get("signature")
                    {
                        let Some(thinking) = thinking.as_str() else {
                            return Err(String::from("Invalid thinking content"));
                        };
                        let Some(signature) = signature.as_str() else {
                            return Err(String::from("Invalid signature content"));
                        };
                        rv.parts.push(PartDelta::TextReasoning {
                            text: thinking.into(),
                            signature: Some(signature.to_owned()),
                        });
                    } else if let Some(ty) = content.get("type")
                        && ty.as_str() == Some("tool_use")
                        && let Some(id) = content.get("id")
                        && let Some(name) = content.get("name")
                        && let Some(input) = content.get("input")
                    {
                        rv.parts.push(PartDelta::FunctionToolCall {
                            id: id.as_str().and_then(|id| Some(id.to_owned())),
                            name: name.as_str().unwrap_or_default().to_owned(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        });
                    }
                    // else if let Some(refusal) = content.get("refusal")
                    //     && !refusal.is_null()
                    // {
                    //     let Some(text) = refusal.as_str() else {
                    //         return Err(String::from("Invalid refusal content"));
                    //     };
                    //     rv.parts.push(PartDelta::TextRefusal(text.into()));
                    // }
                    else {
                        return Err(String::from("Invalid part"));
                    }
                }
            } else {
                return Err(String::from("Invalid content"));
            }
        }

        Ok(rv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Delta, Marshaled, Message, Role};

    #[test]
    pub fn serialize_text() {
        let mut msg = Message::new(Role::User);
        msg.parts
            .push(Part::text_content("Explain me about Riemann hypothesis"));
        msg.parts.push(Part::text_content(
            "How cold brew is different from the normal coffee?",
        ));
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis"},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let mut msg = Message::new(Role::Assistant);
        msg.parts.push(Part::function_tool_call_with_id(
            "temperature",
            Value::object([("unit", "celsius")]),
            "funcid_123456",
        ));
        msg.parts.push(Part::function_tool_call_with_id(
            "temperature",
            Value::object([("unit", "fahrenheit")]),
            "funcid_7890ab",
        ));
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        // println!("{}", serde_json::to_string(&marshaled).unwrap());
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","content":[{"type":"tool_use","id":"funcid_123456","name":"temperature","input":"{\"unit\":\"celsius\"}"},{"type":"tool_use","id":"funcid_7890ab","name":"temperature","input":"{\"unit\":\"fahrenheit\"}"}]}"#,
        );
    }

    #[test]
    pub fn serialize_image() {
        let raw_pixels: Vec<u8> = vec![
            10, 20, 30, // First row
            40, 50, 60, // Second row
            70, 80, 90, // Third row
        ];

        let mut msg = Message::new(Role::User);
        msg.parts
            .push(Part::text_content("What you can see in this image?"));
        msg.parts.push(Part::image_content(3, 3, 1, raw_pixels));
        let marshaled = Marshaled::<_, AnthropicMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#
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
        assert_eq!(delta.parts.len(), 1);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        assert_eq!(delta.parts[0].as_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.parts.len(), 1);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        assert_eq!(delta.parts[0].as_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.parts.len(), 2);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        match delta.parts.get(0) {
            Some(PartDelta::TextReasoning {
                signature: Some(signature),
                ..
            }) => {
                assert_eq!(signature, "Ev4MCkYIBxgCKkDl5A");
            }
            Some(PartDelta::TextReasoning { .. }) => {
                panic!("PartDelta is TextReasoning, but signature is None.");
            }
            Some(_) => {
                panic!("PartDelta is not TextReasoning.");
            }
            None => {
                panic!("No PartDelta got.");
            }
        }
        assert_eq!(
            delta.parts[0].as_text().unwrap(),
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert!(delta.parts[1].is_text());
        assert_eq!(delta.parts[1].as_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.parts.len(), 2);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        assert_eq!(
            delta.parts[0].as_text().unwrap(),
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert!(delta.parts[1].is_text());
        assert_eq!(delta.parts[1].as_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.parts.len(), 1);
        assert!(delta.parts[0].is_function());
        let (id, name, args) = delta.parts[0].as_function().unwrap();
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
        assert_eq!(delta.parts.len(), 1);
        assert!(delta.parts[0].is_function());
        let (id, name, args) = delta.parts[0].as_function().unwrap();
        assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }
}
