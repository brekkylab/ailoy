use base64::Engine;

use crate::{
    to_value,
    value::{Marshal, Message, MessageDelta, Part, PartDelta, Role, Unmarshal, Value},
};

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
        let mut contents = Value::array_empty();
        // In OpenAI Responses API, a message can contain only one tool call.
        let mut tool_call = Value::object_empty();
        let mut refusal = Value::array_empty();

        // Encode each parts
        for part in &item.parts {
            match part {
                Part::TextReasoning { .. } => {
                    // ignore
                }
                Part::TextContent(s) => {
                    let value = Value::object([("type", "input_text"), ("text", s.as_str())]);
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
                    let value = to_value!({"type": "input_image","image_url": {"url": format!("data:image/png;base64,{}", encoded)}});
                    contents.as_array_mut().unwrap().push(value);
                }
                Part::FunctionToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let arguments_str = serde_json::to_string(arguments)
                        .map_err(|_| String::from("Invald function"))
                        .unwrap();
                    tool_call = Value::object([
                        ("type", Value::string("function_call")),
                        ("call_id", Value::string(id.as_deref().unwrap())),
                        ("name", Value::string(name)),
                        ("arguments", Value::string(arguments_str)),
                    ]);
                }
                Part::TextRefusal(s) => {
                    let value = Value::object([("type", "text"), ("text", s.as_str())]);
                    refusal.as_array_mut().unwrap().push(value);
                }
            };
        }

        // Encode whole message
        let mut rv = Value::object_empty();

        if !tool_call.as_object().unwrap().is_empty() {
            rv = tool_call;
            // rv.as_object_mut().unwrap().extend(tool_call.into());
        } else {
            rv.as_object_mut()
                .unwrap()
                .insert("role".into(), item.role.to_string().into());

            if !contents.as_array().unwrap().is_empty() {
                rv.as_object_mut()
                    .unwrap()
                    .insert("content".into(), contents);
            }
            if !refusal.as_array().unwrap().is_empty() {
                rv.as_object_mut()
                    .unwrap()
                    .insert("refusal".into(), refusal);
            }
        }
        rv
    }
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIUnmarshal;

impl Unmarshal<MessageDelta> for OpenAIUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
        const STREAM_TYPES: &[&str] = &[
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ];

        let mut rv = MessageDelta::default();
        let ty = val
            .pointer("/type")
            .unwrap()
            .as_str()
            .ok_or_else(|| String::from("Stream message type should be a string"))?;
        let streamed = STREAM_TYPES.contains(&ty);

        if streamed {
            match ty {
                "response.output_item.added" => {
                    if let Some(r) = val.pointer_as::<String>("/item/role") {
                        // message with role
                        // r#"{"type":"response.output_item.added","item":{"type":"message","content":[],"role":"assistant"}}"#,
                        let v = match r.as_str() {
                            "system" => Ok(Role::System),
                            "user" => Ok(Role::User),
                            "assistant" => Ok(Role::Assistant),
                            "tool" => Ok(Role::Tool),
                            other => Err(format!("Unknown role: {other}")),
                        }?;
                        rv.role = Some(v);
                    } else if let Some(ty) = val.pointer_as::<String>("/item/type")
                        && ty.as_str() == "function_call"
                        && let call_id = val.pointer_as::<String>("/item/call_id")
                        && let Some(name) = val.pointer_as::<String>("/item/name")
                        && let Some(arguments) = val.pointer_as::<String>("/item/arguments")
                    {
                        // tool call message
                        // r#"{"type":"response.output_item.added","item":{"type":"function_call","arguments":"","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather"}}"#
                        rv.role = Some(Role::Assistant);
                        rv.parts.push(PartDelta::FunctionToolCall {
                            id: call_id.cloned(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        });
                    }
                }
                "response.output_item.done" => {
                    // r#"{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role":"assistant"}}"#,
                }
                "response.content_part.added" => {
                    // r#"{"type":"response.content_part.added","part":{"type":"output_text","text":""}}"#,
                }
                "response.content_part.done" => {
                    // r#"{"type":"response.content_part.done","part":{"type":"output_text","text":"Hello world!"}}"#,
                }
                "response.output_text.delta" => {
                    // r#"{"type":"response.output_text.delta","delta":"Hello"}"#,
                    // r#"{"type":"response.output_text.delta","delta":" world!"}"#,
                    let s = val
                        .pointer_as::<String>("/delta")
                        .ok_or_else(|| String::from("Output text delta should be a string"))?;
                    rv.parts.push(PartDelta::TextContent(s.into()));
                }
                "response.output_text.done" => {
                    // r#"{"type":"response.output_text.done","text":"Hello world!"}"#,
                }
                "response.reasoning_summary_part.added" => {
                    // r#"{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}"#
                }
                "response.reasoning_summary_part.done" => {
                    // r#"{"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"**Responding to a greeting**\n\nThe user just said, \"Hello!\" So, it seems I need to engage. I'll greet them back and offer help since they're looking to chat. I could say something like, \"Hello! How can I assist you today?\" That feels friendly and open. They didn't ask a specific question, so this approach will work well for starting a conversation. Let's see where it goes from there!"}}"#
                }
                "response.reasoning_summary_text.delta" => {
                    // r#"{"type":"response.reasoning_summary_text.delta","delta":"**Responding to a greeting**\n\nThe user just said, \"Hello!\" So, it seems I need to engage. I'll greet them back and offer help since they're looking to chat. I could say something like, \"Hello! How can I assist you today?\" That feels friendly and open. They didn't ask a specific question, so this approach will work well for starting a conversation. Let's see where it goes from there!"}"#
                    let s = val.pointer_as::<String>("/delta").ok_or_else(|| {
                        String::from("Reasoning summary text delta should be a string")
                    })?;
                    rv.parts.push(PartDelta::TextReasoning {
                        text: s.into(),
                        signature: None,
                    });
                }
                "response.reasoning_summary_text.done" => {
                    // r#"{"type":"response.reasoning_summary_text.done","text":"**Responding to a greeting**\n\nThe user just said, \"Hello!\" So, it seems I need to engage. I'll greet them back and offer help since they're looking to chat. I could say something like, \"Hello! How can I assist you today?\" That feels friendly and open. They didn't ask a specific question, so this approach will work well for starting a conversation. Let's see where it goes from there!"}"#
                }
                "response.function_call_arguments.delta" => {
                    // r#"{"type":"response.function_call_arguments.delta","delta":"{\""}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":"location"}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":"\":\""}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":"Paris"}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":","}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":" France"}"#,
                    // r#"{"type":"response.function_call_arguments.delta","delta":"\"}"}"#,
                    let s = val.pointer_as::<String>("/delta").ok_or_else(|| {
                        String::from("Function call argument delta should be a string")
                    })?;
                    rv.parts.push(PartDelta::FunctionToolCall {
                        id: None,
                        name: "".to_owned(),
                        arguments: s.into(),
                    });
                }
                "response.function_call_arguments.done" => {
                    // r#"{"type":"response.function_call_arguments.done","arguments":"{\"location\":\"Paris, France\"}"}"#,
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
                    } else if let Some(refusal) = content.get("refusal")
                        && !refusal.is_null()
                    {
                        let Some(text) = refusal.as_str() else {
                            return Err(String::from("Invalid refusal content"));
                        };
                        rv.parts.push(PartDelta::TextRefusal(text.into()));
                    } else {
                        return Err(String::from("Invalid part"));
                    }
                }
            } else {
                return Err(String::from("Invalid content"));
            }
        }

        // Parse reasoning
        if let Some(summary) = root.get("summary")
            && !summary.is_null()
        {
            if let Some(text) = summary.as_str() {
                // Contents can be a single string
                rv.parts.push(PartDelta::TextReasoning {
                    text: text.into(),
                    signature: None,
                });
            } else if let Some(summary) = summary.as_array() {
                // In case of part vector
                for content in summary {
                    let Some(content) = content.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    if let Some(text) = content.get("text") {
                        let Some(text) = text.as_str() else {
                            return Err(String::from("Invalid content part"));
                        };
                        rv.parts.push(PartDelta::TextReasoning {
                            text: text.into(),
                            signature: None,
                        });
                    } else {
                        return Err(String::from("Invalid part"));
                    }
                }
            } else {
                return Err(String::from("Invalid content"));
            }
        }

        // Parse tool calls
        if let Some(ty) = root.get("type")
            && ty.as_str() == Some("function_call")
        {
            let id = match root.get("call_id") {
                Some(id) if id.is_string() => Some(id.as_str().unwrap().to_owned()),
                _ => None,
            };
            let name = match root.get("name") {
                Some(name) if name.is_string() => name.as_str().unwrap().to_owned(),
                _ => String::new(),
            };
            let arguments = match root.get("arguments") {
                Some(args) if args.is_string() => args.as_str().unwrap().to_owned(),
                _ => String::new(),
            };
            rv.parts.push(PartDelta::FunctionToolCall {
                id,
                name,
                arguments,
            });
        };

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
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"input_text","text":"Explain me about Riemann hypothesis"},{"type":"input_text","text":"How cold brew is different from the normal coffee?"}]}"#
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
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"type":"function_call","call_id":"funcid_123456","name":"temperature","arguments":"{\"unit\":\"celsius\"}"}"#
        );

        let mut msg = Message::new(Role::Assistant);
        msg.parts.push(Part::function_tool_call_with_id(
            "temperature",
            Value::object([("unit", "fahrenheit")]),
            "funcid_7890ab",
        ));
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"type":"function_call","call_id":"funcid_7890ab","name":"temperature","arguments":"{\"unit\":\"fahrenheit\"}"}"#
        );
        // println!("{}", serde_json::to_string(&marshaled).unwrap());
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
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"input_text","text":"What you can see in this image?"},{"type":"input_image","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#
        );
    }

    #[test]
    pub fn deserialize_text() {
        let inputs = [
            r#"{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role": "assistant"}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
            r#"{"type":"response.output_item.added","item":{"type":"message","content":[],"role":"assistant"}}"#,
            r#"{"type":"response.content_part.added","part":{"type":"output_text","text":""}}"#,
            r#"{"type":"response.output_text.delta","delta":"Hello"}"#,
            r#"{"type":"response.output_text.delta","delta":" world!"}"#,
            r#"{"type":"response.output_text.done","text":"Hello world!"}"#,
            r#"{"type":"response.content_part.done","part":{"type":"output_text","text":"Hello world!"}}"#,
            r#"{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role":"assistant"}}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
            r#"{"type":"reasoning","summary":[{"type":"summary_text","text":"**Answering a simple question**\n\nUser is saying hello."}]}"#,
            r#"{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role": "assistant"}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
    pub fn deserialize_text_with_reasoning_stream() {
        let inputs = [
            r#"{"type":"response.output_item.added","item":{"type":"message","content":[],"role":"assistant"}}"#,
            r#"{"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"**Answering a simple question**\n\n"}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"User is saying hello."}"#,
            r#"{"type":"response.reasoning_summary_text.done","text":"**Answering a simple question**\n\nUser is saying hello."}"#,
            r#"{"type":"response.reasoning_summary_part.done","part":{"type":"summary_text","text":"**Answering a simple question**\n\nUser is saying hello."}}"#,
            r#"{"type":"response.content_part.added","part":{"type":"output_text","text":""}}"#,
            r#"{"type":"response.output_text.delta","delta":"Hello"}"#,
            r#"{"type":"response.output_text.delta","delta":" world!"}"#,
            r#"{"type":"response.output_text.done","text":"Hello world!"}"#,
            r#"{"type":"response.content_part.done","part":{"type":"output_text","text":"Hello world!"}}"#,
            r#"{"type":"response.output_item.done","item":{"type":"message","content":[{"type":"output_text","text":"Hello world!"}],"role":"assistant"}}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
            r#"{"type":"function_call","arguments":"{\"location\":\"Paris, France\"}","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather"}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
            r#"{"type":"response.output_item.added","item":{"type":"function_call","arguments":"","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather"}}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":"{\""}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":"location"}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":"\":\""}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":"Paris"}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":","}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":" France"}"#,
            r#"{"type":"response.function_call_arguments.delta","delta":"\"}"}"#,
            r#"{"type":"response.function_call_arguments.done","arguments":"{\"location\":\"Paris, France\"}"}"#,
            r#"{"type":"response.output_item.done","item":{"type":"function_call","arguments":"{\"location\":\"Paris, France\"}","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather"}}"#,
        ];
        let mut u = OpenAIUnmarshal;

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
