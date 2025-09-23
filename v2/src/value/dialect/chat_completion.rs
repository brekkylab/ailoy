/// Marshal and Unmarshal logic for OpenAI chat completion API (a.k.a. OpenAI-compatible API)
use base64::Engine;

use crate::{
    to_value,
    value::{Marshal, Message, MessageDelta, Part, PartDelta, Role, ToolDesc, Unmarshal, Value},
};

#[derive(Clone, Debug, Default)]
pub struct ChatCompletionMarshal;

fn marshal_message(item: &Message) -> Value {
    // Separate arrays for different categories of parts
    let mut contents = Value::array_empty();
    let mut tool_calls = Value::array_empty();
    let mut refusal = Value::array_empty();

    // Encode each message part
    for part in &item.parts {
        match part {
            Part::TextReasoning { .. } => {
                // ignore
            }
            Part::TextContent(s) => {
                let value = to_value!({"type": "text", "text": s});
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
                // TODO: cover mimetypes othen than image/png
                let value = to_value!({"type": "image_url","image_url": {"url": format!("data:image/png;base64,{}", encoded)}});
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
                let mut value = to_value!({"type": "function", "function": {"name": name, "arguments": arguments_string}});
                if let Some(id) = id {
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("id".into(), id.into());
                };
                tool_calls.as_array_mut().unwrap().push(value);
            }
            Part::TextRefusal(s) => {
                let value = to_value!({"type": "text", "text": s});
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
            .insert("tool_calls".into(), tool_calls);
    }
    if !refusal.as_array().unwrap().is_empty() {
        rv.as_object_mut()
            .unwrap()
            .insert("refusal".into(), refusal);
    }
    rv
}

impl Marshal<Message> for ChatCompletionMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        marshal_message(msg)
    }
}

impl Marshal<Vec<Message>> for ChatCompletionMarshal {
    fn marshal(&mut self, msgs: &Vec<Message>) -> Value {
        Value::array(
            msgs.iter()
                .map(|msg| marshal_message(msg))
                .collect::<Vec<_>>(),
        )
    }
}

impl Marshal<ToolDesc> for ChatCompletionMarshal {
    fn marshal(&mut self, item: &ToolDesc) -> Value {
        if let Some(desc) = &item.description {
            to_value!({
                "type": "function",
                "function": {
                    "name": &item.name,
                    "description": desc,
                    "parameters": item.parameters.clone()
                }
            })
        } else {
            to_value!({
                "type": "function",
                "function": {
                    "name": &item.name,
                    "parameters": item.parameters.clone()
                }
            })
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatCompletionUnmarshal;

impl Unmarshal<MessageDelta> for ChatCompletionUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
        // Root must be an object
        let root = val
            .as_object()
            .ok_or_else(|| String::from("Root should be an object"))?;
        let mut rv = MessageDelta::default();

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

        // Parse `content` field (can be null, string, or array of part objects)
        if let Some(contents) = root.get("content")
            && !contents.is_null()
        {
            if let Some(text) = contents.as_str() {
                // Simple string case
                rv.parts.push(PartDelta::TextContent(text.into()));
            } else if let Some(contents) = contents.as_array() {
                // Multiple content parts
                for content in contents {
                    let Some(content) = content.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    if let Some(text) = content.get("text") {
                        let Some(text) = text.as_str() else {
                            return Err(String::from("Invalid content part"));
                        };
                        rv.parts.push(PartDelta::TextContent(text.into()));
                    } else {
                        return Err(String::from("Invalid part"));
                    }
                }
            } else {
                return Err(String::from("Invalid content"));
            }
        }

        // Parse `tool_calls` field (array of function calls)
        if let Some(tool_calls) = root.get("tool_calls")
            && !tool_calls.is_null()
        {
            if let Some(tool_calls) = tool_calls.as_array() {
                for tool_call in tool_calls {
                    let Some(tool_call) = tool_call.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    let id = match tool_call.get("id") {
                        Some(id) if id.is_string() => Some(id.as_str().unwrap().to_owned()),
                        _ => None,
                    };
                    if let Some(func) = tool_call.get("function") {
                        let Some(func) = func.as_object() else {
                            return Err(String::from("Invalid tool call part"));
                        };
                        let name = match func.get("name") {
                            Some(name) if name.is_string() => name.as_str().unwrap().to_owned(),
                            _ => String::new(),
                        };
                        let arguments = match func.get("arguments") {
                            Some(args) if args.is_string() => args.as_str().unwrap().to_owned(),
                            _ => String::new(),
                        };
                        rv.parts.push(PartDelta::FunctionToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                }
            } else {
                return Err(String::from("Invalid tool calls"));
            }
        };

        // Parse `refusal` field (may be null or string)
        if let Some(refusal) = root.get("refusal")
            && !refusal.is_null()
        {
            if let Some(text) = refusal.as_str() {
                rv.parts.push(PartDelta::TextRefusal(text.into()));
            }
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
            .push(Part::text_content("Explain me about Riemann hypothesis."));
        msg.parts.push(Part::text_content(
            "How cold brew is different from the normal coffee?",
        ));
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis."},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_messages_with_reasonings() {
        let mut msgs = vec![
            Message::new(Role::User),
            Message::new(Role::Assistant),
            Message::new(Role::User),
            Message::new(Role::Assistant),
        ];
        msgs[0].parts.push(Part::text_content("Hello there."));
        msgs[0].parts.push(Part::text_content("How are you?"));

        msgs[1].parts.push(Part::text_reasoning(
            "This is reasoning text would be vanished.",
        ));
        msgs[1]
            .parts
            .push(Part::text_content("I'm fine, thank you. And you?"));

        msgs[2].parts.push(Part::text_content("I'm okay."));

        msgs[3].parts.push(Part::text_reasoning(
            "This is reasoning text would be remaining.",
        ));
        msgs[3]
            .parts
            .push(Part::text_content("Is there anything I can help with?"));

        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"text","text":"Hello there."},{"type":"text","text":"How are you?"}]},{"role":"assistant","content":[{"type":"text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"text","text":"I'm okay."}]},{"role":"assistant","content":[{"type":"text","text":"Is there anything I can help with?"}]}]"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let mut msg = Message::new(Role::Assistant);
        msg.parts
            .push(Part::text_content("Calling the functions..."));
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
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","content":[{"type":"text","text":"Calling the functions..."}],"tool_calls":[{"type":"function","function":{"name":"temperature","arguments":"{\"unit\":\"celsius\"}"},"id":"funcid_123456"},{"type":"function","function":{"name":"temperature","arguments":"{\"unit\":\"fahrenheit\"}"},"id":"funcid_7890ab"}]}"#
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
        msg.parts
            .push(Part::image_content(3, 3, "grayscale", raw_pixels).unwrap());
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#
        );
    }

    #[test]
    pub fn deserialize_text() {
        let inputs = [
            r#"{"index":0,"role":"assistant","content":null}"#,
            r#"{"index":0,"content":"Hello"}"#,
            r#"{"index":0,"content":[{"type":"text","text":" world!"}]}"#,
        ];
        let mut u = ChatCompletionUnmarshal;

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
    pub fn deserialize_tool_call() {
        let inputs = [
            r#"{"role":"assistant","tool_calls":[{"index":0,"id":"call_DdmO9pD3xa9XTPNJ32zg2hcA","function":{"arguments":"","name":"get_weather"},"type":"function"}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"{\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"location","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"\":\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"Paris","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":",","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":" France","name":null},"type": null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"\"}","name":null},"type":null}]}"#,
        ];
        let mut u = ChatCompletionUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.parts.len(), 1);
        assert!(delta.parts[0].is_function());
        let (id, name, args) = delta.parts[0].as_function().unwrap();
        assert_eq!(id.unwrap(), "call_DdmO9pD3xa9XTPNJ32zg2hcA");
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }

    #[test]
    pub fn deserialize_parallel_tool_call() {
        let inputs = [
            r#"{"role":"assistant","tool_calls":[{"index":0,"id":"call_DdmO9pD3xa9XTPNJ32zg2hcA","function":{"arguments":"","name":"get_temperature"},"type":"function"}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"{\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"location","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"\":\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"Paris","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":",","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":" France","name":null},"type": null}]}"#,
            r#"{"tool_calls":[{"index":0,"id":null,"function":{"arguments":"\"}","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":"call_ABCDEF1234","function":{"arguments":"","name":"get_wind_speed"},"type":"function"}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":"{\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":"location","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":"\":\"","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":"Dubai","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":",","name":null},"type":null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":" Qatar","name":null},"type": null}]}"#,
            r#"{"tool_calls":[{"index":1,"id":null,"function":{"arguments":"\"}","name":null},"type":null}]}"#,
        ];
        let mut u = ChatCompletionUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.parts.len(), 2);
        assert!(delta.parts[0].is_function());
        let (id, name, args) = delta.parts[0].as_function().unwrap();
        assert_eq!(id.unwrap(), "call_DdmO9pD3xa9XTPNJ32zg2hcA");
        assert_eq!(name, "get_temperature");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
        let (id, name, args) = delta.parts[1].as_function().unwrap();
        assert_eq!(id.unwrap(), "call_ABCDEF1234");
        assert_eq!(name, "get_wind_speed");
        assert_eq!(args, "{\"location\":\"Dubai, Qatar\"}");
    }
}
