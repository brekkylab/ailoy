use base64::Engine;
use indexmap::IndexMap;

use crate::{
    to_value,
    value::{Marshal, Message, MessageDelta, Part, PartDelta, Role, ToolDesc, Unmarshal, Value},
};

#[derive(Clone, Debug, Default)]
pub struct GeminiMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text(s) => {
                to_value!({"text": s})
            }
            Part::Function {
                id,
                name,
                arguments,
            } => {
                let mut value =
                    to_value!({"functionCall": {"name": name, "args": arguments.clone()}});
                if let Some(id) = id {
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("id".into(), id.into());
                }
                value
            }
            Part::Image { .. } => {
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
                // Final value
                to_value!({"inline_data": {"mime_type": "image/png", "data": encoded}})
            }
            Part::Value(v) => v.to_owned(),
        }
    };

    // Collecting contents
    let mut parts = Vec::<Value>::new();
    if !msg.think.is_empty() && include_thinking {
        parts.push(to_value!({"text": msg.think.clone(), "thought": true}));
    }
    parts.extend(msg.contents.iter().map(part_to_value));
    parts.extend(msg.tool_calls.iter().map(part_to_value));

    // Final message object with role and collected parts
    to_value!({"role": msg.role.to_string(), "contents": [{"parts": parts}]})
}

impl Marshal<Message> for GeminiMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        marshal_message(msg, true)
    }
}

impl Marshal<Vec<Message>> for GeminiMarshal {
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

impl Marshal<ToolDesc> for GeminiMarshal {
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
pub struct GeminiUnmarshal;

impl Unmarshal<MessageDelta> for GeminiUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
        let mut rv = MessageDelta::default();

        // r#"{"candidates":[{"content":{"parts":[{"text":"Hello world!"}],"role":"model"}}]}"#,

        let content: &IndexMap<String, Value> = val
            .pointer_as::<IndexMap<String, Value>>("/candidates/0/content")
            .ok_or_else(|| String::from("Content should be an object"))?;

        // Parse role
        if let Some(r) = content.get("role") {
            let s = r
                .as_str()
                .ok_or_else(|| String::from("Role should be a string"))?;
            let v = match s {
                "system" => Ok(Role::System),
                "user" => Ok(Role::User),
                "assistant" => Ok(Role::Assistant),
                "model" => Ok(Role::Assistant),
                "tool" => Ok(Role::Tool),
                other => Err(format!("Unknown role: {other}")),
            }?;
            rv.role = Some(v);
        }

        // Parse parts
        if let Some(parts) = content.get("parts")
            && !parts.is_null()
        {
            if let Some(parts) = parts.as_array() {
                // In case of part vector
                for part in parts {
                    let Some(part) = part.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    let thought = part
                        .get("thought")
                        .and_then(|thought| thought.as_bool())
                        .unwrap_or(false);
                    if let Some(text) = part.get("text") {
                        let Some(text) = text.as_str() else {
                            return Err(String::from("Invalid content part"));
                        };
                        if thought {
                            rv.think = text.into();
                        } else {
                            rv.contents.push(PartDelta::Text(text.into()));
                        }
                    } else if let Some(tool_call_obj) = part.get("functionCall") {
                        let Some(tool_call_obj) = tool_call_obj.as_object() else {
                            return Err(String::from("Invalid functionCall object"));
                        };
                        let name = tool_call_obj
                            .get("name")
                            .and_then(|name| name.as_str())
                            .map(|name| name.to_owned())
                            .unwrap_or_default();
                        let arguments = match tool_call_obj.get("args") {
                            Some(args) => args.to_owned(),
                            None => Value::Null,
                        };
                        rv.tool_calls.push(PartDelta::ParsedFunction {
                            id: None,
                            name,
                            arguments,
                        });
                    } else {
                        return Err(String::from("Invalid part"));
                    }
                }
            } else {
                return Err(String::from("Invalid parts"));
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
        let msg = Message::new(Role::User).with_contents([
            Part::text("Explain me about Riemann hypothesis."),
            Part::text("How cold brew is different from the normal coffee?"),
        ]);
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","contents":[{"parts":[{"text":"Explain me about Riemann hypothesis."},{"text":"How cold brew is different from the normal coffee?"}]}]}"#
        );
    }

    #[test]
    pub fn serialize_messages_with_reasonings() {
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Hello there."), Part::text("How are you?")]),
            Message::new(Role::Assistant)
                .with_think_signature("This is reasoning text would be vanished.", "")
                .with_contents([Part::text("I'm fine, thank you. And you?")]),
            Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
            Message::new(Role::Assistant)
                .with_think_signature(
                    "This is reasoning text would be remaining.",
                    "Ev4MCkYIBxgCKkDl5A",
                )
                .with_contents([Part::text("Is there anything I can help with?")]),
        ];
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msgs);
        println!("{}", serde_json::to_string(&marshaled).unwrap());
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","contents":[{"parts":[{"text":"Hello there."},{"text":"How are you?"}]}]},{"role":"assistant","contents":[{"parts":[{"text":"I'm fine, thank you. And you?"}]}]},{"role":"user","contents":[{"parts":[{"text":"I'm okay."}]}]},{"role":"assistant","contents":[{"parts":[{"text":"This is reasoning text would be remaining.","thought":true},{"text":"Is there anything I can help with?"}]}]}]"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let msg = Message::new(Role::Assistant).with_tool_calls([
            Part::function("temperature", Value::object([("unit", "celsius")])),
            Part::function("temperature", Value::object([("unit", "fahrenheit")])),
        ]);
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","contents":[{"parts":[{"functionCall":{"name":"temperature","args":{"unit":"celsius"}}},{"functionCall":{"name":"temperature","args":{"unit":"fahrenheit"}}}]}]}"#
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
            Part::image(3, 3, "grayscale", raw_pixels).unwrap(),
        ]);
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","contents":[{"parts":[{"text":"What you can see in this image?"},{"inline_data":{"mime_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}]}"#,
        );
    }

    #[test]
    pub fn deserialize_text() {
        let inputs =
            [r#"{"candidates":[{"content":{"parts":[{"text":"Hello world!"}],"role":"model"}}]}"#];
        let mut u = GeminiUnmarshal;

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
            r#"{"candidates": [{"content": {"parts": [{"text": "Hello"}],"role": "model"}}]}"#,
            r#"{"candidates": [{"content": {"parts": [{"text": " world!"}],"role": "model"}}]}"#,
        ];
        let mut u = GeminiUnmarshal;

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
            r#"{"candidates":[{"content":{"parts":[{"text":"**Answering a simple question**\n\nUser is saying hello.","thought":true},{"text":"Hello world!"}],"role":"model"}}]}"#,
        ];
        let mut u = GeminiUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.think,
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_text_with_reasoning_stream() {
        let inputs = [
            r#"{"candidates":[{"content":{"parts":[{"text":"**Answering a simple question**\n\n","thought": true}],"role": "model"}}]}"#,
            r#"{"candidates":[{"content":{"parts":[{"text":"User is saying hello.","thought": true}],"role": "model"}}]}"#,
            r#"{"candidates": [{"content": {"parts": [{"text": "Hello"}],"role": "model"}}]}"#,
            r#"{"candidates": [{"content": {"parts": [{"text": " world!"}],"role": "model"}}]}"#,
        ];
        let mut u = GeminiUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.think,
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_tool_call() {
        let inputs = [
            r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris, France"}}}],"role":"model"}}]}"#,
        ];
        let mut u = GeminiUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.tool_calls.len(), 1);
        let tool_call = delta.tool_calls.pop().unwrap();
        let (id, name, args) = tool_call.to_parsed_function().unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            serde_json::to_string(&args).unwrap(),
            "{\"location\":\"Paris, France\"}"
        );
    }
}
