use base64::Engine;
use indexmap::{IndexMap, indexmap};

use crate::{
    to_value,
    value::{Marshal, Message, MessageDelta, Part, PartDelta, Role, ToolDesc, Unmarshal, Value},
};

#[derive(Clone, Debug, Default)]
pub struct GeminiMarshal;

fn marshal_message(msg: &Message, include_reasoning: bool) -> Value {
    let mut contents = Value::array_empty();
    let mut tool_calls = Value::array_empty();
    // let mut refusal = Value::array_empty();

    // Encode each parts
    for part in &msg.parts {
        match part {
            Part::TextReasoning { text, .. } => {
                if include_reasoning {
                    let value = to_value!({"text": text, "thought": true});
                    contents.as_array_mut().unwrap().push(value);
                }
            }
            Part::TextContent(s) => {
                let value = to_value!({"text": s});
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
                let value = to_value!({"inline_data":{"mime_type": "image/png", "data": encoded}});
                contents.as_array_mut().unwrap().push(value);
            }
            Part::FunctionToolCall {
                id,
                name,
                arguments,
            } => {
                let mut call_obj_map: IndexMap<String, Value> = IndexMap::new();
                call_obj_map.insert("name".to_owned(), name.into());
                call_obj_map.insert("args".to_owned(), arguments.clone());
                let mut value = Value::object([("functionCall", Value::object(call_obj_map))]);
                if let Some(id) = id {
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("id".into(), id.into());
                };
                tool_calls.as_array_mut().unwrap().push(value);
            }
            Part::TextRefusal(_s) => {
                // let value = Value::object([("type", "text"), ("text", s.as_str())]);
                // refusal.as_array_mut().unwrap().push(value);
            }
        };
    }

    // Final message object with role and collected parts
    let mut rv = to_value!({"role": msg.role.to_string()});
    let mut contents_vec: Vec<Value> = vec![];
    if !contents.as_array().unwrap().is_empty() {
        let parts_map: IndexMap<String, Value> = indexmap! {
            "parts".to_owned() => contents,
        };
        contents_vec.push(Value::Object(parts_map));
        // rv.as_object_mut().unwrap().insert(
        //     "contents".into(),
        //     Value::Array(vec![Value::Object(parts_map)]),
        // );
    }
    if !tool_calls.as_array().unwrap().is_empty() {
        let parts_map: IndexMap<String, Value> = indexmap! {
            "parts".to_owned() => tool_calls,
        };
        contents_vec.push(Value::Object(parts_map));
        // rv.as_object_mut().unwrap().insert(
        //     "contents".into(),
        //     Value::Array(vec![Value::Object(parts_map)]),
        // );
    }
    // if !refusal.as_array().unwrap().is_empty() {
    // }
    rv.as_object_mut()
        .unwrap()
        .insert("contents".into(), Value::Array(contents_vec));
    rv
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
                            rv.parts.push(PartDelta::TextReasoning {
                                text: text.into(),
                                signature: None,
                            });
                        } else {
                            rv.parts.push(PartDelta::TextContent(text.into()));
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
                        let arguments = tool_call_obj
                            .get("args")
                            .and_then(|args| args.as_object())
                            .map(|args| serde_json::to_string(args).unwrap())
                            .unwrap_or_default();
                        rv.parts.push(PartDelta::FunctionToolCall {
                            id: None,
                            name,
                            arguments,
                        });
                    }
                    // else if let Some(refusal) = part.get("refusal")
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
        let mut msg = Message::new(Role::User);
        msg.parts
            .push(Part::text_content("Explain me about Riemann hypothesis"));
        msg.parts.push(Part::text_content(
            "How cold brew is different from the normal coffee?",
        ));
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","contents":[{"parts":[{"text":"Explain me about Riemann hypothesis"},{"text":"How cold brew is different from the normal coffee?"}]}]}"#
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

        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msgs);
        println!("{}", serde_json::to_string(&marshaled).unwrap());
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","contents":[{"parts":[{"text":"Hello there."},{"text":"How are you?"}]}]},{"role":"assistant","contents":[{"parts":[{"text":"I'm fine, thank you. And you?"}]}]},{"role":"user","contents":[{"parts":[{"text":"I'm okay."}]}]},{"role":"assistant","contents":[{"parts":[{"text":"This is reasoning text would be remaining.","thought":true},{"text":"Is there anything I can help with?"}]}]}]"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let mut msg = Message::new(Role::Assistant);
        msg.parts.push(Part::function_tool_call(
            "temperature",
            Value::object([("unit", "celsius")]),
        ));
        msg.parts.push(Part::function_tool_call(
            "temperature",
            Value::object([("unit", "fahrenheit")]),
        ));
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

        let mut msg = Message::new(Role::User);
        msg.parts
            .push(Part::text_content("What you can see in this image?"));
        msg.parts
            .push(Part::image_content(3, 3, "grayscale", raw_pixels).unwrap());
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
        assert_eq!(delta.parts.len(), 1);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        assert_eq!(delta.parts[0].as_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.parts.len(), 1);
        assert_eq!(delta.role, Some(Role::Assistant));
        assert!(delta.parts[0].is_text());
        assert_eq!(delta.parts[0].as_text().unwrap(), "Hello world!");
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
            r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris, France"}}}],"role":"model"}}]}"#,
        ];
        let mut u = GeminiUnmarshal;

        let mut delta = MessageDelta::new();

        for input in inputs {
            let val = serde_json::from_str::<Value>(input).unwrap();
            let cur_delta = u.unmarshal(val).unwrap();
            delta = delta.aggregate(cur_delta).unwrap();
        }
        assert_eq!(delta.parts.len(), 1);
        assert!(delta.parts[0].is_function());
        let (_id, name, args) = delta.parts[0].as_function().unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }
}
