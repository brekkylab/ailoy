use base64::Engine;
use indexmap::IndexMap;

use crate::value::{
    Delta, Marshal, Message, MessageDelta, Part, PartDelta, Role, Unmarshal, Value,
};

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
        let mut contents = Value::array_empty();
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
                Part::ImageContent(b) => {
                    // base64 encoding
                    let encoded = base64::engine::general_purpose::STANDARD.encode(b);
                    let value = Value::object([
                        ("type", Value::string("image_url")),
                        ("image_url", Value::object([("url", encoded)])),
                    ]);
                    contents.as_array_mut().unwrap().push(value);
                }
                Part::FunctionToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let mut value = Value::object([("type", Value::string("function"))]);
                    if let Some(id) = id {
                        value
                            .as_object_mut()
                            .unwrap()
                            .insert("id".into(), id.into());
                    };
                    value.as_object_mut().unwrap().insert(
                        "function".into(),
                        Value::object([
                            ("name", name),
                            ("arguments", &serde_json::to_string(arguments).unwrap()),
                        ]),
                    );
                    tool_calls.as_array_mut().unwrap().push(value);
                }
                Part::TextRefusal(s) => {
                    let value = Value::object([("type", "text"), ("text", s.as_str())]);
                    refusal.as_array_mut().unwrap().push(value);
                }
            };
        }

        // Encode whole message
        let mut rv = Value::object([("role", item.role.to_string())]);
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
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIUnmarshal;

impl Unmarshal<MessageDelta> for OpenAIUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
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

        // Parse contents
        if let Some(parts_value) = root.get("content")
            && !parts_value.is_null()
        {
            if let Some(text_part) = parts_value.as_str() {
                rv.parts.push(PartDelta::TextContent(text_part.into()));
            } else if let Some(parts_value) = parts_value.as_array() {
                for part_value in parts_value {
                    let Some(part_value) = part_value.as_object() else {
                        return Err(String::from("Invalid part"));
                    };
                    if let Some(text_value) = part_value.get("text") {
                        let Some(text) = text_value.as_str() else {
                            return Err(String::from("Invalid part"));
                        };
                        rv.parts.push(PartDelta::TextContent(text.into()));
                    } else if let Some(func_value) = part_value.get("function") {
                        let Some(func_value) = func_value.as_object() else {
                            return Err(String::from("Invalid part"));
                        };
                        let name = match func_value.get("name") {
                            Some(name) => name.as_str().unwrap().to_owned(),
                            None => String::new(),
                        };
                        let arguments = match func_value.get("arguments") {
                            Some(name) => name.as_str().unwrap().to_owned(),
                            None => String::new(),
                        };
                        rv.parts
                            .push(PartDelta::FunctionToolCall { name, arguments });
                    }
                }
            }
        };
        Ok(rv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Marshaled, Message, Role, Unmarshaled};

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
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis"},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let mut msg = Message::new(Role::Assistant);
        msg.parts
            .push(Part::text_content("Calling the functions..."));
        msg.parts.push(Part::function_tool_call_with_id(
            "temperatrue",
            "{\"unit\": \"celcius\"}",
            "funcid_123456",
        ));
        msg.parts.push(Part::function_tool_call_with_id(
            "temperatrue",
            "{\"unit\": \"fernheit\"}",
            "funcid_123456",
        ));
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","content":[{"type":"text","text":"Calling the functions..."}],"tool_calls":[{"type":"function","function":{"name":"temperature","arguments":"{\"unit\": \"celcius\"}"},"id":"funcid_123456"},{"type":"function","function":{"name":"temperature","arguments":"{\"unit\": \"fernheit\"}"},"id":"funcid_7890ab"}]}"#
        );
    }

    #[test]
    pub fn serialize_image() {
        let mut msg = Message::new(Role::User);
        msg.parts
            .push(Part::text_content("What you can see in this image?"));
        let bytes: Vec<u8> = base64::engine::general_purpose::STANDARD
            .decode(b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=")
            .expect("invalid base64");
        msg.parts.push(Part::image_content(bytes));
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image_url","image_url":{"url":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="}}]}"#
        );
    }

    #[test]
    pub fn deserialize_text() {
        let delta1 = r#"{"role":"assistant","content":""}"#;
        let delta2 = r#"{"content":"Hello"}"#;
        let delta3 = r#"{"content":[{"type":"text","text":" world!"}]}"#;
        let mut u = OpenAIUnmarshal;

        let delta = MessageDelta::new();

        let val = serde_json::from_str::<Value>(delta1).unwrap();
        let cur_delta = u.unmarshal(val).unwrap();
        assert_eq!(cur_delta.role.clone().unwrap(), Role::Assistant);
        let delta = delta.aggregate(cur_delta).unwrap();

        let val = serde_json::from_str::<Value>(delta2).unwrap();
        let cur_delta = u.unmarshal(val).unwrap();
        assert_eq!(cur_delta.parts.len(), 1);
        assert_eq!(cur_delta.parts[0].as_text().unwrap(), "Hello");
        let delta = delta.aggregate(cur_delta).unwrap();

        let val = serde_json::from_str::<Value>(delta3).unwrap();
        let cur_delta = u.unmarshal(val).unwrap();
        assert_eq!(cur_delta.parts.len(), 1);
        assert_eq!(cur_delta.parts[0].as_text().unwrap(), " world!");
        let delta = delta.aggregate(cur_delta).unwrap();

        assert_eq!(delta.parts[0].as_text().unwrap(), "Hello world!");
    }
}
