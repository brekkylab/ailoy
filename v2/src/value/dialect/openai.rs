use crate::value::{Data, Marshal, MediaData, Message, Mode, Value};

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
        let mut contents = Value::array_empty();
        let mut tool_calls = Value::array_empty();
        let mut refusal = Value::array_empty();

        // Encode each parts
        for part in &item.parts {
            let mut part_value = match &part.data {
                Data::Text(txt) => Value::object([("type", "text"), ("text", txt.as_str())]),
                Data::Function(func) => {
                    let (name, args) = func.clone().parse().unwrap();
                    Value::object([
                        ("type", Value::string("function")),
                        (
                            "function",
                            Value::object([("name", name), ("arguments", args)]),
                        ),
                    ])
                }
                Data::Image(data) => match data {
                    MediaData::URL(url) => Value::object([
                        ("type", Value::string("image_url")),
                        ("image_url", Value::object([("url", url)])),
                    ]),
                    MediaData::Base64 { media_type, data } => Value::object([
                        ("type", Value::string("image_url")),
                        (
                            "image_url",
                            Value::object([(
                                "url",
                                &format!("data:{};base64,{}", media_type, data),
                            )]),
                        ),
                    ]),
                },
            };

            match &part.mode {
                Mode::Think { .. } => {}
                Mode::Content => {
                    contents.as_array_mut().unwrap().push(part_value);
                }
                Mode::ToolCall { id } => {
                    if let Some(id) = id {
                        part_value
                            .as_object_mut()
                            .unwrap()
                            .insert("id".into(), id.into());
                    };
                    tool_calls.as_array_mut().unwrap().push(part_value);
                }
                Mode::Refusal => {
                    refusal.as_array_mut().unwrap().push(part_value);
                }
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Marshaled, Message, PartBuilder, Role};

    #[test]
    pub fn serialize_text() {
        let mut msg = Message::new(Role::User);
        msg.parts.push(
            PartBuilder::new()
                .content()
                .text("Explain me about Riemann hypothesis")
                .build()
                .unwrap(),
        );
        msg.parts.push(
            PartBuilder::new()
                .content()
                .text("How cold brew is different from the normal coffee?")
                .build()
                .unwrap(),
        );
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis"},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let mut msg = Message::new(Role::Assistant);
        msg.parts.push(
            PartBuilder::new()
                .content()
                .text("Calling the functions...")
                .build()
                .unwrap(),
        );
        msg.parts.push(
            PartBuilder::new()
                .content()
                .tool_call_with_id("funcid_123456")
                .function("temperature", "{\"unit\": \"celcius\"}")
                .build()
                .unwrap(),
        );
        msg.parts.push(
            PartBuilder::new()
                .content()
                .tool_call_with_id("funcid_7890ab")
                .function("temperature", "{\"unit\": \"fernheit\"}")
                .build()
                .unwrap(),
        );
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"assistant","content":[{"type":"text","text":"Calling the functions..."}],"tool_calls":[{"type":"function","function":{"name":"temperature","arguments":"{\"unit\": \"celcius\"}"},"id":"funcid_123456"},{"type":"function","function":{"name":"temperature","arguments":"{\"unit\": \"fernheit\"}"},"id":"funcid_7890ab"}]}"#
        );
    }

    #[test]
    pub fn serialize_image() {
        let mut msg = Message::new(Role::User);
        msg.parts.push(
            PartBuilder::new()
                .content()
                .text("What you can see in this image?")
                .build()
                .unwrap(),
        );
        msg.parts
            .push(PartBuilder::new().content().image_url("https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg").build().unwrap());
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image_url","image_url":{"url":"https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg"}}]}"#
        );
    }
}
