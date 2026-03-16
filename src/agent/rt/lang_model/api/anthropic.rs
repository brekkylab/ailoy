use crate::{
    agent::rt::lang_model::LangModelRequest,
    datatype::Value,
    message::{
        Marshal, Message, MessageDeltaOutput, Part, PartFunction, PartImage, Role, ToolDesc,
        Unmarshal,
    },
    to_value,
};

#[derive(Clone, Debug, Default)]
pub struct AnthropicMarshal;

impl Marshal<Message> for AnthropicMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
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
                Part::Image { image } => match image {
                    PartImage::Embedded { mime_type, data } => {
                        to_value!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": data.base64(),
                            }
                        })
                    }
                    PartImage::Url { url } => {
                        to_value!({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url,
                            }
                        })
                    }
                },
            }
        };

        if item.role == Role::Tool {
            return to_value!(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.id.clone().expect("Tool call id must exist."),
                            "content": part_to_value(&item.contents[0])
                        }
                    ]
                }
            );
        }

        // Collecting contents
        let mut contents = Vec::<Value>::new();
        if let Some(thinking) = &item.thinking
            && !thinking.is_empty()
        {
            let mut part = to_value!({"type": "thinking", "thinking": thinking});
            if let Some(sig) = &item.signature {
                part.as_object_mut()
                    .unwrap()
                    .insert("signature".into(), sig.into());
            }
            contents.push(part);
        }
        contents.extend(item.contents.iter().map(part_to_value));
        contents.extend(
            item.tool_calls
                .clone()
                .unwrap_or(vec![])
                .iter()
                .map(part_to_value),
        );

        // Final message object with role and collected parts
        to_value!({"role": item.role.to_string(), "content": contents})
    }
}

impl Marshal<ToolDesc> for AnthropicMarshal {
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

impl<'a> Marshal<LangModelRequest<'a>> for AnthropicMarshal {
    fn marshal(&mut self, req: &LangModelRequest) -> Value {
        let model = Value::from(req.model);
        let messages = self.marshal(req.messages);
        let tools = if !req.tools.is_empty() {
            self.marshal(req.tools)
        } else {
            Value::Null
        };

        let url = req.url.to_string();

        let mut header = to_value!({
            "Content-Type": "application/json",
        });
        if let Some(api_key) = &req.api_key {
            header
                .as_object_mut()
                .unwrap()
                .insert("Authorization".into(), format!("Bearer {}", api_key).into());
        }

        let mut body = to_value!({
            "model": model,
            "messages": messages,
        });
        if !tools.is_null() {
            body.as_object_mut()
                .unwrap()
                .insert("tool_choice".to_owned(), to_value!("auto"));
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_owned(), tools);
        }
        body.as_object_mut()
            .unwrap()
            .retain(|_key, value| !value.is_null());

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicUnmarshal;

impl Unmarshal<MessageDeltaOutput> for AnthropicUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        todo!()
    }
}
