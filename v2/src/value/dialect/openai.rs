use base64::Engine;

use crate::{
    to_value,
    value::{
        Config, Marshal, Message, MessageDelta, Part, PartDelta, PartDeltaFunction, PartFunction,
        Role, ThinkingOption, ToolDesc, Unmarshal, Value,
    },
};

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Vec<Value> {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => to_value!({"type": "input_text", "text": text}),
            Part::Function {
                id,
                f: PartFunction { name, args },
            } => {
                let arguments_string = serde_json::to_string(args).unwrap();
                if let Some(id) = id {
                    to_value!({"type": "function_call", "call_id":id, "name": name, "arguments": arguments_string})
                } else {
                    to_value!({"type": "function_call", "name": name, "arguments": arguments_string})
                }
            }
            Part::Value { value } => {
                to_value!({"type": "input_text", "text": serde_json::to_string(value).unwrap()})
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
                to_value!({"type": "input_image","image_url": {"url": format!("data:image/png;base64,{}", encoded)}})
            }
        }
    };
    let mut rv = Vec::<Value>::new();
    if !msg.thinking.is_empty() && include_thinking {
        rv.push(to_value!({"type": "reasoning", "summary": [{"type": "summary_text", "text": &msg.thinking}]}));
    }
    if !msg.contents.is_empty() {
        rv.push(to_value!({"role": msg.role.to_string(), "content": msg.contents.iter().map(part_to_value).collect::<Vec<_>>()}));
    }
    rv.extend(msg.tool_calls.iter().map(part_to_value));
    rv
}

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        to_value!(marshal_message(msg, true))
    }
}

impl Marshal<Vec<Message>> for OpenAIMarshal {
    fn marshal(&mut self, msgs: &Vec<Message>) -> Value {
        let last_user_index = msgs
            .iter()
            .rposition(|m| m.role == Role::User)
            .unwrap_or_else(|| msgs.len());
        Value::array(
            msgs.iter()
                .enumerate()
                .map(|(i, msg)| marshal_message(msg, i > last_user_index))
                .flatten()
                .collect::<Vec<_>>(),
        )
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

impl Marshal<Config> for OpenAIMarshal {
    fn marshal(&mut self, config: &Config) -> Value {
        let Some(model) = &config.model else {
            panic!("Cannot marshal `Config` without `model`.");
        };

        let is_reasoning_model = if model.starts_with("o") || model.starts_with("gpt-5") {
            true
        } else {
            false
        };

        let (reasoning_effort, reasoning_summary) = if is_reasoning_model {
            match &config.thinking_option {
                ThinkingOption::Disable => {
                    if model.starts_with("gpt-5") {
                        (Some("minimal"), None)
                    } else {
                        (Some("low"), None)
                    }
                }
                ThinkingOption::Enable | ThinkingOption::Medium => (Some("medium"), Some("auto")),
                ThinkingOption::Low => (Some("low"), Some("auto")),
                ThinkingOption::High => (Some("high"), Some("auto")),
            }
        } else {
            (None, None)
        };
        let reasoning = match (reasoning_effort, reasoning_summary) {
            (Some(effort), Some(summary)) => {
                to_value!({"effort": effort, "summary": summary})
            }
            (Some(effort), None) => {
                to_value!({"effort": effort})
            }
            (None, _) => Value::Null,
        };

        let instruction = if let Some(system_message) = &config.system_message {
            to_value!(system_message)
        } else {
            Value::Null
        };

        let max_output_tokens = if let Some(max_tokens) = &config.max_tokens {
            to_value!(*max_tokens as i64)
        } else {
            Value::Null
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
            "instructions": instruction,
            "reasoning": reasoning,
            "stream": stream,
            "max_output_tokens": max_output_tokens,
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
                        && let id = if let Some(id) = val.pointer_as::<String>("/item/call_id") {
                            Some(id.to_owned())
                        } else {
                            None
                        }
                        && let Some(name) = val.pointer_as::<String>("/item/name")
                        && let Some(args) = val.pointer_as::<String>("/item/arguments")
                    {
                        // tool call message
                        // r#"{"type":"response.output_item.added","item":{"type":"function_call","arguments":"","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather"}}"#
                        rv.role = Some(Role::Assistant);
                        rv.tool_calls.push(PartDelta::Function {
                            id,
                            f: PartDeltaFunction::WithStringArgs {
                                name: name.to_owned(),
                                args: args.to_owned(),
                            },
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
                    let text = val
                        .pointer_as::<String>("/delta")
                        .ok_or_else(|| String::from("Output text delta should be a string"))?
                        .to_owned();
                    rv.contents.push(PartDelta::Text { text });
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
                    rv.thinking.push_str(s.as_str());
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
                    let args = val
                        .pointer_as::<String>("/delta")
                        .ok_or_else(|| {
                            String::from("Function call argument delta should be a string")
                        })?
                        .to_owned();
                    rv.tool_calls.push(PartDelta::Function {
                        id: None,
                        f: PartDeltaFunction::WithStringArgs {
                            name: String::new(),
                            args,
                        },
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
                rv.contents.push(PartDelta::Text { text: text.into() });
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
                        rv.contents.push(PartDelta::Text { text: text.into() });
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

        // Parse reasoning
        if let Some(summary) = root.get("summary")
            && !summary.is_null()
        {
            if let Some(text) = summary.as_str() {
                // Can summary be a single string?
                rv.thinking.push_str(text);
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
                        rv.thinking.push_str(text);
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
            let args = match root.get("arguments") {
                Some(args) if args.is_string() => args.as_str().unwrap().to_owned(),
                _ => String::new(),
            };
            rv.tool_calls.push(PartDelta::Function {
                id,
                f: PartDeltaFunction::WithStringArgs { name, args },
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
        let msg = Message::new(Role::User)
            .with_contents([Part::text("Explain me about Riemann hypothesis.")]);
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"input_text","text":"Explain me about Riemann hypothesis."}]}]"#
        );
    }

    #[test]
    pub fn serialize_messages_with_thinkings() {
        let msgs = vec![
            Message::new(Role::User).with_contents([Part::text("Hello there.")]),
            Message::new(Role::Assistant)
                .with_thinking_signature("This is thinking text would be vanished.", "")
                .with_contents([Part::text("I'm fine, thank you. And you?")]),
            Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
            Message::new(Role::Assistant)
                .with_thinking_signature(
                    "This is thinking text would be remaining.",
                    "Ev4MCkYIBxgCKkDl5A",
                )
                .with_contents([Part::text("Is there anything I can help with?")]),
        ];
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"input_text","text":"Hello there."}]},{"role":"assistant","content":[{"type":"input_text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"input_text","text":"I'm okay."}]},{"type":"reasoning","summary":[{"type":"summary_text","text":"This is thinking text would be remaining."}]},{"role":"assistant","content":[{"type":"input_text","text":"Is there anything I can help with?"}]}]"#
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
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"type":"function_call","call_id":"funcid_123456","name":"temperature","arguments":"{\"unit\":\"celsius\"}"},{"type":"function_call","call_id":"funcid_7890ab","name":"temperature","arguments":"{\"unit\":\"fahrenheit\"}"}]"#
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
        let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"input_text","text":"What you can see in this image?"},{"type":"input_image","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}]"#
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
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.thinking,
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(
            delta.thinking,
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.tool_calls.len(), 1);
        let tool_call = delta.tool_calls.pop().unwrap();
        let (id, name, args) = tool_call.to_function().unwrap();
        assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
        assert_eq!(name, "get_weather");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
    }
}
