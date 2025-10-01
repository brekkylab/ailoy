use base64::Engine;

use crate::{
    model::{ServerEvent, ThinkEffort, api::RequestConfig},
    to_value,
    value::{
        FinishReason, Marshal, Marshaled, Message, MessageDelta, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, Role, ToolDesc, Unmarshal, Unmarshaled, Value,
    },
};

#[derive(Clone, Debug, Default)]
struct ChatCompletionMarshal;

fn marshal_message(item: &Message) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => to_value!({"type": "text", "text": text}),
            Part::Function {
                id,
                f: PartFunction { name, args },
            } => {
                let mut value = to_value!({"type": "function", "function": {"name": name, "arguments": serde_json::to_string(&args).unwrap()}});
                if let Some(id) = &id {
                    value
                        .as_object_mut()
                        .unwrap()
                        .insert("id".into(), id.into());
                }
                value
            }
            Part::Value { value } => {
                to_value!({"type": "text", "text": serde_json::to_string(&value).unwrap()})
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
                // To value
                to_value!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", encoded)}})
            }
        }
    };

    let mut rv = to_value!({"role": item.role.to_string()});
    if item.role == Role::Tool
        && let Some(id) = &item.id
    {
        rv.as_object_mut()
            .unwrap()
            .insert("tool_call_id".into(), id.into());
    }
    if !item.contents.is_empty() {
        rv.as_object_mut().unwrap().insert(
            "content".into(),
            item.contents
                .iter()
                .map(part_to_value)
                .collect::<Vec<_>>()
                .into(),
        );
    }
    if !item.tool_calls.is_empty() {
        rv.as_object_mut().unwrap().insert(
            "tool_calls".into(),
            item.tool_calls
                .iter()
                .map(part_to_value)
                .collect::<Vec<_>>()
                .into(),
        );
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

impl Marshal<RequestConfig> for ChatCompletionMarshal {
    fn marshal(&mut self, config: &RequestConfig) -> Value {
        let Some(model) = &config.model else {
            panic!("Cannot marshal `Config` without `model`.");
        };

        // "grok-4", "grok-code-fast-1" are reasoning models, but only treating reasoning internally.
        let is_reasoning_model = model.starts_with("grok-3-mini");

        let reasoning_effort = is_reasoning_model
            .then(|| {
                let effort = match &config.think_effort {
                    ThinkEffort::Disable => return Value::Null,
                    // grok doesn't have 'medium', setting default option as 'low'.
                    ThinkEffort::Enable | ThinkEffort::Medium | ThinkEffort::Low => "low",
                    ThinkEffort::High => "high",
                };
                Value::String(effort.to_owned())
            })
            .unwrap_or(Value::Null);

        let max_completion_tokens = if let Some(max_tokens) = &config.max_tokens {
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
            "reasoning_effort": reasoning_effort,
            "stream": stream,
            "max_completion_tokens": max_completion_tokens,
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
struct ChatCompletionUnmarshal;

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
                rv.contents.push(PartDelta::Text { text: text.into() });
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
                        rv.contents.push(PartDelta::Text { text: text.into() });
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
                        let args = match func.get("arguments") {
                            Some(args) if args.is_string() => args.as_str().unwrap().to_owned(),
                            _ => String::new(),
                        };
                        rv.tool_calls.push(PartDelta::Function {
                            id,
                            f: PartDeltaFunction::WithStringArgs { name, args },
                        });
                    }
                }
            } else {
                return Err(String::from("Invalid tool calls"));
            }
        };

        Ok(rv)
    }
}

pub(super) fn make_request(
    url: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: RequestConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(Marshaled::<_, ChatCompletionMarshal>::new(&config));
    let msgs = if let Some(system_message) = config.system_message {
        let mut new_msgs =
            vec![Message::new(Role::System).with_contents([Part::text(system_message)])];
        new_msgs.extend(msgs);
        new_msgs
    } else {
        msgs
    };
    body["messages"] = serde_json::json!(Marshaled::<_, ChatCompletionMarshal>::new(&msgs));

    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!("auto");
        body["tools"] = serde_json::json!(
            tools
                .iter()
                .map(|v| Marshaled::<_, ChatCompletionMarshal>::new(v))
                .collect::<Vec<_>>()
        );
    }

    reqwest::Client::new()
        .request(reqwest::Method::POST, url)
        .bearer_auth(api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub(super) fn handle_event(evt: ServerEvent) -> MessageOutput {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };
    let Some(choice) = j.pointer("/choices/0/delta") else {
        return MessageOutput::default();
    };
    let finish_reason = j
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "stop" => FinishReason::Stop(),
            "length" => FinishReason::Length(),
            "tool_calls" => FinishReason::ToolCall(),
            "content_filter" => {
                FinishReason::Refusal("Model output violated XAI's safety policy.".to_owned())
            }
            reason => FinishReason::Refusal(format!("reason: {}", reason)),
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal(_)) => MessageDelta::default(),
        _ => serde_json::from_value::<Unmarshaled<_, ChatCompletionUnmarshal>>(choice.clone())
            .ok()
            .map(|decoded| decoded.get())
            .unwrap_or_default(),
    };

    MessageOutput {
        delta,
        finish_reason,
    }
}

#[cfg(test)]
mod dialect_tests {
    use super::*;
    use crate::value::{Delta, Marshaled, Message, Role};

    #[test]
    pub fn serialize_text() {
        let msg = Message::new(Role::User).with_contents([
            Part::text("Explain me about Riemann hypothesis."),
            Part::text("How cold brew is different from the normal coffee?"),
        ]);
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"Explain me about Riemann hypothesis."},{"type":"text","text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_messages_with_reasonings() {
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Hello there."), Part::text("How are you?")]),
            Message::new(Role::Assistant)
                .with_thinking("This is reasoning text would be vanished.")
                .with_contents([Part::text("I'm fine, thank you. And you?")]),
            Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
            Message::new(Role::Assistant)
                .with_thinking("This is reasoning text would be remaining.")
                .with_contents([Part::text("Is there anything I can help with?")]),
        ];
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","content":[{"type":"text","text":"Hello there."},{"type":"text","text":"How are you?"}]},{"role":"assistant","content":[{"type":"text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"text","text":"I'm okay."}]},{"role":"assistant","content":[{"type":"text","text":"Is there anything I can help with?"}]}]"#
        );
    }

    #[test]
    pub fn serialize_function() {
        let msg = Message::new(Role::Assistant)
            .with_contents([Part::text("Calling the functions...")])
            .with_tool_calls([
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
        let msg = Message::new(Role::User).with_contents([
            Part::text("What you can see in this image?"),
            Part::image_binary(3, 3, "grayscale", raw_pixels).unwrap(),
        ]);
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","content":[{"type":"text","text":"What you can see in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#
        );
    }

    #[test]
    pub fn serialize_config() {
        let config = RequestConfig {
            model: Some("grok-3-mini".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"model":"grok-3-mini","reasoning_effort":"low","stream":true,"max_completion_tokens":1024}"#
        );

        let config = RequestConfig {
            model: Some("grok-4-fast".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, ChatCompletionMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"model":"grok-4-fast","stream":true,"max_completion_tokens":1024,"temperature":0.6,"top_p":0.9}"#
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
        assert_eq!(delta.role, Some(Role::Assistant));
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
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
        assert_eq!(delta.tool_calls.len(), 1);
        let tool_call = delta.tool_calls.pop().unwrap();
        assert!(tool_call.is_function());
        let (id, name, args) = tool_call.to_function().unwrap();
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
        assert_eq!(delta.tool_calls.len(), 2);
        assert!(delta.tool_calls[0].is_function());
        let (id, name, args) = delta.tool_calls[0].clone().to_function().unwrap();
        assert_eq!(id.unwrap(), "call_DdmO9pD3xa9XTPNJ32zg2hcA");
        assert_eq!(name, "get_temperature");
        assert_eq!(args, "{\"location\":\"Paris, France\"}");
        let (id, name, args) = delta.tool_calls[1].clone().to_function().unwrap();
        assert_eq!(id.unwrap(), "call_ABCDEF1234");
        assert_eq!(name, "get_wind_speed");
        assert_eq!(args, "{\"location\":\"Dubai, Qatar\"}");
    }
}

#[cfg(test)]
mod api_tests {
    use std::sync::LazyLock;

    use futures::StreamExt;

    use crate::{
        debug,
        model::{InferenceConfig, LangModelInference as _, StreamAPILangModel, api::APIProvider},
        to_value,
        value::{Delta, FinishReason, Message, MessageDelta, Part, Role, ToolDescBuilder},
    };

    static XAI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("XAI_API_KEY")
            .expect("Environment variable 'XAI_API_KEY' is required for the tests.")
    });

    #[tokio::test]
    async fn infer_simple_chat() {
        let mut model = StreamAPILangModel::new(APIProvider::XAI, "grok-4-0709", *XAI_API_KEY);

        let msgs = vec![
            Message::new(Role::System).with_contents([Part::text("You are a helpful assistant.")]),
            Message::new(Role::User).with_contents([Part::text("Hi what's your name?")]),
        ];
        let mut assistant_msg = MessageDelta::new();
        let mut strm = model.infer(msgs, Vec::new(), InferenceConfig::default());
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop()));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        let mut model = StreamAPILangModel::new(APIProvider::XAI, "grok-4-0709", *XAI_API_KEY);
        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
        ];
        let mut assistant_msg = MessageDelta::default();
        let mut strm = model.infer(msgs, tools, InferenceConfig::default());
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::ToolCall()));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!(
                "{:?}",
                message.tool_calls.first().and_then(|f| f.as_function())
            );
            message.tool_calls.len() > 0
                && message
                    .tool_calls
                    .first()
                    .and_then(|f| f.as_function())
                    .map(|f| f.1 == "temperature")
                    .unwrap_or(false)
        }));
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_response() {
        let mut model = StreamAPILangModel::new(APIProvider::XAI, "grok-4-0709", *XAI_API_KEY);
        let tools = vec![
            ToolDescBuilder::new("temperature")
                .description("Get current temperature")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                        "unit": {"type": "string", "description": "The unit of temperature", "enum": ["celsius", "fahrenheit"]}
                    }
                })).build(),
        ];
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "call_46035067",
                "temperature",
                to_value!({"location": "Dubai", "unit": "fahrenheit"}),
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function_with_id(
                "call_48738904",
                "temperature",
                to_value!({"location": "Dubai", "unit": "celsius"}),
            )]),
            Message::new(Role::Tool)
                .with_id("call_46035067")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
            Message::new(Role::Tool)
                .with_id("call_48738904")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 30, "unit": "celsius"}),
                }]),
        ];
        let mut strm = model.infer(msgs, tools, InferenceConfig::default());
        let mut assistant_msg = MessageDelta::default();
        let mut finish_reason = None;
        while let Some(output_opt) = strm.next().await {
            let output = output_opt.unwrap();
            assistant_msg = assistant_msg.aggregate(output.delta).unwrap();
            finish_reason = output.finish_reason;
        }
        assert_eq!(finish_reason, Some(FinishReason::Stop()));
        assert!(assistant_msg.finish().is_ok_and(|message| {
            debug!("{:?}", message.contents.first().and_then(|c| c.as_text()));
            message.contents.len() > 0
        }));
    }
}
