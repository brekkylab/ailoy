use base64::Engine;
use indexmap::IndexMap;

use crate::{
    model::{ServerEvent, ThinkEffort, api::RequestConfig},
    to_value,
    value::{
        FinishReason, Marshal, Marshaled, Message, MessageDelta, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, Role, ToolDesc, Unmarshal, Unmarshaled, Value,
    },
};

#[derive(Clone, Debug, Default)]
struct GeminiMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => {
                to_value!({"text": text})
            }
            Part::Function {
                id,
                f: PartFunction { name, args },
            } => {
                let mut value = to_value!({"functionCall": {"name": name, "args": args.clone()}});
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
            Part::Value { value } => value.to_owned(),
        }
    };

    if msg.role == Role::Tool {
        return to_value!(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": msg.id.clone().expect("Tool call id must exist."),
                            "response": {
                                "result": part_to_value(&msg.contents[0])
                            }
                        }
                    }
                ]
            }
        );
    }

    // Collecting contents
    let mut parts = Vec::<Value>::new();
    if !msg.thinking.is_empty() && include_thinking {
        parts.push(to_value!({"text": msg.thinking.clone(), "thought": true}));
    }
    parts.extend(msg.contents.iter().map(part_to_value));
    parts.extend(msg.tool_calls.iter().map(part_to_value));

    // Final message object with role and collected parts
    to_value!({"role": msg.role.to_string(), "parts": parts})
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
                "name": &item.name,
                "description": desc,
                "parameters": item.parameters.clone()
            })
        } else {
            to_value!({
                "name": &item.name,
                "parameters": item.parameters.clone()
            })
        }
    }
}

impl Marshal<RequestConfig> for GeminiMarshal {
    fn marshal(&mut self, config: &RequestConfig) -> Value {
        let (include_thoughts, thinking_budget) = if let Some(model) = &config.model
            && matches!(
                model.as_str(),
                "gemini-2.5-pro" | "gemini-2.5-flash" | "gemini-2.5-flash-lite"
            ) {
            match &config.think_effort {
                ThinkEffort::Disable => match model.as_str() {
                    "gemini-2.5-pro" => (false, 128),
                    "gemini-2.5-flash" => (false, 0),
                    "gemini-2.5-flash-lite" => (false, 0),
                    _ => unreachable!(""),
                },
                ThinkEffort::Enable => (true, -1),
                ThinkEffort::Low => match model.as_str() {
                    "gemini-2.5-pro" => (true, 1024),
                    "gemini-2.5-flash" => (true, 1024),
                    "gemini-2.5-flash-lite" => (true, 1024),
                    _ => unreachable!(""),
                },
                ThinkEffort::Medium => match model.as_str() {
                    "gemini-2.5-pro" => (true, 8192),
                    "gemini-2.5-flash" => (true, 8192),
                    "gemini-2.5-flash-lite" => (true, 8192),
                    _ => unreachable!(""),
                },
                ThinkEffort::High => match model.as_str() {
                    "gemini-2.5-pro" => (true, 24576),
                    "gemini-2.5-flash" => (true, 24576),
                    "gemini-2.5-flash-lite" => (true, 24576),
                    _ => unreachable!(""),
                },
            }
        } else {
            (false, 0)
        };

        let system_instruction = if let Some(system_message) = &config.system_message {
            to_value!({
                "parts": [{"text": system_message}]
            })
        } else {
            Value::Null
        };

        let thinking_config = to_value!({"includeThoughts": include_thoughts, "thinkingBudget": thinking_budget as i64});

        let mut generation_config = to_value!({
            "thinkingConfig": thinking_config,
        });

        if let Some(max_tokens) = config.max_tokens {
            generation_config.as_object_mut().unwrap().insert(
                "max_output_tokens".into(),
                Value::Integer(max_tokens.into()),
            );
        }
        if let Some(temperature) = config.temperature {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("temperature".into(), Value::Float(temperature.into()));
        }
        if let Some(top_p) = config.top_p {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("top_p".into(), Value::Float(top_p.into()));
        }

        to_value!({"system_instruction": system_instruction, "generationConfig": generation_config})
    }
}

#[derive(Clone, Debug, Default)]
struct GeminiUnmarshal;

impl Unmarshal<MessageDelta> for GeminiUnmarshal {
    fn unmarshal(&mut self, val: Value) -> Result<MessageDelta, String> {
        let mut rv = MessageDelta::default();

        let content: &IndexMap<String, Value> = val
            .pointer_as::<IndexMap<String, Value>>("/content")
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
                            rv.thinking = text.into();
                        } else {
                            rv.contents.push(PartDelta::Text { text: text.into() });
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
                        let args = match tool_call_obj.get("args") {
                            Some(args) => args.to_owned(),
                            None => Value::Null,
                        };
                        rv.tool_calls.push(PartDelta::Function {
                            id: None,
                            f: PartDeltaFunction::WithParsedArgs { name, args },
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

pub(super) fn make_request(
    url: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDesc>,
    config: RequestConfig,
) -> reqwest::RequestBuilder {
    let mut body = serde_json::json!(&Marshaled::<_, GeminiMarshal>::new(&config));

    body["contents"] = serde_json::json!(&Marshaled::<_, GeminiMarshal>::new(&msgs));
    if !tools.is_empty() {
        body["tools"] = serde_json::json!(
            {
                "functionDeclarations": tools
                    .iter()
                    .map(|v| Marshaled::<_, GeminiMarshal>::new(v))
                    .collect::<Vec<_>>()
            }
        );
    };

    // let model = model_name;
    let model = config.model.unwrap();
    let generate_method = if config.stream {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };
    let url = format!("{}/{}:{}", url, model, generate_method);

    reqwest::Client::new()
        .request(reqwest::Method::POST, url)
        .header("x-goog-api-key", api_key)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
}

pub(super) fn handle_event(evt: ServerEvent) -> MessageOutput {
    let Ok(j) = serde_json::from_str::<serde_json::Value>(&evt.data) else {
        return MessageOutput::default();
    };

    let Some(candidate) = j.pointer("/candidates/0") else {
        return MessageOutput::default();
    };

    let finish_reason = candidate
        .pointer("/finishReason")
        .and_then(|v| v.as_str())
        .map(|reason| match reason {
            "STOP" => FinishReason::Stop(),
            "MAX_TOKENS" => FinishReason::Length(),
            reason => FinishReason::Refusal(reason.to_owned()),
        });

    let delta = match finish_reason {
        Some(FinishReason::Refusal(_)) => MessageDelta::default(),
        _ => serde_json::from_value::<Unmarshaled<_, GeminiUnmarshal>>(candidate.clone())
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
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","parts":[{"text":"Explain me about Riemann hypothesis."},{"text":"How cold brew is different from the normal coffee?"}]}"#
        );
    }

    #[test]
    pub fn serialize_messages_with_thinkings() {
        let msgs = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Hello there."), Part::text("How are you?")]),
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
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","parts":[{"text":"Hello there."},{"text":"How are you?"}]},{"role":"assistant","parts":[{"text":"I'm fine, thank you. And you?"}]},{"role":"user","parts":[{"text":"I'm okay."}]},{"role":"assistant","parts":[{"text":"This is thinking text would be remaining.","thought":true},{"text":"Is there anything I can help with?"}]}]"#
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
            r#"{"role":"assistant","parts":[{"functionCall":{"name":"temperature","args":{"unit":"celsius"}}},{"functionCall":{"name":"temperature","args":{"unit":"fahrenheit"}}}]}"#
        );
    }

    #[test]
    pub fn serialize_tool_response() {
        let msgs = vec![
            Message::new(Role::Tool)
                .with_id("temperature")
                .with_contents(vec![Part::Value {
                    value: to_value!({"temperature": 30, "unit": "celsius"}),
                }]),
            Message::new(Role::Tool)
                .with_id("temperature")
                .with_contents(vec![Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
        ];
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msgs);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"[{"role":"user","parts":[{"functionResponse":{"name":"temperature","response":{"result":{"temperature":30,"unit":"celsius"}}}}]},{"role":"user","parts":[{"functionResponse":{"name":"temperature","response":{"result":{"temperature":86,"unit":"fahrenheit"}}}}]}]"#
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
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"role":"user","parts":[{"text":"What you can see in this image?"},{"inline_data":{"mime_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII="}}]}"#,
        );
    }

    #[test]
    pub fn serialize_config() {
        let config = RequestConfig {
            model: Some("gemini-2.5-pro".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"system_instruction":{"parts":[{"text":"You are a helpful assistant."}]},"generationConfig":{"thinkingConfig":{"includeThoughts":true,"thinkingBudget":-1},"max_output_tokens":1024,"temperature":0.6,"top_p":0.9}}"#
        );

        let config = RequestConfig {
            model: Some("gemini-2.0-flash".to_owned()),
            system_message: Some("You are a helpful assistant.".to_owned()),
            stream: true,
            think_effort: ThinkEffort::Enable,
            temperature: Some(0.6),
            top_p: Some(0.9),
            max_tokens: Some(1024),
        };
        let marshaled = Marshaled::<_, GeminiMarshal>::new(&config);
        assert_eq!(
            serde_json::to_string(&marshaled).unwrap(),
            r#"{"system_instruction":{"parts":[{"text":"You are a helpful assistant."}]},"generationConfig":{"thinkingConfig":{"includeThoughts":false,"thinkingBudget":0},"max_output_tokens":1024,"temperature":0.6,"top_p":0.9}}"#
        );
    }

    #[test]
    pub fn deserialize_text() {
        let inputs = [r#"{"content":{"parts":[{"text":"Hello world!"}],"role":"model"}}"#];
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
            r#"{"content": {"parts": [{"text": "Hello"}],"role": "model"}}"#,
            r#"{"content": {"parts": [{"text": " world!"}],"role": "model"}}"#,
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
    pub fn deserialize_text_with_thinking() {
        let inputs = [
            r#"{"content":{"parts":[{"text":"**Answering a simple question**\n\nUser is saying hello.","thought":true},{"text":"Hello world!"}],"role":"model"}}"#,
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
            delta.thinking,
            "**Answering a simple question**\n\nUser is saying hello."
        );
        assert_eq!(delta.contents.len(), 1);
        let content = delta.contents.pop().unwrap();
        assert_eq!(content.to_text().unwrap(), "Hello world!");
    }

    #[test]
    pub fn deserialize_text_with_thinking_stream() {
        let inputs = [
            r#"{"content":{"parts":[{"text":"**Answering a simple question**\n\n","thought": true}],"role": "model"}}"#,
            r#"{"content":{"parts":[{"text":"User is saying hello.","thought": true}],"role": "model"}}"#,
            r#"{"content": {"parts": [{"text": "Hello"}],"role": "model"}}"#,
            r#"{"content": {"parts": [{"text": " world!"}],"role": "model"}}"#,
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
            r#"{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris, France"}}}],"role":"model"}}"#,
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
        let (_, name, args) = tool_call.to_parsed_function().unwrap();
        assert_eq!(name, "get_weather");
        assert_eq!(
            serde_json::to_string(&args).unwrap(),
            "{\"location\":\"Paris, France\"}"
        );
    }
}

#[cfg(test)]
mod api_tests {
    use std::sync::LazyLock;

    use ailoy_macros::multi_platform_test;
    use futures::StreamExt;

    use super::*;
    use crate::{
        debug,
        model::{
            InferenceConfig, LangModelInference as _, StreamAPILangModel, api::APISpecification,
        },
        value::{Delta, ToolDescBuilder},
    };

    static GEMINI_API_KEY: LazyLock<&'static str> = LazyLock::new(|| {
        option_env!("GEMINI_API_KEY")
            .expect("Environment variable 'GEMINI_API_KEY' is required for the tests.")
    });

    #[multi_platform_test]
    async fn infer_simple_chat() {
        let mut model = StreamAPILangModel::new(
            APISpecification::Gemini,
            "gemini-2.5-flash-lite",
            *GEMINI_API_KEY,
        );

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
        assert_eq!(finish_reason, Some(FinishReason::Stop()));
    }

    #[multi_platform_test]
    async fn infer_tool_call() {
        let mut model = StreamAPILangModel::new(
            APISpecification::Gemini,
            "gemini-2.5-flash",
            *GEMINI_API_KEY,
        );
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
            Message::new(Role::System).with_contents([Part::text("You are a helpful assistant.")]),
            Message::new(Role::User)
                .with_contents([Part::text("How much hot currently in Dubai?")]),
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

    #[multi_platform_test]
    async fn infer_tool_response() {
        let mut model = StreamAPILangModel::new(
            APISpecification::Gemini,
            "gemini-2.5-flash",
            *GEMINI_API_KEY,
        );
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
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "temperature",
                to_value!({"location": "Dubai", "unit": "fahrenheit"}),
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "temperature",
                to_value!({"location": "Dubai", "unit": "celsius"}),
            )]),
            Message::new(Role::Tool)
                .with_id("temperature")
                .with_contents([Part::Value {
                    value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
                }]),
            Message::new(Role::Tool)
                .with_id("temperature")
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
