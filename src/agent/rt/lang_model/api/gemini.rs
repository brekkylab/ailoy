use anyhow::bail;
use url::Url;

use super::super::{LangModelAPISchema, LangModelProvider};
use crate::{
    agent::rt::lang_model::LangModelRequest,
    datatype::Value,
    message::{
        FinishReason, Marshal, Message, MessageDelta, MessageDeltaOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, ToolDesc, Unmarshal,
    },
    to_value,
};

impl LangModelProvider {
    pub fn gemini(api_key: String) -> Self {
        Self::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: Some(api_key),
            max_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeminiMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => {
                to_value!({"text": text})
            }
            Part::Function {
                function: PartFunction { name, arguments },
                ..
            } => {
                to_value!({"functionCall": {"name": name, "args": arguments.clone()}})
            }
            Part::Image { image } => {
                let (mime_type, b64) = match image {
                    PartImage::Embedded { mime_type, data } => (mime_type.clone(), data.base64()),
                    PartImage::Url { url } => {
                        // If url is a form of base64 data uri, use the data part as inline data.
                        // Otherwise, Gemini does not support public url image inputs.
                        let re = fancy_regex::Regex::new(
                            r"^data:([a-z]+/[a-z0-9-+.]+(;[a-z-]+=[a-z0-9-]+)?)?;base64,(.*)$",
                        )
                        .unwrap();
                        if let Some(captures) = re.captures(url).unwrap() {
                            let mime = captures
                                .get(1)
                                .map(|m| m.as_str().to_string())
                                .unwrap_or_else(|| "image/png".to_string());
                            let data = captures.get(3).map(|m| m.as_str().to_string()).unwrap();
                            (mime, data)
                        } else {
                            panic!("Gemini does not support image url inputs")
                        }
                    }
                };
                to_value!({"inline_data": {"mime_type": mime_type, "data": b64}})
            }
            Part::Value { value } => value.to_owned(),
        }
    };

    if msg.role == Role::Tool {
        let tool_call_id = msg.id.clone().expect("Tool call id must exist.");
        let (tool_name, _) = tool_call_id.split_once("/").unwrap();
        return to_value!(
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": tool_name,
                            "response": {
                                "result": part_to_value(&msg.contents[0])
                            }
                        }
                    }
                ]
            }
        );
    }

    // Role
    let role: String = if msg.role == Role::Assistant {
        "model".into()
    } else if msg.role == Role::User {
        "user".into()
    } else {
        panic!("Gemini accepts \"model\" and \"user\" role only.")
    };

    // Collecting contents
    let mut parts = Vec::<Value>::new();
    if let Some(thinking) = &msg.thinking
        && !thinking.is_empty()
        && include_thinking
    {
        parts.push(to_value!({"text": thinking, "thought": true}));
    }
    parts.extend(msg.contents.iter().map(part_to_value));
    parts.extend(
        msg.tool_calls
            .clone()
            .unwrap_or(vec![])
            .iter()
            .map(part_to_value),
    );

    // Final message object with role and collected parts
    to_value!({"role": role, "parts": parts})
}

fn marshal_messages(msgs: &[Message]) -> Value {
    let last_user_index = msgs
        .iter()
        .rposition(|m| m.role == Role::User || m.role == Role::Tool)
        .unwrap_or_else(|| msgs.len());
    Value::Array(
        msgs.iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .map(|(i, msg)| marshal_message(msg, i > last_user_index))
            .collect::<Vec<_>>(),
    )
}

impl Marshal<Message> for GeminiMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        marshal_message(msg, true)
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

impl Marshal<LangModelRequest<'_>> for GeminiMarshal {
    fn marshal(&mut self, req: &LangModelRequest<'_>) -> Value {
        // Extract system instruction from system message if present
        let system_instruction = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.contents.first())
            .and_then(|p| {
                if let Part::Text { text } = p {
                    Some(to_value!({"parts": [{"text": text}]}))
                } else {
                    None
                }
            });

        let contents = marshal_messages(&req.messages);

        let tools = if !req.tools.is_empty() {
            let declarations = req
                .tools
                .iter()
                .map(|t| self.marshal(t))
                .collect::<Vec<_>>();
            to_value!({"functionDeclarations": declarations})
        } else {
            Value::Null
        };

        let url = format!("{}{}:generateContent", req.url, req.model);

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = req.api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("x-goog-api-key".into(), api_key.into());
        }

        let mut body = to_value!({
            "contents": contents,
        });
        if let Some(system_instruction) = system_instruction {
            body.as_object_mut()
                .unwrap()
                .insert("system_instruction".into(), system_instruction);
        }
        if !tools.is_null() {
            body.as_object_mut().unwrap().insert("tools".into(), tools);
        }
        if let Some(max_tokens) = req.max_tokens {
            body.as_object_mut().unwrap().insert(
                "generationConfig".into(),
                to_value!({"maxOutputTokens": max_tokens as i64}),
            );
        }

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeminiUnmarshal;

impl Unmarshal<MessageDeltaOutput> for GeminiUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let candidate = val
            .pointer("/candidates/0")
            .ok_or_else(|| anyhow::anyhow!("Missing candidates[0] in response"))?
            .to_owned();

        let mut finish_reason = candidate
            .pointer("/finishReason")
            .and_then(|v| v.as_str())
            .map(|reason| match reason {
                "STOP" => FinishReason::Stop {},
                "MAX_TOKENS" => FinishReason::Length {},
                reason => FinishReason::Refusal {
                    reason: reason.to_owned(),
                },
            });

        let delta = match &finish_reason {
            Some(FinishReason::Refusal { .. }) => MessageDelta::default(),
            _ => parse_candidate_content(&candidate)?,
        };

        // Gemini always returns "STOP" even for tool call responses,
        // so adjust the finish reason when tool calls exist.
        if !delta.tool_calls.is_empty()
            && finish_reason
                .clone()
                .is_some_and(|reason| matches!(reason, FinishReason::Stop {}))
        {
            finish_reason = Some(FinishReason::ToolCall {});
        }

        Ok(MessageDeltaOutput {
            delta,
            finish_reason,
        })
    }
}

fn parse_candidate_content(candidate: &Value) -> anyhow::Result<MessageDelta> {
    let mut rv = MessageDelta::default();

    let content = candidate
        .pointer("/content")
        .ok_or_else(|| anyhow::anyhow!("Missing content in candidate"))?;

    // Parse role
    if let Some(r) = content.pointer("/role") {
        let s = r
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Role should be a string"))?;
        let v = match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "model" => Role::Assistant,
            "tool" => Role::Tool,
            other => bail!("Unknown role: {other}"),
        };
        rv.role = Some(v);
    }

    // Parse parts
    if let Some(parts) = content.pointer("/parts")
        && !parts.is_null()
    {
        if let Some(parts) = parts.as_array() {
            for part in parts {
                let Some(part) = part.as_object() else {
                    bail!("Invalid part")
                };
                let thought = part
                    .get("thought")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false);
                if let Some(text) = part.get("text") {
                    let Some(text) = text.as_str() else {
                        bail!("Invalid content part")
                    };
                    if thought {
                        rv.thinking = Some(text.into());
                    } else {
                        rv.contents.push(PartDelta::Text { text: text.into() });
                    }
                } else if let Some(tool_call_obj) = part.get("functionCall") {
                    let Some(tool_call_obj) = tool_call_obj.as_object() else {
                        bail!("Invalid functionCall object");
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
                    rv.tool_calls.push(PartDelta::Function {
                        // Generate tool call id with a form of "{tool_name}/{random_id}",
                        // and use {tool_name} part only on Marshal.
                        id: Some(format!(
                            "{}/call-{}",
                            name,
                            &uuid::Uuid::new_v4().to_string()[..8]
                        )),
                        function: PartDeltaFunction::WithParsedArgs { name, arguments },
                    });
                } else {
                    bail!("Invalid part");
                }
            }
        } else {
            bail!("Invalid parts");
        }
    }

    Ok(rv)
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::{
        agent::{LangModelAPISchema, LangModelProvider, LangModelRuntime},
        message::{FinishReason, Message, Part, Role, ToolDesc},
    };

    fn with_req<F, R>(model: &str, max_tokens: Option<u64>, f: F) -> R
    where
        F: FnOnce(&LangModelRequest<'_>) -> R,
    {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap();
        let api_key: Option<String> = None;
        let req = LangModelRequest {
            model,
            messages: &messages,
            tools: &tools,
            url: &url,
            api_key: &api_key,
            max_tokens,
        };
        f(&req)
    }

    #[test]
    fn test_marshal_max_output_tokens_set() {
        with_req("gemini-2.5-flash-lite", Some(2048), |req| {
            let val = GeminiMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            let gen_config = body.as_object().unwrap().get("generationConfig").unwrap();
            let max_tokens = gen_config
                .as_object()
                .unwrap()
                .get("maxOutputTokens")
                .unwrap();
            assert_eq!(max_tokens.as_integer().unwrap(), 2048);
        });
    }

    #[test]
    fn test_marshal_max_output_tokens_absent_when_none() {
        with_req("gemini-2.5-flash-lite", None, |req| {
            let val = GeminiMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(body.as_object().unwrap().get("generationConfig").is_none());
        });
    }

    /// Verifies that max_tokens is respected by the Gemini API (finishReason: MAX_TOKENS).
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set in .env");

        let lm = LangModelRuntime::new(
            "gemini-2.5-flash-lite".to_string(),
            LangModelProvider::API {
                schema: LangModelAPISchema::Gemini,
                url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/")
                    .unwrap(),
                api_key: Some(api_key),
                max_tokens: Some(5),
            },
        );
        let messages = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Tell me a long story about a dragon.")]),
        ];
        let tools: Vec<ToolDesc> = vec![];

        let resp = lm.run(&messages, &tools).await.unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length {});
    }
}

// #[cfg(test)]
// mod dialect_tests {
//     use super::*;
//     use crate::{
//         datatype::Bytes,
//         message::{Marshaled, Message, Role},
//     };

//     #[test]
//     pub fn serialize_text() {
//         let msg = Message::new(Role::User).with_contents([
//             Part::text("Explain me about Riemann hypothesis."),
//             Part::text("How cold brew is different from the normal coffee?"),
//         ]);
//         let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"{"role":"user","parts":[{"text":"Explain me about Riemann hypothesis."},{"text":"How cold brew is different from the normal coffee?"}]}"#
//         );
//     }

//     #[test]
//     pub fn serialize_messages_with_thinkings() {
//         let msgs = vec![
//             Message::new(Role::User)
//                 .with_contents([Part::text("Hello there."), Part::text("How are you?")]),
//             Message::new(Role::Assistant)
//                 .with_thinking_signature("This is thinking text would be vanished.", "")
//                 .with_contents([Part::text("I'm fine, thank you. And you?")]),
//             Message::new(Role::User).with_contents([Part::text("I'm okay.")]),
//             Message::new(Role::Assistant)
//                 .with_thinking_signature(
//                     "This is thinking text would be remaining.",
//                     "Ev4MCkYIBxgCKkDl5A",
//                 )
//                 .with_contents([Part::text("Is there anything I can help with?")]),
//         ];
//         // Use marshal_messages directly to test position-aware thinking inclusion.
//         let marshaled = marshal_messages(&msgs);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","parts":[{"text":"Hello there."},{"text":"How are you?"}]},{"role":"model","parts":[{"text":"I'm fine, thank you. And you?"}]},{"role":"user","parts":[{"text":"I'm okay."}]},{"role":"model","parts":[{"text":"This is thinking text would be remaining.","thought":true},{"text":"Is there anything I can help with?"}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_function() {
//         let msg = Message::new(Role::Assistant).with_tool_calls([
//             Part::function("temperature", Value::object([("unit", "celsius")])),
//             Part::function("temperature", Value::object([("unit", "fahrenheit")])),
//         ]);
//         let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"{"role":"model","parts":[{"functionCall":{"name":"temperature","args":{"unit":"celsius"}}},{"functionCall":{"name":"temperature","args":{"unit":"fahrenheit"}}}]}"#
//         );
//     }

//     #[test]
//     pub fn serialize_tool_response() {
//         let msgs = vec![
//             Message::new(Role::Tool)
//                 .with_id("temperature/call-1")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 30, "unit": "celsius"}),
//                 }]),
//             Message::new(Role::Tool)
//                 .with_id("temperature/call-2")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
//                 }]),
//         ];
//         let marshaled = Marshaled::<_, GeminiMarshal>::new(&msgs);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","parts":[{"functionResponse":{"name":"temperature","response":{"result":{"temperature":30,"unit":"celsius"}}}}]},{"role":"user","parts":[{"functionResponse":{"name":"temperature","response":{"result":{"temperature":86,"unit":"fahrenheit"}}}}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_image() {
//         use base64::prelude::*;

//         let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAD0lEQVR4nGPh4uJikYNgAANQAI8386KKAAAAAElFTkSuQmCC";
//         let png_bytes = BASE64_STANDARD.decode(png_base64).unwrap();
//         let msg = Message::new(Role::User).with_contents([
//             Part::text("What you can see in this image?"),
//             Part::image_embedded("image/png".to_owned(), Bytes::from(png_bytes)).unwrap(),
//         ]);
//         let marshaled = Marshaled::<_, GeminiMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"{"role":"user","parts":[{"text":"What you can see in this image?"},{"inline_data":{"mime_type":"image/png","data":""#.to_owned()
//                 + png_base64
//                 + r#""}}]}"#,
//         );
//     }

//     #[test]
//     pub fn deserialize_text() {
//         let input =
//             r#"{"candidates":[{"content":{"parts":[{"text":"Hello world!"}],"role":"model"},"finishReason":"STOP"}]}"#;
//         let mut u = GeminiUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Stop {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.role, Some(Role::Assistant));
//         assert_eq!(delta.contents.len(), 1);
//         let content = delta.contents.pop().unwrap();
//         assert_eq!(content.to_text().unwrap(), "Hello world!");
//     }

//     #[test]
//     pub fn deserialize_text_with_thinking() {
//         let input = r#"{"candidates":[{"content":{"parts":[{"text":"**Answering a simple question**\n\nUser is saying hello.","thought":true},{"text":"Hello world!"}],"role":"model"},"finishReason":"STOP"}]}"#;
//         let mut u = GeminiUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Stop {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.role, Some(Role::Assistant));
//         assert_eq!(
//             delta.thinking,
//             Some("**Answering a simple question**\n\nUser is saying hello.".into())
//         );
//         assert_eq!(delta.contents.len(), 1);
//         let content = delta.contents.pop().unwrap();
//         assert_eq!(content.to_text().unwrap(), "Hello world!");
//     }

//     #[test]
//     pub fn deserialize_tool_call() {
//         let input = r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris, France"}}}],"role":"model"},"finishReason":"STOP"}]}"#;
//         let mut u = GeminiUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::ToolCall {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.tool_calls.len(), 1);
//         let tool_call = delta.tool_calls.pop().unwrap();
//         let (_, name, args) = tool_call.to_parsed_function().unwrap();
//         assert_eq!(name, "get_weather");
//         assert_eq!(
//             serde_json::to_string(&args).unwrap(),
//             "{\"location\":\"Paris, France\"}"
//         );
//     }
// }
