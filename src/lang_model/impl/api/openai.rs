use url::Url;

use crate::{
    datatype::Value,
    lang_model::{LangModelAPISchema, LangModelProvider, LangModelProviderElem, LangModelRequest},
    message::{
        FinishReason, Marshal, Message, MessageDelta, MessageDeltaOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage, Unmarshal,
    },
    to_value,
    tool::ToolDesc,
};

impl LangModelProvider {
    pub fn openai(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
            api_key: Some(api_key),
            max_tokens: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

fn marshal_message(msg: &Message, include_thinking: bool) -> Vec<Value> {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => {
                let r#type = if msg.role == Role::Assistant {
                    "output_text"
                } else {
                    "input_text"
                };
                to_value!({"type": r#type, "text": text})
            }
            Part::Function {
                id,
                function: PartFunction { name, arguments },
            } => {
                let arguments_string = serde_json::to_string(arguments).unwrap();
                to_value!({"type": "function_call", "call_id": id, "name": name, "arguments": arguments_string})
            }
            Part::Value { value } => {
                to_value!(serde_json::to_string(value).unwrap())
            }
            Part::Image { image } => {
                let url = match image {
                    PartImage::Embedded { mime_type, data } => {
                        format!("data:{};base64,{}", mime_type, data.base64())
                    }
                    PartImage::Url { url } => url.clone(),
                };
                to_value!({"type": "input_image", "image_url": url})
            }
        }
    };

    if msg.role == Role::Tool {
        let output: Vec<Value> = msg
            .contents
            .iter()
            .filter_map(|p| match p {
                Part::Text { .. } | Part::Image { .. } => Some(part_to_value(p)),
                Part::Value { value } => {
                    let text = match value {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    Some(to_value!({"type": "input_text", "text": text}))
                }
                _ => None,
            })
            .collect();
        return vec![to_value!(
            {
                "type": "function_call_output",
                "call_id": msg.id.clone().expect("Tool call id must exist."),
                "output": output
            }
        )];
    }

    let mut rv = Vec::<Value>::new();
    if let Some(thinking) = &msg.thinking
        && !thinking.is_empty()
        && include_thinking
    {
        rv.push(
            to_value!({"type": "reasoning", "summary": [{"type": "summary_text", "text": thinking}]}),
        );
    }
    if !msg.contents.is_empty() {
        rv.push(to_value!({"role": msg.role.to_string(), "content": msg.contents.iter().map(part_to_value).collect::<Vec<_>>()}));
    }
    rv.extend(
        msg.tool_calls
            .clone()
            .unwrap_or(vec![])
            .iter()
            .map(part_to_value),
    );
    rv
}

fn marshal_messages(msgs: &[Message]) -> Value {
    let last_user_index = msgs
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or_else(|| msgs.len());
    Value::Array(
        msgs.iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .flat_map(|(i, msg)| marshal_message(msg, i > last_user_index))
            .collect::<Vec<_>>(),
    )
}

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&mut self, msg: &Message) -> Value {
        to_value!(marshal_message(msg, true))
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

impl Marshal<LangModelRequest<'_>> for OpenAIMarshal {
    fn marshal(&mut self, req: &LangModelRequest<'_>) -> Value {
        // Extract system instruction from system message if present
        let instructions = req
            .messages
            .iter()
            .find(|m| m.role == Role::System)
            .and_then(|m| m.contents.first())
            .and_then(|p| {
                if let Part::Text { text } = p {
                    Some(Value::from(text.as_str()))
                } else {
                    None
                }
            });

        let input = marshal_messages(&req.messages);

        let tools = if !req.tools.is_empty() {
            Value::Array(req.tools.iter().map(|t| self.marshal(t)).collect())
        } else {
            Value::Null
        };

        let url = req.url.to_string();

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = req.api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("Authorization".into(), format!("Bearer {}", api_key).into());
        }

        let mut body = to_value!({
            "model": req.model,
            "input": input,
        });
        if let Some(instructions) = instructions {
            body.as_object_mut()
                .unwrap()
                .insert("instructions".into(), instructions);
        }
        if !tools.is_null() {
            body.as_object_mut()
                .unwrap()
                .insert("tool_choice".into(), to_value!("auto"));
            body.as_object_mut().unwrap().insert("tools".into(), tools);
        }
        if let Some(max_tokens) = req.max_tokens {
            body.as_object_mut()
                .unwrap()
                .insert("max_output_tokens".into(), (max_tokens as i64).into());
        }

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIUnmarshal;

impl Unmarshal<MessageDeltaOutput> for OpenAIUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let root = val
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Root should be an object"))?;

        // Parse finish reason from status
        let status = root
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed");

        let mut finish_reason = match status {
            "completed" => Some(FinishReason::Stop {}),
            "incomplete" => {
                let reason = root
                    .get("incomplete_details")
                    .and_then(|d| d.pointer("/reason"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                Some(match reason {
                    "max_output_tokens" => FinishReason::Length {},
                    "content_filter" => FinishReason::Refusal {
                        reason: "Model output violated OpenAI's safety policy.".to_owned(),
                    },
                    other => FinishReason::Refusal {
                        reason: format!("reason: {}", other),
                    },
                })
            }
            _ => None,
        };

        // Parse output items
        let mut delta = MessageDelta::default();

        if let Some(output) = root.get("output")
            && let Some(items) = output.as_array()
        {
            for item in items {
                let Some(item_obj) = item.as_object() else {
                    continue;
                };
                let ty = item_obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match ty {
                    "message" => {
                        // Parse role
                        if delta.role.is_none() {
                            let role = item_obj
                                .get("role")
                                .and_then(|v| v.as_str())
                                .map(|s| match s {
                                    "assistant" => Role::Assistant,
                                    "user" => Role::User,
                                    _ => Role::Assistant,
                                })
                                .unwrap_or(Role::Assistant);
                            delta.role = Some(role);
                        }
                        // Parse content parts
                        if let Some(content) = item_obj.get("content")
                            && let Some(parts) = content.as_array()
                        {
                            for part in parts {
                                let part_ty =
                                    part.pointer("/type").and_then(|v| v.as_str()).unwrap_or("");
                                if part_ty == "output_text" || part_ty == "text" {
                                    if let Some(text) =
                                        part.pointer("/text").and_then(|v| v.as_str())
                                    {
                                        delta.contents.push(PartDelta::Text {
                                            text: text.to_owned(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                    "function_call" => {
                        if delta.role.is_none() {
                            delta.role = Some(Role::Assistant);
                        }
                        let id = item_obj
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_owned());
                        let name = item_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let arguments = item_obj
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        delta.tool_calls.push(PartDelta::Function {
                            id,
                            function: PartDeltaFunction::WithStringArgs { name, arguments },
                        });
                    }
                    "reasoning" => {
                        // Parse summary text as thinking
                        if let Some(summary) = item_obj.get("summary")
                            && let Some(parts) = summary.as_array()
                        {
                            for part in parts {
                                if let Some(text) = part.pointer("/text").and_then(|v| v.as_str()) {
                                    delta.thinking =
                                        Some(delta.thinking.unwrap_or_default() + text);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Adjust finish reason for tool calls
        if !delta.tool_calls.is_empty()
            && finish_reason
                .clone()
                .is_some_and(|r| matches!(r, FinishReason::Stop {}))
        {
            finish_reason = Some(FinishReason::ToolCall {});
        }

        // Parse usage (OpenAI Responses API: usage.input_tokens / output_tokens)
        let usage = val
            .as_object()
            .and_then(|r| r.get("usage"))
            .and_then(|u| u.as_object())
            .map(|u| TokenUsage {
                input_tokens: u
                    .get("input_tokens")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as u64,
                output_tokens: u
                    .get("output_tokens")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as u64,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            });

        Ok(MessageDeltaOutput {
            delta,
            finish_reason,
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::{
        datatype::Bytes,
        lang_model::{LangModel, LangModelAPISchema, LangModelProviderElem},
        message::{FinishReason, Message, Part, Role, TokenUsage},
        tool::{ToolDesc, ToolDescBuilder},
    };

    fn with_req<F, R>(model: &str, max_tokens: Option<u64>, f: F) -> R
    where
        F: FnOnce(&LangModelRequest<'_>) -> R,
    {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://api.openai.com/v1/responses").unwrap();
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
    fn test_unmarshal_usage() {
        let response = to_value!({
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 200, "output_tokens": 75}
        });
        let usage = OpenAIUnmarshal::default()
            .unmarshal(response)
            .unwrap()
            .usage;
        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 200,
                output_tokens: 75,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            })
        );
    }

    /// Verifies that function_call_output.output is an array of text/image blocks.
    /// Unsupported part types (e.g. Part::Value) are filtered out.
    #[test]
    fn test_tool_result_content_marshaling() {
        // Part::Text → array with a single {"type":"input_text","text":"..."} block.
        let msg_text = Message::new(Role::Tool)
            .with_id("call_1")
            .with_contents([Part::text("tool output")]);
        let items = marshal_message(&msg_text, false);
        let output = items[0]
            .pointer("/output")
            .expect("output must exist")
            .as_array()
            .expect("output must be an array");
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].pointer("/type").and_then(|v| v.as_str()),
            Some("input_text")
        );
        assert_eq!(
            output[0].pointer("/text").and_then(|v| v.as_str()),
            Some("tool output")
        );

        // Part::Image (embedded) → array with a single {"type":"input_image","image_url":"data:..."} block.
        let img_bytes = crate::datatype::Bytes::from(vec![0xFFu8, 0xD8, 0xFF]);
        let msg_img = Message::new(Role::Tool)
            .with_id("call_2")
            .with_contents([Part::image_embedded("image/jpeg", img_bytes.clone()).unwrap()]);
        let items = marshal_message(&msg_img, false);
        let output = items[0]
            .pointer("/output")
            .expect("output must exist")
            .as_array()
            .expect("output must be an array");
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].pointer("/type").and_then(|v| v.as_str()),
            Some("input_image")
        );
        assert_eq!(
            output[0].pointer("/image_url").and_then(|v| v.as_str()),
            Some(format!("data:image/jpeg;base64,{}", img_bytes.base64()).as_str())
        );

        // Part::Image (url) → array with a single {"type":"input_image","image_url":"https://..."} block.
        let msg_img_url = Message::new(Role::Tool)
            .with_id("call_3")
            .with_contents([Part::image_url("https://example.com/img.png".to_string()).unwrap()]);
        let items = marshal_message(&msg_img_url, false);
        let output = items[0]
            .pointer("/output")
            .expect("output must exist")
            .as_array()
            .expect("output must be an array");
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].pointer("/type").and_then(|v| v.as_str()),
            Some("input_image")
        );
        assert_eq!(
            output[0].pointer("/image_url").and_then(|v| v.as_str()),
            Some("https://example.com/img.png")
        );

        // Part::Value(String) → {"type":"input_text","text":"..."} block; no double-encoding.
        let msg_str = Message::new(Role::Tool)
            .with_id("call_4")
            .with_contents([Part::value(crate::datatype::Value::string(
                "ok".to_string(),
            ))]);
        let items = marshal_message(&msg_str, false);
        let output = items[0]
            .pointer("/output")
            .expect("output must exist")
            .as_array()
            .expect("output must be an array");
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].pointer("/type").and_then(|v| v.as_str()),
            Some("input_text")
        );
        assert_eq!(
            output[0].pointer("/text").and_then(|v| v.as_str()),
            Some("ok"),
            "Part::Value(String) must not be double-encoded"
        );

        // Part::Value(Object) → JSON-encoded as {"type":"input_text","text":"{...}"} block.
        let msg_obj = Message::new(Role::Tool)
            .with_id("call_5")
            .with_contents([Part::value(crate::to_value!({"temp": 25}))]);
        let items = marshal_message(&msg_obj, false);
        let output = items[0]
            .pointer("/output")
            .expect("output must exist")
            .as_array()
            .expect("output must be an array");
        assert_eq!(output.len(), 1);
        assert_eq!(
            output[0].pointer("/type").and_then(|v| v.as_str()),
            Some("input_text")
        );
        assert_eq!(
            output[0].pointer("/text").and_then(|v| v.as_str()),
            Some(r#"{"temp":25}"#),
            "Part::Value(Object) must be JSON-encoded into a text block"
        );
    }

    #[test]
    fn test_marshal_max_output_tokens_set() {
        with_req("gpt-5.4-mini", Some(1024), |req| {
            let val = OpenAIMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            let max_tokens = body.as_object().unwrap().get("max_output_tokens").unwrap();
            assert_eq!(max_tokens.as_integer().unwrap(), 1024);
        });
    }

    #[test]
    fn test_marshal_max_output_tokens_absent_when_none() {
        with_req("gpt-5.4-mini", None, |req| {
            let val = OpenAIMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(body.as_object().unwrap().get("max_output_tokens").is_none());
        });
    }

    /// Verifies that max_tokens is respected by the OpenAI Responses API (incomplete: max_output_tokens).
    /// Note: OpenAI Responses API enforces a minimum of 16 for max_output_tokens.
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = LangModel::new(
            "gpt-5.4-mini".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::OpenAI,
                url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
                api_key: Some(api_key),
                max_tokens: Some(16),
            },
        );
        let messages = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Tell me a long story about a dragon.")]),
        ];
        let tools: Vec<ToolDesc> = vec![];

        let resp = model.run(&messages, &tools).await.unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length {});
    }

    /// Verifies that an image embedded in a Role::Tool message is accepted by the OpenAI
    /// Responses API (output as content array with input_image) and the model can respond.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_tool_result_with_image() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();

        // Fetch a real JPEG image to use as the tool result
        let img_bytes = reqwest::get(
            "https://cdn.britannica.com/60/257460-050-62FF74CB/NVIDIA-Jensen-Huang.jpg",
        )
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();

        let model = LangModel::new(
            "gpt-5.4-mini".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::OpenAI,
                url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
                api_key: Some(api_key),
                max_tokens: None,
            },
        );

        let messages = vec![
            Message::new(Role::User).with_contents([Part::text(
                "Describe the image returned by the file_read tool.",
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "call_test_001",
                "file_read",
                to_value!({"path": "/tmp/test.png"}),
            )]),
            Message::new(Role::Tool)
                .with_id("call_test_001")
                .with_contents([
                    Part::image_embedded("image/jpeg", Bytes::from(img_bytes)).unwrap()
                ]),
        ];
        let tools =
            vec![ToolDescBuilder::new("file_read")
            .description("Read a file and return its contents. Images are returned inline.")
            .parameters(to_value!({
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"]
            }))
            .build()];

        let resp = model.run(&messages, &tools).await.unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop {});
        assert!(
            resp.message.contents.iter().any(|p| p.as_text().is_some()),
            "Expected text response after image tool result"
        );
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
//         let msg = Message::new(Role::User)
//             .with_contents([Part::text("Explain me about Riemann hypothesis.")]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","content":[{"type":"input_text","text":"Explain me about Riemann hypothesis."}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_messages_with_thinkings() {
//         let msgs = vec![
//             Message::new(Role::User).with_contents([Part::text("Hello there.")]),
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
//             r#"[{"role":"user","content":[{"type":"input_text","text":"Hello there."}]},{"role":"assistant","content":[{"type":"output_text","text":"I'm fine, thank you. And you?"}]},{"role":"user","content":[{"type":"input_text","text":"I'm okay."}]},{"type":"reasoning","summary":[{"type":"summary_text","text":"This is thinking text would be remaining."}]},{"role":"assistant","content":[{"type":"output_text","text":"Is there anything I can help with?"}]}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_function() {
//         let msg = Message::new(Role::Assistant).with_tool_calls([
//             Part::function_with_id(
//                 "funcid_123456",
//                 "temperature",
//                 Value::object([("unit", "celsius")]),
//             ),
//             Part::function_with_id(
//                 "funcid_7890ab",
//                 "temperature",
//                 Value::object([("unit", "fahrenheit")]),
//             ),
//         ]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"type":"function_call","call_id":"funcid_123456","name":"temperature","arguments":"{\"unit\":\"celsius\"}"},{"type":"function_call","call_id":"funcid_7890ab","name":"temperature","arguments":"{\"unit\":\"fahrenheit\"}"}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_tool_response() {
//         let msgs = vec![
//             Message::new(Role::Tool)
//                 .with_id("funcid_123456")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 30, "unit": "celsius"}),
//                 }]),
//             Message::new(Role::Tool)
//                 .with_id("funcid_7890ab")
//                 .with_contents(vec![Part::Value {
//                     value: to_value!({"temperature": 86, "unit": "fahrenheit"}),
//                 }]),
//         ];
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msgs);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"type":"function_call_output","call_id":"funcid_123456","output":"{\"temperature\":30,\"unit\":\"celsius\"}"},{"type":"function_call_output","call_id":"funcid_7890ab","output":"{\"temperature\":86,\"unit\":\"fahrenheit\"}"}]"#
//         );
//     }

//     #[test]
//     pub fn serialize_image() {
//         use base64::prelude::*;

//         let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAMAAAADCAAAAABzQ+pjAAAAF0lEQVR4AQEMAPP/AAoUHgAoMjwARlBaB4wBw+VFyrAAAAAASUVORK5CYII=";
//         let png_bytes = BASE64_STANDARD.decode(png_base64).unwrap();
//         let msg = Message::new(Role::User).with_contents([
//             Part::text("What you can see in this image?"),
//             Part::image_embedded("image/png".to_owned(), Bytes::from(png_bytes)).unwrap(),
//         ]);
//         let marshaled = Marshaled::<_, OpenAIMarshal>::new(&msg);
//         assert_eq!(
//             serde_json::to_string(&marshaled).unwrap(),
//             r#"[{"role":"user","content":[{"type":"input_text","text":"What you can see in this image?"},{"type":"input_image","image_url":{"url":"data:image/png;base64,"#.to_owned()
//                 + png_base64
//                 + r#""}}]}]"#,
//         );
//     }

//     #[test]
//     pub fn deserialize_text() {
//         let input = r#"{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world!"}]}]}"#;
//         let mut u = OpenAIUnmarshal;
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
//     pub fn deserialize_text_with_reasoning() {
//         let input = r#"{"status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"**Answering a simple question**\n\nUser is saying hello."}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world!"}]}]}"#;
//         let mut u = OpenAIUnmarshal;
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
//         let input = r#"{"status":"completed","output":[{"type":"function_call","call_id":"call_DF3wZtLHv5eBNfURjvI8MULJ","name":"get_weather","arguments":"{\"location\":\"Paris, France\"}"}]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::ToolCall {}));
//         let mut delta = output.delta;
//         assert_eq!(delta.tool_calls.len(), 1);
//         let tool_call = delta.tool_calls.pop().unwrap();
//         let (id, name, args) = tool_call.to_function().unwrap();
//         assert_eq!(id.unwrap(), "call_DF3wZtLHv5eBNfURjvI8MULJ");
//         assert_eq!(name, "get_weather");
//         assert_eq!(args, "{\"location\":\"Paris, France\"}");
//     }

//     #[test]
//     pub fn deserialize_incomplete() {
//         let input =
//             r#"{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]}"#;
//         let mut u = OpenAIUnmarshal;
//         let val = serde_json::from_str::<Value>(input).unwrap();
//         let output = u.unmarshal(val).unwrap();
//         assert_eq!(output.finish_reason, Some(FinishReason::Length {}));
//     }
// }
