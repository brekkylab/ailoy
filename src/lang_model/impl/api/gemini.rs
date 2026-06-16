use anyhow::bail;
use url::Url;

use super::super::response_format::ResponseSchemaMarshal;
use crate::{
    datatype::Value,
    lang_model::{
        LangModelAPISchema, LangModelProvider, LangModelProviderElem, LangModelRequest,
        ResponseFormat,
    },
    message::{
        FinishReason, Marshal, Message, MessageDelta, MessageDeltaOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage, Unmarshal,
    },
    to_value,
    tool::ToolDesc,
};

impl LangModelProvider {
    pub fn gemini(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: Some(api_key),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeminiMarshal;

impl ResponseSchemaMarshal for GeminiMarshal {
    fn marshal_response_schema(&self, schema: &Value) -> Value {
        const STRIP: &[&str] = &["additionalProperties", "$schema", "$defs", "definitions"];
        fn strip(schema: &Value) -> Value {
            match schema {
                Value::Object(obj) => Value::Object(
                    obj.iter()
                        .filter(|(k, _)| !STRIP.contains(&k.as_str()))
                        .map(|(k, v)| (k.clone(), strip(v)))
                        .collect(),
                ),
                Value::Array(arr) => Value::Array(arr.iter().map(strip).collect()),
                other => other.clone(),
            }
        }
        strip(schema)
    }
}

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
                let mut part_obj =
                    to_value!({"functionCall": {"name": name, "args": arguments.clone()}});
                // thoughtSignature is a sibling of functionCall at the part level
                if let Some(sig) = &msg.signature {
                    part_obj
                        .as_object_mut()
                        .unwrap()
                        .insert("thoughtSignature".into(), sig.into());
                }
                part_obj
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
                            return to_value!({
                                "error": "Gemini does not support image URL inputs; provide a base64 data URI instead"
                            });
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
        let (tool_name, _) = tool_call_id
            .split_once('/')
            .expect("Tool call id must be in \"{name}/call-{id}\" format");

        // Split contents: images become sibling inline_data parts alongside functionResponse;
        // non-image parts go into the functionResponse.response object.
        // The Gemini REST API FunctionResponse proto has no "parts" field — multimodal data
        // must live as separate parts in the outer parts array.
        let mut response_value: Option<Value> = None;
        let mut inline_data_parts: Vec<Value> = Vec::new();
        for part in msg.contents.iter() {
            match part {
                Part::Image {
                    image: PartImage::Embedded { mime_type, data },
                } => {
                    inline_data_parts.push(to_value!({
                        "inline_data": { "mime_type": mime_type, "data": data.base64() }
                    }));
                }
                other => {
                    response_value = Some(part_to_value(other));
                }
            }
        }

        let response_body = if let Some(rv) = response_value {
            to_value!({"result": rv})
        } else if !inline_data_parts.is_empty() {
            // Image-only result: provide a text description in response; actual bytes go
            // as sibling inline_data parts in the outer parts array.
            let mime = inline_data_parts
                .iter()
                .find_map(|p| p.pointer("/inline_data/mime_type").and_then(|v| v.as_str()))
                .unwrap_or("image/*");
            to_value!({"result": {"mimeType": mime, "type": "image"}})
        } else {
            to_value!({"result": {}})
        };

        let function_response_part = to_value!({
            "functionResponse": {
                "name": tool_name,
                "response": response_body
            }
        });

        // Combine functionResponse and any inline_data blobs as sibling parts
        let mut parts = vec![function_response_part];
        parts.extend(inline_data_parts);

        return to_value!({
            "role": "user",
            "parts": parts
        });
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
        let mut thought_part = to_value!({"text": thinking, "thought": true});
        if let Some(sig) = &msg.signature {
            thought_part
                .as_object_mut()
                .unwrap()
                .insert("thoughtSignature".into(), sig.into());
        }
        parts.push(thought_part);
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
    // For Gemini, always include thinking/thoughtSignature for model turns — the signature must
    // be replayed verbatim so Gemini can verify continuity across tool-use turns.
    Value::Array(
        msgs.iter()
            .filter(|m| m.role != Role::System)
            .map(|msg| marshal_message(msg, true))
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

        let contents = marshal_messages(req.messages);

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
        let mut generation_config = to_value!({});
        if let Some(max_tokens) = req.max_tokens {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("maxOutputTokens".into(), (max_tokens as i64).into());
        }
        if let Some(temperature) = req.temperature {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = req.top_p {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("topP".into(), top_p.into());
        }
        if let Some(top_k) = req.top_k {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("topK".into(), (top_k as i64).into());
        }
        if let Some(ResponseFormat::JsonSchema(schema)) = req.response_format {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("responseMimeType".into(), "application/json".into());
            generation_config.as_object_mut().unwrap().insert(
                "responseSchema".into(),
                self.marshal_response_schema(schema),
            );
        }
        if !generation_config.as_object().unwrap().is_empty() {
            body.as_object_mut()
                .unwrap()
                .insert("generationConfig".into(), generation_config);
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

impl super::QuotaClassifier for GeminiUnmarshal {
    fn is_permanent_quota_error(&self, body: &str) -> bool {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };
        let error = &json["error"];
        // RESOURCE_EXHAUSTED covers both; a RetryInfo detail marks the transient case.
        error["status"] == "RESOURCE_EXHAUSTED"
            && !error["details"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|d| {
                    d["@type"]
                        .as_str()
                        .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
                })
    }
}

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

        // Parse usage (Gemini: usageMetadata.promptTokenCount / candidatesTokenCount)
        let usage = val
            .as_object()
            .and_then(|r| r.get("usageMetadata"))
            .and_then(|u| u.as_object())
            .map(|u| TokenUsage {
                input_tokens: u
                    .get("promptTokenCount")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as u64,
                output_tokens: u
                    .get("candidatesTokenCount")
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

fn parse_candidate_content(candidate: &Value) -> anyhow::Result<MessageDelta> {
    let mut rv = MessageDelta::default();

    let content = candidate
        .pointer("/content")
        .ok_or_else(|| anyhow::anyhow!("Missing content in candidate"))?;

    // Gemini occasionally omits `content.role`; candidates are always model output.
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
    } else {
        rv.role = Some(Role::Assistant);
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
                        if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                            rv.signature = Some(sig.to_owned());
                        }
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
                    // thoughtSignature is at the part level (sibling of functionCall)
                    if let Some(sig) = part.get("thoughtSignature").and_then(|v| v.as_str()) {
                        rv.signature = Some(sig.to_owned());
                    }
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

    use super::super::QuotaClassifier;
    use super::*;
    use crate::{
        datatype::{Bytes, Value},
        lang_model::{LangModel, LangModelAPISchema, LangModelOptions, LangModelProviderElem},
        message::{Delta, FinishReason, Message, Part, Role, TokenUsage},
        tool::{ToolDesc, ToolDescBuilder},
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
            temperature: None,
            top_p: None,
            top_k: None,
            response_format: None,
        };
        f(&req)
    }

    #[test]
    fn test_unmarshal_usage() {
        let response = to_value!({
            "candidates": [{
                "content": {"role": "model", "parts": []},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 150,
                "candidatesTokenCount": 60
            }
        });
        let usage = GeminiUnmarshal::default()
            .unmarshal(response)
            .unwrap()
            .usage;
        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 150,
                output_tokens: 60,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            })
        );
    }

    /// Missing `content.role` defaults to Assistant instead of bailing.
    #[test]
    fn test_unmarshal_missing_content_role_defaults_to_assistant() {
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [{"text": "Hello from Gemini."}]},
                "finishReason": "STOP"
            }]
        });
        let out = GeminiUnmarshal::default().unmarshal(response).unwrap();
        assert_eq!(out.delta.role, Some(Role::Assistant));
        let msg = out.delta.finish().expect("finish() must not bail");
        assert_eq!(msg.role, Role::Assistant);
    }

    /// Verifies functionResponse.response.result marshaling for all Part variants.
    ///
    /// Gemini accepts arbitrary values in `result`, so:
    /// - Part::Text  → {"text": "..."} object (no double-encoding issue; object is valid)
    /// - Part::Value(String) → plain string "..." (no double-encoding via value.to_owned())
    /// - Part::Value(Object) → the object itself passed through
    /// - Part::Image (embedded) → image-only: functionResponse gets a {mimeType, type:"image"}
    ///   placeholder and the actual bytes appear as a sibling inline_data part; mixed with text:
    ///   text goes into functionResponse.response.result and image becomes a sibling inline_data part
    #[test]
    fn test_function_response_result_marshaling() {
        let get_result = |msg: &Message| -> Value {
            let v = marshal_message(msg, false);
            v.pointer("/parts/0/functionResponse/response/result")
                .expect("result must exist")
                .to_owned()
        };

        // Part::Text → {"text": "..."} (truncation placeholder path)
        let msg_text = Message::new(Role::Tool)
            .with_id("dummy_tool/call-1")
            .with_contents([Part::text("[context truncated]")]);
        let result = get_result(&msg_text);
        assert_eq!(
            result.pointer("/text").and_then(|v| v.as_str()),
            Some("[context truncated]"),
            "Part::Text must marshal to {{\"text\": \"...\"}} in functionResponse result"
        );

        // Part::Value(String) → plain string, no double-encoding
        let msg_str = Message::new(Role::Tool)
            .with_id("dummy_tool/call-2")
            .with_contents([Part::value(Value::string("ok".to_string()))]);
        let result = get_result(&msg_str);
        assert_eq!(
            result.as_str(),
            Some("ok"),
            "Part::Value(String) must not be double-encoded in functionResponse result"
        );

        // Part::Value(Object) → object passed through as-is
        let msg_obj = Message::new(Role::Tool)
            .with_id("dummy_tool/call-3")
            .with_contents([Part::value(to_value!({"temperature": 30}))]);
        let result = get_result(&msg_obj);
        assert_eq!(
            result.pointer("/temperature").and_then(|v| v.as_integer()),
            Some(30),
            "Part::Value(Object) must pass through as object in functionResponse result"
        );

        // Part::Image (embedded, image-only) → functionResponse gets a {mimeType, type:"image"}
        // placeholder; actual bytes appear as a sibling inline_data part at parts[1].
        let img_bytes = Bytes::from(vec![0xFFu8, 0xD8, 0xFF]);
        let msg_img = Message::new(Role::Tool)
            .with_id("dummy_tool/call-4")
            .with_contents([Part::image_embedded("image/jpeg", img_bytes.clone()).unwrap()]);
        let val = marshal_message(&msg_img, false);
        let parts = val
            .pointer("/parts")
            .and_then(|v| v.as_array())
            .expect("parts must exist");
        assert_eq!(
            parts.len(),
            2,
            "image-only tool result must produce functionResponse + inline_data sibling"
        );
        assert_eq!(
            parts[0]
                .pointer("/functionResponse/response/result/type")
                .and_then(|v| v.as_str()),
            Some("image"),
            "image-only functionResponse result must have type \"image\""
        );
        assert_eq!(
            parts[0]
                .pointer("/functionResponse/response/result/mimeType")
                .and_then(|v| v.as_str()),
            Some("image/jpeg")
        );
        assert_eq!(
            parts[1]
                .pointer("/inline_data/mime_type")
                .and_then(|v| v.as_str()),
            Some("image/jpeg"),
            "image bytes must appear as sibling inline_data part"
        );
        assert_eq!(
            parts[1]
                .pointer("/inline_data/data")
                .and_then(|v| v.as_str()),
            Some(img_bytes.base64().as_str())
        );

        // Part::Text + Part::Image (embedded) → text in functionResponse result, image as sibling.
        let msg_mixed = Message::new(Role::Tool)
            .with_id("dummy_tool/call-5")
            .with_contents([
                Part::text("here is the file"),
                Part::image_embedded("image/png", img_bytes.clone()).unwrap(),
            ]);
        let val = marshal_message(&msg_mixed, false);
        let parts = val
            .pointer("/parts")
            .and_then(|v| v.as_array())
            .expect("parts must exist");
        assert_eq!(
            parts.len(),
            2,
            "mixed tool result must produce functionResponse + inline_data sibling"
        );
        assert_eq!(
            parts[0]
                .pointer("/functionResponse/response/result/text")
                .and_then(|v| v.as_str()),
            Some("here is the file"),
            "text part must appear in functionResponse result"
        );
        assert_eq!(
            parts[1]
                .pointer("/inline_data/mime_type")
                .and_then(|v| v.as_str()),
            Some("image/png")
        );
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

    #[test]
    fn test_marshal_response_format_absent() {
        with_req("gemini-2.5-flash-lite", None, |req| {
            let val = GeminiMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(body.as_object().unwrap().get("generationConfig").is_none());
        });
    }

    #[test]
    fn test_marshal_response_format_json_schema() {
        let schema = to_value!({"type": "object", "properties": {"city": {"type": "string"}}});
        let fmt = ResponseFormat::json_schema(schema.clone().into()).unwrap();
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap();
        let api_key: Option<String> = None;
        let req = LangModelRequest {
            model: "gemini-2.5-flash-lite",
            messages: &messages,
            tools: &tools,
            url: &url,
            api_key: &api_key,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            response_format: Some(&fmt),
        };
        let val = GeminiMarshal::default().marshal(&req);
        let body = val.as_object().unwrap().get("body").unwrap();
        let gen_cfg = body.as_object().unwrap().get("generationConfig").unwrap();
        assert_eq!(
            gen_cfg
                .pointer("/responseMimeType")
                .and_then(|v| v.as_str()),
            Some("application/json")
        );
        assert_eq!(
            gen_cfg
                .pointer("/responseSchema/type")
                .and_then(|v| v.as_str()),
            Some("object")
        );
    }

    /// Verifies structured output via response_format: the model returns valid JSON matching the schema.
    #[test_with::env(GEMINI_API_KEY)]
    #[tokio::test]
    async fn test_run_response_format_json_schema() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set in .env");

        let schema = to_value!({
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"}
            },
            "required": ["country", "capital"]
        });

        let model = LangModel::new(
            "gemini-2.5-flash-lite".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::Gemini,
                url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/")
                    .unwrap(),
                api_key: Some(api_key),
            },
        );
        let messages = vec![Message::new(Role::User).with_contents([Part::text(
            "Return France's country name and capital city in the requested format.",
        )])];

        let resp = model
            .run(
                &messages,
                &[],
                &LangModelOptions {
                    response_format: Some(ResponseFormat::json_schema(schema).unwrap()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert_eq!(resp.finish_reason, FinishReason::Stop {});
        let text = resp
            .message
            .contents
            .iter()
            .find_map(|p| p.as_text())
            .expect("Expected text content");
        let parsed: serde_json::Value =
            serde_json::from_str(text).expect("Response must be valid JSON");
        assert_eq!(parsed["capital"].as_str().unwrap().to_lowercase(), "paris");
    }

    /// Verifies that max_tokens is respected by the Gemini API (finishReason: MAX_TOKENS).
    #[test_with::env(GEMINI_API_KEY)]
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set in .env");

        let model = LangModel::new(
            "gemini-2.5-flash-lite".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::Gemini,
                url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/")
                    .unwrap(),
                api_key: Some(api_key),
            },
        );
        let messages = vec![
            Message::new(Role::User)
                .with_contents([Part::text("Tell me a long story about a dragon.")]),
        ];
        let tools: Vec<ToolDesc> = vec![];

        let resp = model
            .run(
                &messages,
                &tools,
                &LangModelOptions {
                    max_tokens: Some(5),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length {});
    }

    /// Verifies that an image embedded in a Role::Tool message is accepted by the Gemini API
    /// via functionResponse.parts[].inlineData and that the model can respond after seeing it.
    ///
    /// Uses a 2-turn interaction so the model's own functionCall (with thoughtSignature) is used
    /// in the conversation history — required by Gemini 3 thinking models.
    #[tokio::test]
    async fn test_tool_result_with_image() {
        dotenvy::dotenv().ok();
        let api_key = match std::env::var("GEMINI_API_KEY") {
            Ok(k) => k,
            Err(_) => return,
        };

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
            "gemini-3-flash-preview".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::Gemini,
                url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/")
                    .unwrap(),
                api_key: Some(api_key),
            },
        );

        let tools =
            vec![ToolDescBuilder::new("file_read")
            .description("Read a file and return its contents. Images are returned inline.")
            .parameters(to_value!({
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"]
            }))
            .build()];

        // Turn 1: ask the model to use the file_read tool. The model will respond with a
        // functionCall that includes a thoughtSignature (captured by the unmarshal).
        let user_messages = vec![Message::new(Role::User).with_contents([Part::text(
            "Use the file_read tool to read /tmp/photo.jpg, then describe who you see.",
        )])];
        let step1 = model
            .run(&user_messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();
        assert_eq!(
            step1.finish_reason,
            FinishReason::ToolCall {},
            "Expected model to call file_read"
        );
        assert!(
            step1.message.signature.is_some(),
            "Expected thoughtSignature from gemini-3-flash-preview"
        );

        // Extract the tool call ID from the model's response so we can link the tool result.
        let tool_call_id = step1
            .message
            .tool_calls
            .as_ref()
            .and_then(|calls| calls.first())
            .and_then(|p| {
                if let Part::Function { id, .. } = p {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .expect("Expected a function call with an id");

        // Turn 2: replay the model's functionCall (with thoughtSignature) + our tool result.
        let mut messages = user_messages;
        messages.push(step1.message);
        messages.push(
            Message::new(Role::Tool)
                .with_id(tool_call_id)
                .with_contents([
                    Part::image_embedded("image/jpeg", Bytes::from(img_bytes)).unwrap()
                ]),
        );

        let step2 = model
            .run(&messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();
        assert_eq!(step2.finish_reason, FinishReason::Stop {});
        assert!(
            step2.message.contents.iter().any(|p| p.as_text().is_some()),
            "Expected text response after image tool result"
        );
    }

    #[test]
    fn test_is_permanent_quota_error() {
        let u = GeminiUnmarshal;
        let quota = r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}"#;
        let rate = r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"34s"}]}}"#;
        assert!(u.is_permanent_quota_error(quota));
        assert!(!u.is_permanent_quota_error(rate));
        assert!(!u.is_permanent_quota_error("not json"));
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
