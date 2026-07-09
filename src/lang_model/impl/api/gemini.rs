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
    fn marshal(&self, msg: &Message) -> Value {
        marshal_message(msg, true)
    }
}

impl Marshal<ToolDesc> for GeminiMarshal {
    fn marshal(&self, item: &ToolDesc) -> Value {
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
    fn marshal(&self, req: &LangModelRequest<'_>) -> Value {
        let LangModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;

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

        // Gemini selects streaming via the endpoint, not a body flag:
        // `:streamGenerateContent?alt=sse` emits SSE; `:generateContent` returns one JSON.
        let url = if req.stream {
            format!("{}{}:streamGenerateContent?alt=sse", url, req.model)
        } else {
            format!("{}{}:generateContent", url, req.model)
        };

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("x-goog-api-key".into(), api_key.into());
        }
        if req.stream {
            header
                .as_object_mut()
                .unwrap()
                .insert("accept".into(), "text/event-stream".into());
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
        if let Some(max_tokens) = options.max_tokens {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("maxOutputTokens".into(), (max_tokens as i64).into());
        }
        if let Some(temperature) = options.temperature {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = options.top_p {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("topP".into(), top_p.into());
        }
        if let Some(top_k) = options.top_k {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("topK".into(), (top_k as i64).into());
        }
        if let Some(ResponseFormat::JsonSchema(schema)) = &options.response_format {
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
            && !error["details"].as_array().into_iter().flatten().any(|d| {
                d["@type"]
                    .as_str()
                    .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
            })
    }
}

impl GeminiUnmarshal {
    /// Maps a Gemini `finishReason` to a [`FinishReason`].
    fn parse_finish_reason(reason: &str) -> FinishReason {
        match reason {
            "STOP" => FinishReason::Stop {},
            "MAX_TOKENS" => FinishReason::Length {},
            other => FinishReason::Refusal {
                reason: other.to_owned(),
            },
        }
    }

    /// Parses Gemini `usageMetadata` (`promptTokenCount` / `candidatesTokenCount`).
    /// `promptTokenCount` includes any cached-content tokens (folded into `input_tokens`).
    fn parse_usage(root: &Value) -> Option<TokenUsage> {
        let u = root
            .pointer("/usageMetadata")
            .filter(|u| !u.is_null())?
            .as_object()?;
        Some(TokenUsage {
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
        })
    }
}

/// Parses one Gemini SSE chunk (`?alt=sse`) into a delta. Each chunk is a
/// partial `GenerateContentResponse` of the same shape as the final response
/// and carries incremental text, so it reuses `parse_candidate_content` and
/// accumulates. A chunk without a candidate yields no delta; `finishReason` and
/// `usageMetadata` arrive on the final chunk (alongside the function call, if
/// any — Gemini sends `STOP` even for tool calls, adjusted to `ToolCall`).
impl Unmarshal<MessageDeltaOutput> for GeminiUnmarshal {
    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let val: Value = serde_json::from_str(data)?;
        let Some(candidate) = val.pointer("/candidates/0").map(|c| c.to_owned()) else {
            return Ok(None);
        };

        let finish_reason = candidate
            .pointer("/finishReason")
            .and_then(|v| v.as_str())
            .map(Self::parse_finish_reason);

        let delta = match (&finish_reason, candidate.pointer("/content")) {
            // A finish-only or refusal chunk may omit content. Still set the role
            // (candidates are always model output) so a stream whose only/first
            // chunk is finish-only doesn't accumulate to a role-less message that
            // makes `finish()` bail.
            (Some(FinishReason::Refusal { .. }), _) | (_, None) => {
                MessageDelta::new().with_role(Role::Assistant)
            }
            _ => parse_candidate_content(&candidate)?,
        };

        // NOTE: Gemini reports "STOP" even for tool calls, and can split the
        // functionCall part and the terminal STOP across separate SSE chunks
        // (2.5 Pro / 3.x), so neither chunk alone carries both. The
        // STOP→ToolCall promotion therefore can't be done per chunk — it lives
        // in `MessageDeltaOutput::finish()`, on the fully accumulated message.
        let usage = Self::parse_usage(&val);

        Ok(Some(MessageDeltaOutput {
            delta,
            finish_reason,
            usage,
            depth: None,
            source_agent: None,
        }))
    }

    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let candidate = val
            .pointer("/candidates/0")
            .ok_or_else(|| anyhow::anyhow!("Missing candidates[0] in response"))?
            .to_owned();

        let mut finish_reason = candidate
            .pointer("/finishReason")
            .and_then(|v| v.as_str())
            .map(Self::parse_finish_reason);

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
        let usage = Self::parse_usage(&val);

        Ok(MessageDeltaOutput {
            delta,
            finish_reason,
            usage,
            depth: None,
            source_agent: None,
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
                        // Append rather than overwrite: a candidate may carry
                        // multiple thought parts, and thinking concatenates
                        // across chunks anyway.
                        match &mut rv.thinking {
                            Some(existing) => existing.push_str(text),
                            None => rv.thinking = Some(text.into()),
                        }
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

    use super::*;
    use crate::{
        datatype::{Bytes, Value},
        lang_model::{
            LangModel, LangModelAPISchema, LangModelOptions, LangModelProvider,
            LangModelProviderElem, get_lm_providers_mut,
        },
        message::{Delta, FinishReason, Message, MessageDeltaOutput, Part, Role, TokenUsage},
        tool::{ToolDesc, ToolDescBuilder},
    };

    /// Feeds Gemini SSE chunk payloads through `unmarshal_event`, accumulating
    /// to a final `MessageDeltaOutput`.
    fn accumulate_stream(inputs: &[&str]) -> MessageDeltaOutput {
        let mut u = GeminiUnmarshal;
        let mut acc = MessageDeltaOutput::new();
        for input in inputs {
            if let Some(out) = u.unmarshal_event(input).unwrap() {
                acc = acc.accumulate(out).unwrap();
            }
        }
        acc
    }

    #[test]
    fn test_unmarshal_event_text_stream() {
        // Incremental text across chunks; the final chunk carries finishReason +
        // usageMetadata with no content (exercises the content-optional path).
        let inputs = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" world!"}]}}]}"#,
            r#"{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":3}}"#,
        ];
        let result = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(result.finish_reason, FinishReason::Stop {});
        let usage = result.usage.expect("expected usage");
        assert_eq!(usage.input_tokens, 7);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(result.message.role, Role::Assistant);
        assert_eq!(result.message.contents.len(), 1);
        assert_eq!(result.message.contents[0].as_text(), Some("Hello world!"));
    }

    #[test]
    fn test_unmarshal_event_multiple_thought_parts_concatenate() {
        // A single chunk carrying more than one thought part must concatenate
        // them, not overwrite (thinking accumulates across chunks anyway).
        let inputs = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"one ","thought":true},{"text":"two","thought":true}]}}]}"#,
        ];
        let out = accumulate_stream(&inputs);
        assert_eq!(out.delta.thinking.as_deref(), Some("one two"));
    }

    #[test]
    fn test_unmarshal_event_tool_call_stream() {
        // Gemini sends the whole functionCall and finishReason in one chunk.
        let inputs = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":6}}"#,
        ];
        // STOP is promoted to ToolCall on the finished message, not per chunk.
        let finished = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(finished.finish_reason, FinishReason::ToolCall {});
        let tool_calls = finished.message.tool_calls.expect("expected tool_calls");
        assert_eq!(tool_calls.len(), 1);
        let (id, name, args) = tool_calls[0]
            .as_function()
            .expect("expected a function call");
        assert!(id.starts_with("get_weather/"), "got id {id}");
        assert_eq!(name, "get_weather");
        assert_eq!(
            args.pointer("/location").and_then(|v| v.as_str()),
            Some("Paris")
        );
    }

    #[test]
    fn test_unmarshal_event_tool_call_split_stream() {
        // Gemini 2.5 Pro / 3.x can deliver the functionCall part and the
        // terminal `finishReason: STOP` in *separate* SSE chunks. Neither chunk
        // alone carries both, so the STOP→ToolCall promotion must happen on the
        // accumulated message (in finish()), not per chunk — otherwise the turn
        // looks like a plain Stop and the tool call is silently dropped.
        let inputs = [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}]}}]}"#,
            r#"{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":6}}"#,
        ];
        let finished = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(finished.finish_reason, FinishReason::ToolCall {});
        let tool_calls = finished.message.tool_calls.expect("expected tool_calls");
        assert_eq!(tool_calls.len(), 1);
        let (_, name, _) = tool_calls[0]
            .as_function()
            .expect("expected a function call");
        assert_eq!(name, "get_weather");
    }

    /// End-to-end: `run_stream` over Gemini `:streamGenerateContent?alt=sse`
    /// yields multiple deltas that accumulate into a complete message.
    #[test_with::env(GEMINI_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_text() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set");

        let model = build_gemini_model(
            "gemini_test_run_stream_text",
            "gemini-2.5-flash-lite",
            api_key,
        );
        let messages = vec![
            Message::new(Role::User).with_contents([Part::text("Reply with a short greeting.")]),
        ];

        let mut stream = model.run_stream(&messages, &[], &LangModelOptions::default());
        let mut acc = MessageDeltaOutput::new();
        let mut chunks = 0usize;
        while let Some(item) = stream.next().await {
            acc = acc.accumulate(item.unwrap()).unwrap();
            chunks += 1;
        }

        assert!(
            chunks >= 1,
            "expected at least one streamed delta, got {chunks}"
        );
        let result = acc.finish().unwrap();
        assert_eq!(result.finish_reason, FinishReason::Stop {});
        assert!(
            result
                .message
                .contents
                .iter()
                .any(|p| p.as_text().is_some_and(|t| !t.is_empty())),
            "expected non-empty text in the streamed message"
        );
    }

    /// Register a one-off [`LangModelProvider`] under `provider_name` in the
    /// global registry and build the [`LangModel`] via
    /// [`LangModel::try_from_provider`].  Test fixtures only.
    fn build_gemini_model(provider_name: &str, model: &str, api_key: String) -> LangModel {
        let elem = LangModelProviderElem::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: Some(api_key),
        };
        let mut lmp = LangModelProvider::new();
        lmp.insert(model.into(), elem);
        get_lm_providers_mut().insert(provider_name.into(), lmp);
        LangModel::try_from_provider(model.to_string(), provider_name).unwrap()
    }

    fn with_req<F, R>(model: &str, max_tokens: Option<u64>, f: F) -> R
    where
        F: FnOnce(&LangModelRequest<'_>) -> R,
    {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions {
            max_tokens,
            ..Default::default()
        };
        let req = LangModelRequest {
            model,
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
        };
        f(&req)
    }

    #[test]
    fn test_marshal_stream_endpoint() {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions::default();
        let mut req = LangModelRequest {
            model: "gemini-2.5-flash",
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
        };

        // stream: false → non-streaming endpoint, no accept header.
        let val = GeminiMarshal::default().marshal(&req);
        let endpoint = val.pointer("/url").and_then(|v| v.as_str()).unwrap();
        assert!(endpoint.ends_with(":generateContent"), "got {endpoint}");
        assert!(val.pointer("/header/accept").is_none());

        // stream: true → SSE streaming endpoint (the `?alt=sse` endpoint is the
        // streaming trigger), plus an `Accept: text/event-stream` header so
        // intermediaries don't buffer.
        req.stream = true;
        let val = GeminiMarshal::default().marshal(&req);
        let endpoint = val.pointer("/url").and_then(|v| v.as_str()).unwrap();
        assert!(
            endpoint.ends_with(":streamGenerateContent?alt=sse"),
            "got {endpoint}"
        );
        assert_eq!(
            val.pointer("/header/accept").and_then(|v| v.as_str()),
            Some("text/event-stream")
        );
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
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions {
            response_format: Some(fmt),
            ..Default::default()
        };
        let req = LangModelRequest {
            model: "gemini-2.5-flash-lite",
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
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

        let model = build_gemini_model(
            "gemini_test_run_response_format_json_schema",
            "gemini-2.5-flash-lite",
            api_key,
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

        let model =
            build_gemini_model("gemini_test_run_max_tokens", "gemini-2.5-flash-lite", api_key);
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

        let model = build_gemini_model(
            "gemini_test_tool_result_with_image",
            "gemini-3-flash-preview",
            api_key,
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
