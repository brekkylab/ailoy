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
    pub fn openai(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
            api_key: Some(api_key),
        }
    }
}

/// Returns whether `model` is an OpenAI reasoning model that does not accept
/// the `temperature` / `top_p` / `top_k` sampling parameters.
pub(super) fn is_openai_reasoning_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    if m.starts_with("gpt-5") {
        return true;
    }
    // OpenAI reasoning families share an `o<digit>` prefix: o1, o3, o4, ...
    let bytes = m.as_bytes();
    bytes.len() >= 2 && bytes[0] == b'o' && bytes[1].is_ascii_digit()
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIMarshal;

impl ResponseSchemaMarshal for OpenAIMarshal {}

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
        .unwrap_or(msgs.len());
    Value::Array(
        msgs.iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .flat_map(|(i, msg)| marshal_message(msg, i > last_user_index))
            .collect::<Vec<_>>(),
    )
}

impl Marshal<Message> for OpenAIMarshal {
    fn marshal(&self, msg: &Message) -> Value {
        to_value!(marshal_message(msg, true))
    }
}

impl Marshal<ToolDesc> for OpenAIMarshal {
    fn marshal(&self, item: &ToolDesc) -> Value {
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
    fn marshal(&self, req: &LangModelRequest<'_>) -> Value {
        let LangModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;

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

        let input = marshal_messages(req.messages);

        let tools = if !req.tools.is_empty() {
            Value::Array(req.tools.iter().map(|t| self.marshal(t)).collect())
        } else {
            Value::Null
        };

        let url = url.to_string();

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = api_key.as_ref() {
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
        if let Some(max_tokens) = options.max_tokens {
            body.as_object_mut()
                .unwrap()
                .insert("max_output_tokens".into(), (max_tokens as i64).into());
        }
        // OpenAI reasoning models (o-series, gpt-5) reject temperature/top_p; drop them silently.
        if let Some(temperature) = options.temperature
            && !is_openai_reasoning_model(req.model)
        {
            body.as_object_mut()
                .unwrap()
                .insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = options.top_p
            && !is_openai_reasoning_model(req.model)
        {
            body.as_object_mut()
                .unwrap()
                .insert("top_p".into(), top_p.into());
        }
        // top_k is not part of the OpenAI Responses spec; intentionally ignored.
        if let Some(ResponseFormat::JsonSchema(schema)) = &options.response_format {
            let wire_schema = self.marshal_response_schema(schema);
            body.as_object_mut().unwrap().insert(
                "text".into(),
                to_value!({
                    "format": {
                        "type": "json_schema",
                        "name": "response",
                        "strict": true,
                        "schema": wire_schema
                    }
                }),
            );
        }

        if req.stream {
            body.as_object_mut()
                .unwrap()
                .insert("stream".into(), true.into());
            header
                .as_object_mut()
                .unwrap()
                .insert("accept".into(), "text/event-stream".into());
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

impl super::QuotaClassifier for OpenAIUnmarshal {
    fn is_permanent_quota_error(&self, body: &str) -> bool {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };
        let error = &json["error"];
        error["type"] == "insufficient_quota" || error["code"] == "insufficient_quota"
    }
}

/// Parses one Responses API SSE event (`response.*`) into a delta: incremental
/// text (`output_text.delta`), reasoning (`reasoning_summary_text.delta`), and
/// whole function calls (`output_item.done`); `response.completed` reuses the
/// full-response `Unmarshal` for finish_reason + usage. Other events: no delta.
impl Unmarshal<MessageDeltaOutput> for OpenAIUnmarshal {
    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let val: Value = serde_json::from_str(data)?;
        let ty = val.pointer("/type").and_then(|v| v.as_str()).unwrap_or("");

        let mut delta = MessageDelta::new();
        match ty {
            // A message output item begins: establish the assistant role.
            "response.output_item.added"
                if val.pointer("/item/type").and_then(|v| v.as_str()) == Some("message") =>
            {
                delta = delta.with_role(Role::Assistant);
            }
            // Incremental assistant text.
            "response.output_text.delta" => {
                let Some(text) = val.pointer("/delta").and_then(|v| v.as_str()) else {
                    return Ok(None);
                };
                delta = delta
                    .with_role(Role::Assistant)
                    .with_contents([PartDelta::Text {
                        text: text.to_owned(),
                    }]);
            }
            // Incremental reasoning summary -> thinking. Set the role too (like
            // output_text.delta): a reasoning model truncated mid-reasoning emits
            // only reasoning before the terminal event, and without a role the
            // accumulated message makes finish() bail with "Role not specified".
            "response.reasoning_summary_text.delta" => {
                let Some(text) = val.pointer("/delta").and_then(|v| v.as_str()) else {
                    return Ok(None);
                };
                delta = delta.with_role(Role::Assistant);
                delta.thinking = Some(text.to_owned());
            }
            // A finalized function call (carries call_id + name + full arguments).
            "response.output_item.done"
                if val.pointer("/item/type").and_then(|v| v.as_str()) == Some("function_call") =>
            {
                let id = val
                    .pointer("/item/call_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_owned());
                let name = val
                    .pointer("/item/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();
                let arguments = val
                    .pointer("/item/arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();
                delta = delta
                    .with_role(Role::Assistant)
                    .with_tool_calls([PartDelta::Function {
                        id,
                        function: PartDeltaFunction::WithStringArgs { name, arguments },
                    }]);
            }
            // Terminal success/partial: reuse the full-response parser for
            // finish_reason + usage; streamed content is dropped (already sent).
            "response.completed" | "response.incomplete" => {
                let Some(resp) = val.pointer("/response") else {
                    return Ok(None);
                };
                let parsed = OpenAIUnmarshal.unmarshal(resp.clone())?;
                return Ok(Some(MessageDeltaOutput {
                    delta: MessageDelta::new(),
                    finish_reason: parsed.finish_reason,
                    usage: parsed.usage,
                    depth: None,
                    source_agent: None,
                }));
            }
            // A failed response carries its reason in `response.error`; surface
            // it instead of silently finishing with no finish_reason.
            "response.failed" => {
                let msg = val
                    .pointer("/response/error/message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("response failed");
                anyhow::bail!("OpenAI Responses failed: {msg}");
            }
            "error" => {
                let msg = val
                    .pointer("/message")
                    .or_else(|| val.pointer("/error/message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error");
                anyhow::bail!("OpenAI Responses stream error: {msg}");
            }
            // Lifecycle / no-delta events (created, in_progress, content_part.*,
            // *_text.done, output_item.added for non-message items, ...).
            _ => return Ok(None),
        }
        Ok(Some(MessageDeltaOutput {
            delta,
            finish_reason: None,
            usage: None,
            depth: None,
            source_agent: None,
        }))
    }

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
                                if (part_ty == "output_text" || part_ty == "text")
                                    && let Some(text) =
                                        part.pointer("/text").and_then(|v| v.as_str())
                                {
                                    delta.contents.push(PartDelta::Text {
                                        text: text.to_owned(),
                                    });
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
            depth: None,
            source_agent: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use url::Url;

    use super::*;
    use crate::{
        datatype::Bytes,
        lang_model::{
            LangModel, LangModelAPISchema, LangModelOptions, LangModelProvider,
            LangModelProviderElem, get_lm_providers_mut,
        },
        message::{Delta, FinishReason, Message, MessageDeltaOutput, Part, Role, TokenUsage},
        tool::{ToolDesc, ToolDescBuilder},
    };

    /// Feeds Responses SSE event payloads through `unmarshal_event`,
    /// accumulating to a final `MessageDeltaOutput`.
    fn accumulate_stream(inputs: &[&str]) -> MessageDeltaOutput {
        let mut u = OpenAIUnmarshal;
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
        let inputs = [
            r#"{"type":"response.created","response":{"status":"in_progress"}}"#,
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}"#,
            r#"{"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}"#,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"Hello"}"#,
            r#"{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":" world!"}"#,
            r#"{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"Hello world!"}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello world!"}]}],"usage":{"input_tokens":9,"output_tokens":3}}}"#,
        ];
        let result = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(result.finish_reason, FinishReason::Stop {});
        let usage = result.usage.expect("expected usage");
        assert_eq!(usage.input_tokens, 9);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(result.message.role, Role::Assistant);
        assert_eq!(result.message.contents.len(), 1);
        assert_eq!(result.message.contents[0].as_text(), Some("Hello world!"));
    }

    #[test]
    fn test_unmarshal_event_tool_call_stream() {
        let inputs = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"location"}"#,
            r#"{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"\":\"Paris\"}"}"#,
            r#"{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"location\":\"Paris\"}"}"#,
            r#"{"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}"#,
            r#"{"type":"response.completed","response":{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"location\":\"Paris\"}"}],"usage":{"input_tokens":20,"output_tokens":8}}}"#,
        ];
        let out = accumulate_stream(&inputs);
        assert_eq!(out.finish_reason, Some(FinishReason::ToolCall {}));
        let msg = out.finish().unwrap().message;
        let tool_calls = msg.tool_calls.expect("expected tool_calls");
        assert_eq!(tool_calls.len(), 1);
        let (id, name, args) = tool_calls[0]
            .as_function()
            .expect("expected a function call");
        assert_eq!(id, "call_1");
        assert_eq!(name, "get_weather");
        assert_eq!(
            args.pointer("/location").and_then(|v| v.as_str()),
            Some("Paris")
        );
    }

    #[test]
    fn test_unmarshal_event_reasoning_only_sets_role() {
        // A reasoning model truncated mid-reasoning (max_output_tokens) emits only
        // reasoning before the terminal event, with no message item. The role
        // must still be set so finish() doesn't bail with "Role not specified".
        let inputs = [
            r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1"}}"#,
            r#"{"type":"response.reasoning_summary_text.delta","delta":"thinking hard..."}"#,
            r#"{"type":"response.incomplete","response":{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":5,"output_tokens":50}}}"#,
        ];
        let result = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(result.message.role, Role::Assistant);
        assert_eq!(result.message.thinking.as_deref(), Some("thinking hard..."));
        assert_eq!(result.finish_reason, FinishReason::Length {});
    }

    /// End-to-end: `run_stream` over the OpenAI Responses API yields multiple
    /// deltas that accumulate into a complete message.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_text() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

        let model = build_openai_model("openai_test_run_stream_text", "gpt-4.1-mini", api_key);
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
            chunks > 1,
            "expected multiple streamed deltas, got {chunks}"
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
    fn build_openai_model(provider_name: &str, model: &str, api_key: String) -> LangModel {
        let elem = LangModelProviderElem::API {
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
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
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
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
    fn test_marshal_response_format_absent() {
        with_req("gpt-4o", None, |req| {
            let val = OpenAIMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(
                body.as_object().unwrap().get("text").is_none(),
                "text must not appear when response_format is None"
            );
        });
    }

    #[test]
    fn test_marshal_response_format_json_schema() {
        let schema = to_value!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": false
        });
        let fmt = ResponseFormat::json_schema(schema).unwrap();

        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/responses").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions {
            response_format: Some(fmt),
            ..Default::default()
        };
        let req = LangModelRequest {
            model: "gpt-4o",
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
        };

        let val = OpenAIMarshal::default().marshal(&req);
        let body = val.as_object().unwrap().get("body").unwrap();

        assert_eq!(
            body.pointer("/text/format/type").and_then(|v| v.as_str()),
            Some("json_schema")
        );
        assert_eq!(
            body.pointer("/text/format/name").and_then(|v| v.as_str()),
            Some("response")
        );
        assert_eq!(
            body.pointer("/text/format/strict")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(
            body.pointer("/text/format/schema").is_some(),
            "schema must be present in text.format"
        );
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
        let img_bytes = Bytes::from(vec![0xFFu8, 0xD8, 0xFF]);
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
            .with_contents([Part::value(Value::string("ok".to_string()))]);
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
            .with_contents([Part::value(to_value!({"temp": 25}))]);
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
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = build_openai_model("openai_test_run_max_tokens", "gpt-5.4-mini", api_key);
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
                    max_tokens: Some(16),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length {});
    }

    /// Verifies structured output via response_format: the model returns valid JSON matching the schema.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_response_format_json_schema() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let schema = to_value!({
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"}
            },
            "required": ["country", "capital"],
            "additionalProperties": false
        });

        let model = build_openai_model(
            "openai_test_run_response_format_json_schema",
            "gpt-4.1-mini",
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

        let model =
            build_openai_model("openai_test_tool_result_with_image", "gpt-5.4-mini", api_key);

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

        let resp = model
            .run(&messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();
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
