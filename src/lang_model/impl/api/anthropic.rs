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
    pub fn anthropic(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::Anthropic,
            url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
            api_key: Some(api_key),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnthropicMarshal;

impl ResponseSchemaMarshal for AnthropicMarshal {}

fn marshal_message(item: &Message, include_thinking: bool) -> Value {
    let part_to_value = |part: &Part| -> Value {
        match part {
            Part::Text { text } => to_value!({"type": "text", "text": text}),
            Part::Function {
                id,
                function: PartFunction { name, arguments },
            } => {
                to_value!({"type": "tool_use", "id": id, "name": name, "input": arguments.clone()})
            }
            Part::Value { value } => value.to_owned(),
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
        let content: Vec<Value> = item
            .contents
            .iter()
            .filter_map(|part| match part {
                Part::Text { .. } | Part::Image { .. } => Some(part_to_value(part)),
                Part::Value { value } => {
                    let text = match value {
                        Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    Some(to_value!({"type": "text", "text": text}))
                }
                _ => None,
            })
            .collect();
        return to_value!(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": item.id.clone().expect("Tool call id must exist."),
                        "content": content
                    }
                ]
            }
        );
    }

    let mut contents = Vec::<Value>::new();
    if let Some(thinking) = &item.thinking
        && !thinking.is_empty()
        && include_thinking
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

    to_value!({"role": item.role.to_string(), "content": contents})
}

/// Marshal a message slice with position-aware thinking inclusion.
///
/// Thinking blocks are only included for assistant messages that appear after
/// the last user message, matching Anthropic's extended-thinking requirements.
/// System messages are extracted separately and excluded from the array.
fn marshal_messages(messages: &[Message]) -> Value {
    let last_user_index = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len());
    Value::Array(
        messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != Role::System)
            .map(|(i, msg)| marshal_message(msg, i > last_user_index))
            .collect::<Vec<_>>(),
    )
}

impl Marshal<Message> for AnthropicMarshal {
    fn marshal(&self, item: &Message) -> Value {
        marshal_message(item, true)
    }
}

impl Marshal<ToolDesc> for AnthropicMarshal {
    fn marshal(&self, item: &ToolDesc) -> Value {
        if let Some(desc) = &item.description {
            to_value!({
                "name": &item.name,
                "description": desc,
                "input_schema": item.parameters.clone()
            })
        } else {
            to_value!({
                "name": &item.name,
                "input_schema": item.parameters.clone()
            })
        }
    }
}

impl Marshal<LangModelRequest<'_>> for AnthropicMarshal {
    fn marshal(&self, req: &LangModelRequest<'_>) -> Value {
        let LangModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;
        let model = Value::from(req.model);

        // Extract system message text if present
        let system = req
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

        let messages = marshal_messages(req.messages);

        let tools = if !req.tools.is_empty() {
            self.marshal(req.tools)
        } else {
            Value::Null
        };

        let url = url.to_string();

        let mut header = to_value!({
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        });
        if let Some(api_key) = api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("x-api-key".into(), api_key.into());
        }

        #[cfg(target_arch = "wasm32")]
        header.as_object_mut().unwrap().insert(
            "anthropic-dangerous-direct-browser-access".into(),
            "true".into(),
        );

        // Anthropic requires an explicit max_tokens value, so we set it as 8192
        let max_tokens = options.max_tokens.unwrap_or(8192) as i64;
        let mut body = to_value!({
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        });
        if let Some(system) = system {
            body.as_object_mut()
                .unwrap()
                .insert("system".into(), system);
        }
        if !tools.is_null() {
            body.as_object_mut()
                .unwrap()
                .insert("tool_choice".to_owned(), to_value!({"type": "auto"}));
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_owned(), tools);
        }
        if let Some(temperature) = options.temperature {
            body.as_object_mut()
                .unwrap()
                .insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = options.top_p {
            body.as_object_mut()
                .unwrap()
                .insert("top_p".to_owned(), top_p.into());
        }
        if let Some(top_k) = options.top_k {
            body.as_object_mut()
                .unwrap()
                .insert("top_k".to_owned(), (top_k as i64).into());
        }
        if let Some(ResponseFormat::JsonSchema(schema)) = &options.response_format {
            let wire_schema = self.marshal_response_schema(schema);
            body.as_object_mut().unwrap().insert(
                "output_config".into(),
                to_value!({"format": {"type": "json_schema", "schema": wire_schema}}),
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
pub struct AnthropicUnmarshal;

// 429 is always transient; billing exhaustion is reported as 402, outside this path.
impl super::QuotaClassifier for AnthropicUnmarshal {}

impl AnthropicUnmarshal {
    /// Maps an Anthropic `stop_reason` to a [`FinishReason`].
    fn parse_finish_reason(reason: &str) -> FinishReason {
        match reason {
            "end_turn" | "pause_turn" => FinishReason::Stop {},
            "max_tokens" => FinishReason::Length {},
            "tool_use" => FinishReason::ToolCall {},
            other => FinishReason::Refusal {
                reason: format!("reason: {}", other),
            },
        }
    }

    /// Maps an Anthropic role string to a [`Role`], defaulting to assistant.
    fn parse_role(s: &str) -> Role {
        match s {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Assistant,
        }
    }

    /// Parses an Anthropic `usage` object into [`TokenUsage`]. Returns `None`
    /// when `usage` is absent or not an object.
    fn parse_usage(usage: &Value) -> Option<TokenUsage> {
        let u = usage.as_object()?;
        Some(TokenUsage {
            input_tokens: u
                .get("input_tokens")
                .and_then(|v| v.as_integer())
                .unwrap_or(0) as u64,
            output_tokens: u
                .get("output_tokens")
                .and_then(|v| v.as_integer())
                .unwrap_or(0) as u64,
            cache_creation_input_tokens: u
                .get("cache_creation_input_tokens")
                .and_then(|v| v.as_integer())
                .map(|v| v as u64),
            cache_read_input_tokens: u
                .get("cache_read_input_tokens")
                .and_then(|v| v.as_integer())
                .map(|v| v as u64),
        })
    }
}

/// Parses one Anthropic SSE stream event into an incremental delta.
///
/// Each event type contributes a fragment that [`MessageDelta::accumulate`]
/// stitches together:
/// - `message_start`: role + initial usage (input / cache tokens)
/// - `content_block_start`: begins a `tool_use` function call (id + name)
/// - `content_block_delta`: a text / thinking / signature / tool-args fragment
/// - `message_delta`: `stop_reason` + final usage (output tokens)
/// - `ping` / `content_block_stop` / `message_stop`: no delta → `Ok(None)`
/// - `error`: fails with the server-reported error type + message
/// - any other (future) type: ignored → `Ok(None)`
impl Unmarshal<MessageDeltaOutput> for AnthropicUnmarshal {
    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let val: Value = serde_json::from_str(data)?;
        let ty = val.pointer("/type").and_then(|v| v.as_str()).unwrap_or("");

        let mut delta = MessageDelta::new();
        let mut finish_reason = None;
        let mut usage = None;

        match ty {
            "message_start" => {
                if let Some(role) = val.pointer("/message/role").and_then(|v| v.as_str()) {
                    delta = delta.with_role(Self::parse_role(role));
                }
                if let Some(u) = val.pointer("/message/usage") {
                    usage = Self::parse_usage(u);
                }
            }
            "content_block_start" => {
                if val.pointer("/content_block/type").and_then(|v| v.as_str()) == Some("tool_use") {
                    let id = val
                        .pointer("/content_block/id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_owned());
                    let name = val
                        .pointer("/content_block/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned();
                    delta = delta.with_tool_calls([PartDelta::Function {
                        id,
                        function: PartDeltaFunction::WithStringArgs {
                            name,
                            arguments: String::new(),
                        },
                    }]);
                }
            }
            "content_block_delta" => {
                if let Some(text) = val.pointer("/delta/text").and_then(|v| v.as_str()) {
                    delta = delta.with_contents([PartDelta::Text {
                        text: text.to_owned(),
                    }]);
                }
                if let Some(t) = val.pointer("/delta/thinking").and_then(|v| v.as_str()) {
                    delta.thinking = Some(t.to_owned());
                }
                if let Some(s) = val.pointer("/delta/signature").and_then(|v| v.as_str()) {
                    delta.signature = Some(s.to_owned());
                }
                if let Some(args) = val.pointer("/delta/partial_json").and_then(|v| v.as_str()) {
                    delta = delta.with_tool_calls([PartDelta::Function {
                        id: None,
                        function: PartDeltaFunction::WithStringArgs {
                            name: String::new(),
                            arguments: args.to_owned(),
                        },
                    }]);
                }
            }
            "message_delta" => {
                if let Some(reason) = val.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                    finish_reason = Some(Self::parse_finish_reason(reason));
                }
                if let Some(u) = val.pointer("/usage") {
                    usage = Self::parse_usage(u);
                }
            }
            // Control events carry no delta.
            "ping" | "content_block_stop" | "message_stop" => return Ok(None),
            // Mid-stream error (e.g. `overloaded_error`, rate limit). Surface the
            // server's own type + message rather than a generic "unknown event".
            "error" => {
                let err_type = val
                    .pointer("/error/type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let message = val
                    .pointer("/error/message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no message)");
                bail!("Anthropic stream error ({err_type}): {message}");
            }
            // Unknown / future event types carry no delta we understand; skip
            // them rather than aborting an otherwise-complete stream.
            other => {
                log::debug!("ignoring unknown Anthropic stream event type: {other}");
                return Ok(None);
            }
        }

        Ok(Some(MessageDeltaOutput {
            delta,
            finish_reason,
            usage,
            depth: None,
            source_agent: None,
        }))
    }

    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let root = val
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Root should be an object"))?;

        // Parse stop_reason -> finish_reason
        let finish_reason = root
            .get("stop_reason")
            .and_then(|v| v.as_str())
            .map(Self::parse_finish_reason);

        // Parse role
        let role = root
            .get("role")
            .and_then(|v| v.as_str())
            .map(Self::parse_role)
            .unwrap_or(Role::Assistant);

        let mut delta = MessageDelta::new().with_role(role);

        // Parse content array
        if let Some(contents) = root.get("content")
            && !contents.is_null()
        {
            if let Some(text) = contents.as_str() {
                delta = delta.with_contents([PartDelta::Text { text: text.into() }]);
            } else if let Some(contents_arr) = contents.as_array() {
                let mut text_parts: Vec<PartDelta> = Vec::new();
                let mut tool_call_parts: Vec<PartDelta> = Vec::new();
                let mut thinking: Option<String> = None;
                let mut signature: Option<String> = None;

                for content in contents_arr {
                    let Some(content_obj) = content.as_object() else {
                        bail!("Invalid content part");
                    };
                    let ty = content_obj
                        .get("type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    match ty {
                        "text" => {
                            if let Some(text) = content_obj.get("text").and_then(|v| v.as_str()) {
                                text_parts.push(PartDelta::Text { text: text.into() });
                            }
                        }
                        "thinking" => {
                            if let Some(t) = content_obj.get("thinking").and_then(|v| v.as_str()) {
                                thinking = Some(t.to_owned());
                            }
                            if let Some(s) = content_obj.get("signature").and_then(|v| v.as_str()) {
                                signature = Some(s.to_owned());
                            }
                        }
                        "tool_use" => {
                            let id = content_obj
                                .get("id")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_owned());
                            let name = content_obj
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_owned();
                            let arguments = content_obj
                                .get("input")
                                .map(|v| serde_json::to_string(v).unwrap_or_default())
                                .unwrap_or_default();
                            tool_call_parts.push(PartDelta::Function {
                                id,
                                function: PartDeltaFunction::WithStringArgs { name, arguments },
                            });
                        }
                        _ => {}
                    }
                }

                if !text_parts.is_empty() {
                    delta = delta.with_contents(text_parts);
                }
                if !tool_call_parts.is_empty() {
                    delta = delta.with_tool_calls(tool_call_parts);
                }
                if let Some(t) = thinking {
                    delta.thinking = Some(t);
                }
                if let Some(s) = signature {
                    delta.signature = Some(s);
                }
            } else {
                bail!("Invalid content");
            }
        }

        // Parse usage
        let usage = root.get("usage").and_then(Self::parse_usage);

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

    /// Register a one-off [`LangModelProvider`] under `provider_name` in the
    /// global registry and build the [`LangModel`] via
    /// [`LangModel::try_from_provider`].  Test fixtures only.
    fn build_anthropic_model(provider_name: &str, model: &str, api_key: String) -> LangModel {
        let elem = LangModelProviderElem::API {
            schema: LangModelAPISchema::Anthropic,
            url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
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
            schema: LangModelAPISchema::Anthropic,
            url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
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
    fn test_marshal_max_tokens_set() {
        with_req("claude-haiku-4-5", Some(512), |req| {
            let val = AnthropicMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            let max_tokens = body.as_object().unwrap().get("max_tokens").unwrap();
            assert_eq!(max_tokens.as_integer().unwrap(), 512);
        });
    }

    #[test]
    fn test_marshal_max_tokens_default() {
        with_req("claude-haiku-4-5", None, |req| {
            let val = AnthropicMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            let max_tokens = body.as_object().unwrap().get("max_tokens").unwrap();
            // Falls back to 8192 when not configured
            assert_eq!(max_tokens.as_integer().unwrap(), 8192);
        });
    }

    #[test]
    fn test_marshal_stream_flag() {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::Anthropic,
            url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions::default();
        let mut req = LangModelRequest {
            model: "claude-haiku-4-5",
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
        };

        // stream: false → no "stream" body key, no accept header.
        let val = AnthropicMarshal::default().marshal(&req);
        assert!(val.pointer("/body/stream").is_none());
        assert!(val.pointer("/header/accept").is_none());

        // stream: true → body "stream": true (the SSE trigger), plus an
        // `Accept: text/event-stream` header so intermediaries don't buffer.
        req.stream = true;
        let val = AnthropicMarshal::default().marshal(&req);
        assert_eq!(
            val.pointer("/body/stream").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            val.pointer("/header/accept").and_then(|v| v.as_str()),
            Some("text/event-stream")
        );
    }

    /// Feeds a sequence of SSE event payloads through `unmarshal_event`,
    /// accumulating into a final `MessageDeltaOutput`.
    fn accumulate_stream(inputs: &[&str]) -> MessageDeltaOutput {
        let mut u = AnthropicUnmarshal;
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
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world!"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let out = accumulate_stream(&inputs);
        assert_eq!(out.finish_reason, Some(FinishReason::Stop {}));
        let msg = out.finish().unwrap().message;
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.contents.len(), 1);
        assert_eq!(msg.contents[0].as_text(), Some("Hello world!"));
    }

    #[test]
    fn test_unmarshal_event_error_surfaces_message() {
        // A mid-stream `error` event must fail with the server's own type +
        // message, not a generic "unknown event type".
        let mut u = AnthropicUnmarshal;
        let event = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let err = u.unmarshal_event(event).unwrap_err().to_string();
        assert!(err.contains("overloaded_error"), "got: {err}");
        assert!(err.contains("Overloaded"), "got: {err}");
    }

    #[test]
    fn test_unmarshal_event_unknown_type_is_ignored() {
        // An unknown / future event type is skipped (no delta), not fatal.
        let mut u = AnthropicUnmarshal;
        let event = r#"{"type":"some_future_event","foo":"bar"}"#;
        assert!(u.unmarshal_event(event).unwrap().is_none());
    }

    #[test]
    fn test_unmarshal_event_thinking_stream() {
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"**Answering**\n\n"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"User is saying hello."}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"Ev4MCkYIBxgCKkDl5A"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hello"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":" world!"}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let out = accumulate_stream(&inputs);
        let msg = out.finish().unwrap().message;
        assert_eq!(
            msg.thinking.as_deref(),
            Some("**Answering**\n\nUser is saying hello.")
        );
        assert_eq!(msg.signature.as_deref(), Some("Ev4MCkYIBxgCKkDl5A"));
        assert_eq!(msg.contents.len(), 1);
        assert_eq!(msg.contents[0].as_text(), Some("Hello world!"));
    }

    #[test]
    fn test_unmarshal_event_tool_call_stream() {
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[]}}"#,
            r#"{"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\""}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"location"}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\":\""}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"Paris"}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":", France"}}"#,
            r#"{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"\"}"}}"#,
            r#"{"type":"content_block_stop"}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let out = accumulate_stream(&inputs);
        assert_eq!(out.finish_reason, Some(FinishReason::ToolCall {}));
        let msg = out.finish().unwrap().message;
        let tool_calls = msg.tool_calls.expect("expected tool_calls");
        assert_eq!(tool_calls.len(), 1);
        let (id, name, args) = tool_calls[0]
            .as_function()
            .expect("expected a function call");
        assert_eq!(id, "toolu_01");
        assert_eq!(name, "get_weather");
        assert_eq!(
            args.pointer("/location").and_then(|v| v.as_str()),
            Some("Paris, France")
        );
    }

    #[test]
    fn test_unmarshal_event_usage_merge() {
        // input_tokens/cache arrive in `message_start`; final output_tokens in
        // `message_delta`. Field-wise merge must keep both.
        let inputs = [
            r#"{"type":"message_start","message":{"type":"message","role":"assistant","content":[],"usage":{"input_tokens":100,"output_tokens":2,"cache_read_input_tokens":10}}}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":57}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let out = accumulate_stream(&inputs);
        let usage = out.usage.expect("expected usage");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 57);
        assert_eq!(usage.cache_read_input_tokens, Some(10));
    }

    /// End-to-end: `run_stream` against the real API yields multiple deltas that
    /// accumulate into a complete message terminated by a `finish_reason`.
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_text() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set");

        let model = build_anthropic_model("test_run_stream_text", "claude-haiku-4-5", api_key);
        let messages =
            vec![Message::new(Role::User).with_contents([Part::text("Reply with a short greeting.")])];

        let mut stream = model.run_stream(&messages, &[], &LangModelOptions::default());
        let mut acc = MessageDeltaOutput::new();
        let mut chunks = 0usize;
        while let Some(item) = stream.next().await {
            acc = acc.accumulate(item.unwrap()).unwrap();
            chunks += 1;
        }

        assert!(chunks > 1, "expected multiple streamed deltas, got {chunks}");
        let out = acc.finish().unwrap();
        assert_eq!(out.finish_reason, FinishReason::Stop {});
        assert!(
            out.message
                .contents
                .iter()
                .any(|p| p.as_text().is_some_and(|t| !t.is_empty())),
            "expected non-empty text in the streamed message"
        );
        // usage is split across message_start (input) and message_delta (output);
        // the field-wise merge must surface both.
        let usage = out.usage.expect("expected usage from the stream");
        assert!(usage.input_tokens > 0, "input_tokens should be > 0");
        assert!(usage.output_tokens > 0, "output_tokens should be > 0");
    }

    #[test]
    fn test_unmarshal_usage() {
        // All four fields present, including cache fields.
        let response = to_value!({
            "stop_reason": "end_turn",
            "role": "assistant",
            "content": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 10
            }
        });
        let usage = AnthropicUnmarshal::default()
            .unmarshal(response)
            .unwrap()
            .usage;
        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: Some(20),
                cache_read_input_tokens: Some(10),
            })
        );

        // Cache fields absent → None.
        let response = to_value!({
            "stop_reason": "end_turn",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 30, "output_tokens": 15}
        });
        let usage = AnthropicUnmarshal::default()
            .unmarshal(response)
            .unwrap()
            .usage;
        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 30,
                output_tokens: 15,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            })
        );
    }

    /// Verifies that tool_result content is an array of text/image blocks per the Anthropic spec.
    /// Unsupported part types (e.g. Part::Value) are filtered out.
    #[test]
    fn test_tool_result_content_marshaling() {
        // Part::Text → array with a single {"type":"text","text":"..."} block.
        let msg_text = Message::new(Role::Tool)
            .with_id("call_1")
            .with_contents([Part::text("tool output")]);
        let val = AnthropicMarshal.marshal(&msg_text);
        let content = val
            .pointer("/content/0/content")
            .expect("content/0/content must exist")
            .as_array()
            .expect("content must be an array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].pointer("/type").and_then(|v| v.as_str()),
            Some("text")
        );
        assert_eq!(
            content[0].pointer("/text").and_then(|v| v.as_str()),
            Some("tool output")
        );

        // Part::Image (embedded) → array with a single {"type":"image","source":{"type":"base64",...}} block.
        let img_bytes = Bytes::from(vec![0xFFu8, 0xD8, 0xFF]);
        let msg_img = Message::new(Role::Tool)
            .with_id("call_2")
            .with_contents([Part::image_embedded("image/jpeg", img_bytes.clone()).unwrap()]);
        let val = AnthropicMarshal.marshal(&msg_img);
        let content = val
            .pointer("/content/0/content")
            .expect("content/0/content must exist")
            .as_array()
            .expect("content must be an array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].pointer("/type").and_then(|v| v.as_str()),
            Some("image")
        );
        assert_eq!(
            content[0].pointer("/source/type").and_then(|v| v.as_str()),
            Some("base64")
        );
        assert_eq!(
            content[0]
                .pointer("/source/media_type")
                .and_then(|v| v.as_str()),
            Some("image/jpeg")
        );
        assert_eq!(
            content[0].pointer("/source/data").and_then(|v| v.as_str()),
            Some(img_bytes.base64().as_str())
        );

        // Part::Image (url) → array with a single {"type":"image","source":{"type":"url",...}} block.
        let msg_img_url = Message::new(Role::Tool)
            .with_id("call_3")
            .with_contents([Part::image_url("https://example.com/img.png".to_string()).unwrap()]);
        let val = AnthropicMarshal.marshal(&msg_img_url);
        let content = val
            .pointer("/content/0/content")
            .expect("content/0/content must exist")
            .as_array()
            .expect("content must be an array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].pointer("/source/type").and_then(|v| v.as_str()),
            Some("url")
        );
        assert_eq!(
            content[0].pointer("/source/url").and_then(|v| v.as_str()),
            Some("https://example.com/img.png")
        );

        // Part::Value(String) → {"type":"text","text":"..."} block; no double-encoding.
        let msg_str = Message::new(Role::Tool)
            .with_id("call_4")
            .with_contents([Part::value(Value::string("ok".to_string()))]);
        let val = AnthropicMarshal.marshal(&msg_str);
        let content = val
            .pointer("/content/0/content")
            .expect("content/0/content must exist")
            .as_array()
            .expect("content must be an array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].pointer("/type").and_then(|v| v.as_str()),
            Some("text")
        );
        assert_eq!(
            content[0].pointer("/text").and_then(|v| v.as_str()),
            Some("ok"),
            "Part::Value(String) must not be double-encoded"
        );

        // Part::Value(Object) → JSON-encoded as {"type":"text","text":"{...}"} block.
        let msg_obj = Message::new(Role::Tool)
            .with_id("call_5")
            .with_contents([Part::value(to_value!({"temp": 25}))]);
        let val = AnthropicMarshal.marshal(&msg_obj);
        let content = val
            .pointer("/content/0/content")
            .expect("content/0/content must exist")
            .as_array()
            .expect("content must be an array");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].pointer("/type").and_then(|v| v.as_str()),
            Some("text")
        );
        assert_eq!(
            content[0].pointer("/text").and_then(|v| v.as_str()),
            Some(r#"{"temp":25}"#),
            "Part::Value(Object) must be JSON-encoded into a text block"
        );
    }

    #[test]
    fn test_marshal_response_format_absent() {
        with_req("claude-haiku-4-5", None, |req| {
            let val = AnthropicMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(body.as_object().unwrap().get("output_config").is_none());
        });
    }

    #[test]
    fn test_marshal_response_format_json_schema() {
        let schema = to_value!({"type": "object", "properties": {"name": {"type": "string"}}});
        let fmt = ResponseFormat::json_schema(schema.clone().into()).unwrap();
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let provider = LangModelProviderElem::API {
            schema: LangModelAPISchema::Anthropic,
            url: Url::parse("https://api.anthropic.com/v1/messages").unwrap(),
            api_key: None,
        };
        let options = LangModelOptions {
            response_format: Some(fmt),
            ..Default::default()
        };
        let req = LangModelRequest {
            model: "claude-haiku-4-5",
            messages: &messages,
            tools: &tools,
            provider: &provider,
            options: &options,
            stream: false,
        };
        let val = AnthropicMarshal::default().marshal(&req);
        let body = val.as_object().unwrap().get("body").unwrap();
        let fmt_type = body
            .pointer("/output_config/format/type")
            .and_then(|v| v.as_str());
        assert_eq!(fmt_type, Some("json_schema"));
        let fmt_schema = body.pointer("/output_config/format/schema").unwrap();
        assert_eq!(
            fmt_schema.pointer("/type").and_then(|v| v.as_str()),
            Some("object")
        );
    }

    /// Verifies structured output via response_format: the model returns valid JSON matching the schema.
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_run_response_format_json_schema() {
        dotenvy::dotenv().ok();
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in .env");

        // Intentionally omit additionalProperties to verify normalize_schema adds it.
        let schema = to_value!({
            "type": "object",
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"}
            },
            "required": ["country", "capital"]
        });

        let model = build_anthropic_model(
            "test_run_response_format_json_schema",
            "claude-haiku-4-5",
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

    /// Verifies that max_tokens is respected by the Anthropic API (stop_reason: max_tokens).
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key =
            std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY must be set in .env");

        let model = build_anthropic_model("test_run_max_tokens", "claude-haiku-4-5", api_key);
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

    /// Verifies that an image embedded in a Role::Tool message is accepted by the Anthropic API
    /// and that the model can respond after seeing the image in a tool result.
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_tool_result_with_image() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();

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
            build_anthropic_model("test_tool_result_with_image", "claude-haiku-4-5", api_key);

        let messages = vec![
            Message::new(Role::User).with_contents([Part::text(
                "Describe the image returned by the file_read tool.",
            )]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "toolu_test_001",
                "file_read",
                to_value!({"path": "/tmp/test.png"}),
            )]),
            Message::new(Role::Tool)
                .with_id("toolu_test_001")
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
