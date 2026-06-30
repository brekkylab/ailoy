use url::Url;

use super::{super::response_format::ResponseSchemaMarshal, openai::is_openai_reasoning_model};
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
    pub fn grok(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: Url::parse("https://api.x.ai/v1/chat/completions").unwrap(),
            api_key: Some(api_key),
        }
    }

    pub fn deepseek(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: Url::parse("https://api.deepseek.com/chat/completions").unwrap(),
            api_key: Some(api_key),
        }
    }

    pub fn kimi(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: Url::parse("https://api.moonshot.ai/v1/chat/completions").unwrap(),
            api_key: Some(api_key),
        }
    }

    pub fn chat_completion(
        url: &str,
        api_key: Option<String>,
    ) -> anyhow::Result<LangModelProviderElem> {
        Ok(LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: Url::parse(url)?,
            api_key,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatCompletionMarshal;

impl ResponseSchemaMarshal for ChatCompletionMarshal {}

impl Marshal<Message> for ChatCompletionMarshal {
    fn marshal(&mut self, item: &Message) -> Value {
        let part_to_value = |part: &Part| -> Value {
            match part {
                Part::Text { text } => to_value!({"type": "text", "text": text}),
                Part::Function {
                    id,
                    function: PartFunction { name, arguments },
                } => {
                    to_value!({"type": "function", "id": id, "function": {"name": name, "arguments": serde_json::to_string(&arguments).unwrap()}})
                }
                Part::Value { value } => {
                    to_value!({"type": "text", "text": serde_json::to_string(&value).unwrap()})
                }
                Part::Image { image } => {
                    let url = match image {
                        PartImage::Embedded { mime_type, data } => {
                            format!("data:{};base64,{}", mime_type, data.base64())
                        }
                        PartImage::Url { url } => url.clone(),
                    };
                    to_value!({"type": "image_url", "image_url": {"url": url}})
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
        // DeepSeek thinking-mode endpoint requires the prior `reasoning_content`
        // to be replayed on every follow-up turn.
        if item.role == Role::Assistant
            && let Some(thinking) = &item.thinking
            && !thinking.is_empty()
        {
            rv.as_object_mut()
                .unwrap()
                .insert("reasoning_content".into(), thinking.clone().into());
        }
        if !item.contents.is_empty() {
            let contents: Vec<Value> = item
                .contents
                .iter()
                // Some ChatCompletion-compatible backends (e.g. Kimi)
                // reject content arrays containing an empty-text part
                .filter(|p| !matches!(p, Part::Text { text } if text.is_empty()))
                .map(|p| {
                    // ChatCompletion backends don't reliably accept images in tool results;
                    // substitute a text label so the model knows an image was returned.
                    if item.role == Role::Tool {
                        match p {
                            Part::Image {
                                image: PartImage::Embedded { mime_type, .. },
                            } => {
                                return part_to_value(&Part::text(format!(
                                    "[image: {}]",
                                    mime_type
                                )));
                            }
                            Part::Image {
                                image: PartImage::Url { url },
                            } => {
                                return part_to_value(&Part::text(format!("[image at {}]", url)));
                            }
                            _ => {}
                        }
                    }
                    part_to_value(p)
                })
                .collect();
            if !contents.is_empty() {
                rv.as_object_mut()
                    .unwrap()
                    .insert("content".into(), contents.into());
            }
        }
        if let Some(tool_calls) = &item.tool_calls
            && !tool_calls.is_empty()
        {
            rv.as_object_mut().unwrap().insert(
                "tool_calls".into(),
                tool_calls
                    .iter()
                    .map(part_to_value)
                    .collect::<Vec<_>>()
                    .into(),
            );
        }
        rv
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

impl Marshal<LangModelRequest<'_>> for ChatCompletionMarshal {
    fn marshal(&mut self, req: &LangModelRequest<'_>) -> Value {
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
        if let Some(max_tokens) = req.max_tokens {
            body.as_object_mut().unwrap().insert(
                "max_completion_tokens".to_owned(),
                (max_tokens as i64).into(),
            );
        }
        // OpenAI reasoning models (o-series, gpt-5) reject temperature/top_p; drop them silently.
        if let Some(temperature) = req.temperature
            && !is_openai_reasoning_model(req.model)
        {
            body.as_object_mut()
                .unwrap()
                .insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = req.top_p
            && !is_openai_reasoning_model(req.model)
        {
            body.as_object_mut()
                .unwrap()
                .insert("top_p".to_owned(), top_p.into());
        }
        // top_k is not part of the OpenAI ChatCompletion spec; intentionally ignored.
        if let Some(ResponseFormat::JsonSchema(schema)) = req.response_format {
            let wire_schema = self.marshal_response_schema(schema);
            body.as_object_mut().unwrap().insert(
                "response_format".into(),
                to_value!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": wire_schema,
                        "strict": true
                    }
                }),
            );
        }
        body.as_object_mut()
            .unwrap()
            .retain(|_key, value| !value.is_null());

        if req.stream {
            let body_obj = body.as_object_mut().unwrap();
            body_obj.insert("stream".into(), true.into());
            // Request token usage in the final (empty-choices) SSE chunk; without
            // this, ChatCompletion streaming reports no usage. Supported by OpenAI
            // and the major OpenAI-compatible backends (Grok / DeepSeek / Kimi).
            body_obj.insert("stream_options".into(), to_value!({"include_usage": true}));
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
pub struct ChatCompletionUnmarshal;

// Shared by arbitrary OpenAI-compatible providers, so no provider-specific
// quota signal can be assumed; treat 429 as transient and retry.
impl super::QuotaClassifier for ChatCompletionUnmarshal {}

/// Parses one ChatCompletion SSE chunk (`chat.completion.chunk`) into a delta:
/// the incremental `choices[0].delta` (role / content / reasoning_content /
/// tool_call fragments), `finish_reason`, and usage from the final chunk.
/// `[DONE]` carries no delta.
impl super::StreamUnmarshal for ChatCompletionUnmarshal {
    fn unmarshal_event(&self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        // OpenAI-compatible streams end with a `[DONE]` sentinel (not JSON).
        if data.trim() == "[DONE]" {
            return Ok(None);
        }
        let val: Value = serde_json::from_str(data)?;

        let usage = Self::parse_usage(&val);

        // The usage-only final chunk (stream_options.include_usage) has empty
        // `choices`; emit a usage-only delta.
        let Some(choice) = val.pointer("/choices/0") else {
            return Ok(usage.map(|u| MessageDeltaOutput {
                delta: MessageDelta::new(),
                finish_reason: None,
                usage: Some(u),
            }));
        };

        let finish_reason = choice
            .pointer("/finish_reason")
            .filter(|v| !v.is_null())
            .map(Self::parse_finish_reason);

        let mut delta = MessageDelta::new();

        // role: usually present only on the first chunk. Default to Assistant
        // when a chunk omits it (some OpenAI-compatible backends do), mirroring
        // the non-streaming `unmarshal` default — otherwise a role-less stream
        // accumulates to a message with no role and `finish()` bails with
        // "Role not specified".
        let role = choice
            .pointer("/delta/role")
            .and_then(|v| v.as_str())
            .map(|r| r.parse::<Role>().unwrap_or(Role::Assistant))
            .unwrap_or(Role::Assistant);
        delta = delta.with_role(role);

        // content: incremental text (empty on the first chunk, null on the last).
        if let Some(text) = choice.pointer("/delta/content").and_then(|v| v.as_str())
            && !text.is_empty()
        {
            delta = delta.with_contents([PartDelta::Text {
                text: text.to_owned(),
            }]);
        }

        // reasoning_content: DeepSeek streams thinking incrementally.
        if let Some(t) = choice
            .pointer("/delta/reasoning_content")
            .and_then(|v| v.as_str())
            && !t.is_empty()
        {
            delta.thinking = Some(t.to_owned());
        }

        // tool_calls: the first delta of each call carries id + name; later
        // deltas carry only an `arguments` fragment (id/name absent) and merge
        // into the in-progress call during accumulation.
        if let Some(tcs) = choice
            .pointer("/delta/tool_calls")
            .and_then(|v| v.as_array())
            && !tcs.is_empty()
        {
            let parts: Vec<PartDelta> = tcs
                .iter()
                .map(|tc| {
                    let id = tc
                        .pointer("/id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_owned());
                    let name = tc
                        .pointer("/function/name")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned();
                    let arguments = tc
                        .pointer("/function/arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned();
                    PartDelta::Function {
                        id,
                        function: PartDeltaFunction::WithStringArgs { name, arguments },
                    }
                })
                .collect();
            delta = delta.with_tool_calls(parts);
        }

        Ok(Some(MessageDeltaOutput {
            delta,
            finish_reason,
            usage,
        }))
    }
}

impl ChatCompletionUnmarshal {
    fn parse_finish_reason(val: &Value) -> FinishReason {
        match val.as_str() {
            Some("stop") => FinishReason::Stop {},
            Some("length") => FinishReason::Length {},
            Some("tool_calls") => FinishReason::ToolCall {},
            Some("content_filter") => FinishReason::Refusal {
                reason: "content_filter".into(),
            },
            _ => FinishReason::Stop {},
        }
    }

    fn parse_content(content: &Value) -> Vec<PartDelta> {
        match content {
            Value::String(text) => vec![PartDelta::Text { text: text.clone() }],
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| {
                    let ty = part.pointer("/type")?.as_str()?;
                    match ty {
                        "text" => {
                            let text = part.pointer("/text")?.as_str()?;
                            Some(PartDelta::Text {
                                text: text.to_owned(),
                            })
                        }
                        _ => None,
                    }
                })
                .collect(),
            _ => vec![],
        }
    }

    fn parse_tool_calls(tool_calls: &Value) -> Vec<PartDelta> {
        let Some(arr) = tool_calls.as_array() else {
            return vec![];
        };
        arr.iter()
            .filter_map(|tc| {
                let id = tc.pointer("/id")?.as_str().map(|s| s.to_owned());
                let name = tc.pointer("/function/name")?.as_str()?.to_owned();
                let arguments = tc
                    .pointer("/function/arguments")?
                    .as_str()
                    .unwrap_or("")
                    .to_owned();
                Some(PartDelta::Function {
                    id,
                    function: PartDeltaFunction::WithStringArgs { name, arguments },
                })
            })
            .collect()
    }

    /// Parses the ChatCompletion `usage` object (`prompt_tokens` /
    /// `completion_tokens`). Returns `None` when absent or null.
    fn parse_usage(root: &Value) -> Option<TokenUsage> {
        let u = root
            .pointer("/usage")
            .filter(|u| !u.is_null())?
            .as_object()?;
        Some(TokenUsage {
            input_tokens: u
                .get("prompt_tokens")
                .and_then(|v| v.as_integer())
                .unwrap_or(0) as u64,
            output_tokens: u
                .get("completion_tokens")
                .and_then(|v| v.as_integer())
                .unwrap_or(0) as u64,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        })
    }
}

impl Unmarshal<MessageDeltaOutput> for ChatCompletionUnmarshal {
    fn unmarshal(&mut self, val: Value) -> anyhow::Result<MessageDeltaOutput> {
        let choice = val
            .pointer("/choices/0")
            .ok_or_else(|| anyhow::anyhow!("Missing 'choices[0]' in response"))?;

        // Parse finish_reason
        let finish_reason = choice
            .pointer("/finish_reason")
            .map(Self::parse_finish_reason);

        // Check for refusal
        let message = choice
            .pointer("/message")
            .ok_or_else(|| anyhow::anyhow!("Missing 'message' in choice"))?;

        if let Some(refusal) = message.pointer("/refusal")
            && !refusal.is_null()
        {
            let reason = refusal.as_str().unwrap_or("unknown").to_owned();
            return Ok(MessageDeltaOutput {
                delta: MessageDelta::new().with_role(Role::Assistant),
                finish_reason: Some(FinishReason::Refusal { reason }),
                usage: None,
            });
        }

        // Parse role
        let role: Role = message
            .pointer("/role")
            .and_then(|v| v.as_str())
            .unwrap_or("assistant")
            .parse()
            .unwrap_or(Role::Assistant);

        // Parse content
        let contents = message
            .pointer("/content")
            .filter(|v| !v.is_null())
            .map(Self::parse_content)
            .unwrap_or_default();

        // Parse tool_calls
        let tool_calls = message
            .pointer("/tool_calls")
            .filter(|v| !v.is_null())
            .map(Self::parse_tool_calls)
            .unwrap_or_default();

        // DeepSeek thinking-mode responses include `reasoning_content` as a
        // sibling of `content`. Map it onto ailoy's canonical `thinking` field
        // so the same value gets replayed on follow-up turns.
        let thinking = message
            .pointer("/reasoning_content")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_owned());

        let mut delta = MessageDelta::new().with_role(role);
        if !contents.is_empty() {
            delta = delta.with_contents(contents);
        }
        if !tool_calls.is_empty() {
            delta = delta.with_tool_calls(tool_calls);
        }
        if let Some(t) = thinking {
            delta.thinking = Some(t);
        }

        // Parse usage (Chat Completion: usage.prompt_tokens / completion_tokens)
        let usage = Self::parse_usage(&val);

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

    use super::super::StreamUnmarshal;
    use super::*;
    use crate::{
        lang_model::{LangModel, LangModelAPISchema, LangModelOptions, LangModelProviderElem},
        message::{Delta, FinishReason, Message, MessageDeltaOutput, Part, Role},
        tool::ToolDesc,
    };

    #[test]
    fn test_marshal_stream_options() {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
        let api_key: Option<String> = None;
        let mut req = LangModelRequest {
            model: "gpt-4.1-mini",
            messages: &messages,
            tools: &tools,
            url: &url,
            api_key: &api_key,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            response_format: None,
            stream: false,
        };

        // stream: false → no stream / stream_options.
        let val = ChatCompletionMarshal::default().marshal(&req);
        assert!(val.pointer("/body/stream").is_none());
        assert!(val.pointer("/body/stream_options").is_none());

        // stream: true → stream + stream_options.include_usage so usage is reported.
        req.stream = true;
        let val = ChatCompletionMarshal::default().marshal(&req);
        assert_eq!(
            val.pointer("/body/stream").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            val.pointer("/body/stream_options/include_usage")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    /// Feeds SSE chunk payloads through `unmarshal_event`, accumulating to a
    /// final `MessageDeltaOutput`.
    fn accumulate_stream(inputs: &[&str]) -> MessageDeltaOutput {
        let u = ChatCompletionUnmarshal;
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
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}"#,
            r#"[DONE]"#,
        ];
        let result = accumulate_stream(&inputs).finish().unwrap();
        assert_eq!(result.finish_reason, FinishReason::Stop {});
        let usage = result.usage.expect("expected usage from final chunk");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(result.message.role, Role::Assistant);
        assert_eq!(result.message.contents.len(), 1);
        assert_eq!(result.message.contents[0].as_text(), Some("Hello world!"));
    }

    #[test]
    fn test_unmarshal_event_tool_call_stream() {
        let inputs = [
            r#"{"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\":\"Paris\"}"}}]},"finish_reason":null}]}"#,
            r#"{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            r#"[DONE]"#,
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

    /// End-to-end: `run_stream` over OpenAI chat/completions yields multiple
    /// deltas that accumulate into a complete message.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_text() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

        let model = LangModel::new(
            "gpt-4.1-mini".to_string(),
            LangModelProvider::chat_completion(
                "https://api.openai.com/v1/chat/completions",
                Some(api_key),
            )
            .unwrap(),
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
        // stream_options.include_usage → usage reported on the final chunk.
        let usage = result.usage.expect("expected usage from the stream");
        assert!(usage.input_tokens > 0, "input_tokens should be > 0");
        assert!(usage.output_tokens > 0, "output_tokens should be > 0");
    }

    fn with_req<F, R>(model: &str, max_tokens: Option<u64>, f: F) -> R
    where
        F: FnOnce(&LangModelRequest<'_>) -> R,
    {
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
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
            stream: false,
        };
        f(&req)
    }

    #[test]
    fn test_marshal_max_tokens_set() {
        with_req("gpt-4.1-mini", Some(256), |req| {
            let val = ChatCompletionMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            let max_tokens = body
                .as_object()
                .unwrap()
                .get("max_completion_tokens")
                .unwrap();
            assert_eq!(max_tokens.as_integer().unwrap(), 256);
        });
    }

    #[test]
    fn test_marshal_max_tokens_absent_when_none() {
        with_req("gpt-4.1-mini", None, |req| {
            let val = ChatCompletionMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(
                body.as_object()
                    .unwrap()
                    .get("max_completion_tokens")
                    .is_none()
            );
        });
    }

    #[test]
    fn test_marshal_response_format_absent() {
        with_req("gpt-4.1-mini", None, |req| {
            let val = ChatCompletionMarshal::default().marshal(req);
            let body = val.as_object().unwrap().get("body").unwrap();
            assert!(body.as_object().unwrap().get("response_format").is_none());
        });
    }

    #[test]
    fn test_marshal_response_format_json_schema() {
        let schema = to_value!({"type": "object", "properties": {"score": {"type": "integer"}}});
        let fmt = ResponseFormat::json_schema(schema.clone().into()).unwrap();
        let messages: Vec<Message> = vec![];
        let tools: Vec<ToolDesc> = vec![];
        let url = Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
        let api_key: Option<String> = None;
        let req = LangModelRequest {
            model: "gpt-4.1-mini",
            messages: &messages,
            tools: &tools,
            url: &url,
            api_key: &api_key,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            response_format: Some(&fmt),
            stream: false,
        };
        let val = ChatCompletionMarshal::default().marshal(&req);
        let body = val.as_object().unwrap().get("body").unwrap();
        let rf = body.as_object().unwrap().get("response_format").unwrap();
        assert_eq!(
            rf.pointer("/type").and_then(|v| v.as_str()),
            Some("json_schema")
        );
        assert_eq!(
            rf.pointer("/json_schema/name").and_then(|v| v.as_str()),
            Some("response")
        );
        assert_eq!(
            rf.pointer("/json_schema/strict").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(
            rf.pointer("/json_schema/schema/type")
                .and_then(|v| v.as_str()),
            Some("object")
        );
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

        let model = LangModel::new(
            "gpt-4.1-mini".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::ChatCompletion,
                url: Url::parse("https://api.openai.com/v1/chat/completions").unwrap(),
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

    /// Verifies that max_tokens is respected by the ChatCompletion API (finish_reason: length).
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_max_tokens() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = LangModel::new(
            "gpt-4.1-mini".to_string(),
            LangModelProviderElem::API {
                schema: LangModelAPISchema::ChatCompletion,
                url: Url::parse("https://api.openai.com/v1/chat/completions").unwrap(),
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
                    max_tokens: Some(32),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Length {});
    }
}
