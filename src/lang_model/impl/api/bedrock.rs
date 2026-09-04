//! Amazon Bedrock: the endpoint shared by every schema served there, plus the
//! model-agnostic Converse wire format.

use anyhow::bail;
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
    /// Bedrock over the Converse API, which speaks to every model family in
    /// the region with one body format.
    ///
    /// `model` must be an id Bedrock accepts for on-demand throughput, e.g. the
    /// inference-profile id `global.anthropic.claude-sonnet-5`.
    pub fn bedrock(region: impl AsRef<str>, api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::Bedrock,
            url: runtime_url(region.as_ref()),
            api_key: Some(api_key),
        }
    }
}

/// Runtime base for a region; the registered `url` for both Bedrock schemas.
fn runtime_url(region: &str) -> Url {
    Url::parse(&format!("https://bedrock-runtime.{region}.amazonaws.com")).unwrap()
}

/// `<base>/model/<model>/<action>`. `/` in the model id (application
/// inference-profile ARNs) is escaped so it stays one path segment.
pub(super) fn model_url(base: &Url, model: &str, action: &str) -> String {
    format!(
        "{}/model/{}/{action}",
        base.as_str().trim_end_matches('/'),
        model.replace('/', "%2F")
    )
}

/// Bearer-token headers. Streaming responses are the binary event stream, so
/// `accept` names it explicitly.
pub(super) fn headers(api_key: Option<&str>, stream: bool) -> Value {
    let mut header = to_value!({"content-type": "application/json"});
    let h = header.as_object_mut().unwrap();
    if let Some(api_key) = api_key {
        h.insert("Authorization".into(), format!("Bearer {api_key}").into());
    }
    if stream {
        h.insert("accept".into(), "application/vnd.amazon.eventstream".into());
    }
    header
}

/// Rejects requests Bedrock would 400 on, before anything is sent. Marshals
/// return a bare `Value`, so this is where a Bedrock-only limitation becomes an
/// error instead of a silently dropped field.
pub(in crate::lang_model) fn validate_request(req: &LangModelRequest<'_>) -> anyhow::Result<()> {
    if req.options.response_format.is_some() {
        bail!("Bedrock does not support response_format (structured output)");
    }
    let has_url_image = req.messages.iter().any(|m| {
        m.contents.iter().any(|p| {
            matches!(
                p,
                Part::Image {
                    image: PartImage::Url { .. }
                }
            )
        })
    });
    if has_url_image {
        bail!("Bedrock does not accept image URLs; embed the image bytes instead");
    }
    Ok(())
}

/// Converse request marshal. Model-agnostic, so nothing here depends on the
/// model family behind the id.
#[derive(Clone, Debug, Default)]
pub struct BedrockMarshal;

/// Converse image `format` from a MIME type; Converse accepts png/jpeg/gif/webp.
fn image_format(mime_type: &str) -> &str {
    match mime_type.strip_prefix("image/").unwrap_or(mime_type) {
        "jpg" => "jpeg",
        other => other,
    }
}

fn marshal_image(image: &PartImage) -> Value {
    match image {
        PartImage::Embedded { mime_type, data } => to_value!({
            "image": {
                "format": image_format(mime_type),
                "source": {"bytes": data.base64()},
            }
        }),
        // Rejected up front by `validate_request`; keep the shape valid anyway.
        PartImage::Url { url } => to_value!({"text": url}),
    }
}

/// A content block for a user/assistant message. Converse rejects empty text
/// blocks, so those yield `None`.
fn marshal_part(part: &Part) -> Option<Value> {
    match part {
        Part::Text { text } if text.is_empty() => None,
        Part::Text { text } => Some(to_value!({"text": text})),
        Part::Function {
            id,
            function: PartFunction { name, arguments },
        } => Some(to_value!({
            "toolUse": {"toolUseId": id, "name": name, "input": arguments.clone()}
        })),
        // Converse has no free-form JSON block outside tool results.
        Part::Value { value } => Some(to_value!({
            "text": serde_json::to_string(value).unwrap_or_default()
        })),
        Part::Image { image } => Some(marshal_image(image)),
    }
}

/// A `toolResult` content block; here JSON may travel as-is.
fn marshal_tool_result_part(part: &Part) -> Option<Value> {
    match part {
        Part::Value { value } if value.is_object() => Some(to_value!({"json": value.clone()})),
        Part::Value { value } => Some(to_value!({
            "text": match value {
                Value::String(s) => s.clone(),
                other => serde_json::to_string(other).unwrap_or_default(),
            }
        })),
        Part::Function { .. } => None,
        other => marshal_part(other),
    }
}

fn marshal_message(item: &Message, include_thinking: bool) -> Value {
    if item.role == Role::Tool {
        let mut content: Vec<Value> = item
            .contents
            .iter()
            .filter_map(marshal_tool_result_part)
            .collect();
        if content.is_empty() {
            content.push(to_value!({"text": "(no output)"}));
        }
        return to_value!({
            "role": "user",
            "content": [{
                "toolResult": {
                    "toolUseId": item.id.clone().expect("Tool call id must exist."),
                    "content": content,
                }
            }]
        });
    }

    let mut contents = Vec::<Value>::new();
    if let Some(thinking) = &item.thinking
        && !thinking.is_empty()
        && include_thinking
    {
        let mut text = to_value!({"text": thinking});
        if let Some(sig) = &item.signature {
            text.as_object_mut()
                .unwrap()
                .insert("signature".into(), sig.into());
        }
        contents.push(to_value!({"reasoningContent": {"reasoningText": text}}));
    }
    contents.extend(item.contents.iter().filter_map(marshal_part));
    contents.extend(item.tool_calls.iter().flatten().filter_map(marshal_part));

    to_value!({"role": item.role.to_string(), "content": contents})
}

/// Marshals the conversation, folding tool results into `user` turns and
/// merging consecutive same-role turns: Converse requires strict user /
/// assistant alternation, and several tool results after one assistant turn
/// would otherwise be several `user` messages in a row.
///
/// Thinking is replayed only for assistant turns after the last user message,
/// the same rule the Anthropic marshal applies.
fn marshal_messages(messages: &[Message]) -> Value {
    let last_user_index = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len());
    let mut out: Vec<Value> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::System {
            continue;
        }
        let mut v = marshal_message(msg, i > last_user_index);
        let same_role = out
            .last()
            .and_then(|prev| prev.pointer("/role"))
            .is_some_and(|r| r == v.pointer("/role").unwrap());
        if same_role {
            let extra = v.pointer_mut("/content").unwrap().as_array_mut().unwrap();
            out.last_mut()
                .unwrap()
                .pointer_mut("/content")
                .unwrap()
                .as_array_mut()
                .unwrap()
                .append(extra);
        } else {
            out.push(v);
        }
    }
    Value::Array(out)
}

impl Marshal<Message> for BedrockMarshal {
    fn marshal(&self, item: &Message) -> Value {
        marshal_message(item, true)
    }
}

impl Marshal<ToolDesc> for BedrockMarshal {
    fn marshal(&self, item: &ToolDesc) -> Value {
        let mut spec = to_value!({
            "name": &item.name,
            "inputSchema": {"json": item.parameters.clone()},
        });
        // Converse rejects an empty description, so omit rather than blank it.
        if let Some(desc) = item.description.as_deref().filter(|d| !d.is_empty()) {
            spec.as_object_mut()
                .unwrap()
                .insert("description".into(), desc.into());
        }
        to_value!({"toolSpec": spec})
    }
}

impl Marshal<LangModelRequest<'_>> for BedrockMarshal {
    fn marshal(&self, req: &LangModelRequest<'_>) -> Value {
        let LangModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;

        let action = if req.stream {
            "converse-stream"
        } else {
            "converse"
        };
        let url = model_url(url, req.model, action);
        let header = headers(api_key.as_deref(), req.stream);

        let mut body = to_value!({"messages": marshal_messages(req.messages)});
        let body_obj = body.as_object_mut().unwrap();

        let system: Vec<Value> = req
            .messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter())
            .filter_map(|p| p.as_text())
            .filter(|t| !t.is_empty())
            .map(|t| to_value!({"text": t}))
            .collect();
        if !system.is_empty() {
            body_obj.insert("system".into(), Value::Array(system));
        }

        if !req.tools.is_empty() {
            body_obj.insert(
                "toolConfig".into(),
                to_value!({
                    "tools": self.marshal(req.tools),
                    "toolChoice": {"auto": {}},
                }),
            );
        }

        // No default maxTokens: the ceiling differs per model family and an
        // over-limit value is a 400, so leave it to the model unless asked.
        let mut inference = Value::object_empty();
        let inf = inference.as_object_mut().unwrap();
        if let Some(max_tokens) = options.max_tokens {
            inf.insert("maxTokens".into(), (max_tokens as i64).into());
        }
        if let Some(temperature) = options.temperature {
            inf.insert("temperature".into(), temperature.into());
        }
        if let Some(top_p) = options.top_p {
            inf.insert("topP".into(), top_p.into());
        }
        if !inf.is_empty() {
            body_obj.insert("inferenceConfig".into(), inference);
        }
        // Converse has no portable top-k; `top_k` is the Anthropic/Cohere
        // spelling and is passed through as a model-specific field.
        if let Some(top_k) = options.top_k {
            body_obj.insert(
                "additionalModelRequestFields".into(),
                to_value!({"top_k": top_k as i64}),
            );
        }

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

/// Converse response and `ConverseStream` event parser.
#[derive(Clone, Debug, Default)]
pub struct BedrockUnmarshal;

// Bedrock throttling is a 429 `ThrottlingException` and always transient.
impl super::QuotaClassifier for BedrockUnmarshal {}

impl BedrockUnmarshal {
    fn parse_finish_reason(reason: &str) -> FinishReason {
        match reason {
            "end_turn" | "stop_sequence" => FinishReason::Stop {},
            "max_tokens" => FinishReason::Length {},
            "tool_use" => FinishReason::ToolCall {},
            other => FinishReason::Refusal {
                reason: format!("reason: {other}"),
            },
        }
    }

    fn parse_role(s: &str) -> Role {
        match s {
            "user" => Role::User,
            _ => Role::Assistant,
        }
    }

    fn parse_usage(usage: &Value) -> Option<TokenUsage> {
        let u = usage.as_object()?;
        let int = |k: &str| u.get(k).and_then(|v| v.as_integer());
        Some(TokenUsage {
            input_tokens: int("inputTokens").unwrap_or(0) as u64,
            output_tokens: int("outputTokens").unwrap_or(0) as u64,
            cache_creation_input_tokens: int("cacheWriteInputTokens").map(|v| v as u64),
            cache_read_input_tokens: int("cacheReadInputTokens").map(|v| v as u64),
        })
    }

    fn tool_use_delta(id: Option<String>, name: &str, arguments: String) -> PartDelta {
        PartDelta::Function {
            id,
            function: PartDeltaFunction::WithStringArgs {
                name: name.to_owned(),
                arguments,
            },
        }
    }
}

/// Events arrive as `{"<eventType>": body}` (see
/// [`frame_to_event_data`](crate::lang_model::r#impl::framing::eventstream::frame_to_event_data));
/// each maps onto the same delta fragments the Anthropic stream produces:
/// - `messageStart`: role
/// - `contentBlockStart`: begins a `toolUse` call (id + name)
/// - `contentBlockDelta`: text / reasoning / signature / tool-input fragment
/// - `messageStop`: `stopReason`
/// - `metadata`: usage
/// - `contentBlockStop` and unknown events: no delta
impl Unmarshal<MessageDeltaOutput> for BedrockUnmarshal {
    fn unmarshal_event(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let val: Value = serde_json::from_str(data)?;
        let Some((ty, ev)) = val.as_object().and_then(|o| o.iter().next()) else {
            bail!("Converse stream event must be a single-key object");
        };

        let mut delta = MessageDelta::new();
        let mut finish_reason = None;
        let mut usage = None;

        match ty.as_str() {
            "messageStart" => {
                if let Some(role) = ev.pointer("/role").and_then(|v| v.as_str()) {
                    delta = delta.with_role(Self::parse_role(role));
                }
            }
            "contentBlockStart" => {
                if let Some(tool) = ev.pointer("/start/toolUse") {
                    let id = tool
                        .pointer("/toolUseId")
                        .and_then(|v| v.as_str())
                        .map(str::to_owned);
                    let name = tool.pointer("/name").and_then(|v| v.as_str()).unwrap_or("");
                    delta = delta.with_tool_calls([Self::tool_use_delta(id, name, String::new())]);
                }
            }
            "contentBlockDelta" => {
                if let Some(text) = ev.pointer("/delta/text").and_then(|v| v.as_str()) {
                    delta = delta.with_contents([PartDelta::Text {
                        text: text.to_owned(),
                    }]);
                }
                if let Some(t) = ev
                    .pointer("/delta/reasoningContent/text")
                    .and_then(|v| v.as_str())
                {
                    delta.thinking = Some(t.to_owned());
                }
                if let Some(s) = ev
                    .pointer("/delta/reasoningContent/signature")
                    .and_then(|v| v.as_str())
                {
                    delta.signature = Some(s.to_owned());
                }
                if let Some(args) = ev.pointer("/delta/toolUse/input").and_then(|v| v.as_str()) {
                    delta =
                        delta.with_tool_calls([Self::tool_use_delta(None, "", args.to_owned())]);
                }
            }
            "messageStop" => {
                if let Some(reason) = ev.pointer("/stopReason").and_then(|v| v.as_str()) {
                    finish_reason = Some(Self::parse_finish_reason(reason));
                }
            }
            "metadata" => {
                if let Some(u) = ev.pointer("/usage") {
                    usage = Self::parse_usage(u);
                }
            }
            "contentBlockStop" => return Ok(None),
            other => {
                log::debug!("ignoring unknown Converse stream event type: {other}");
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
        let message = val
            .pointer("/output/message")
            .ok_or_else(|| anyhow::anyhow!("Converse response has no output.message"))?;

        let role = message
            .pointer("/role")
            .and_then(|v| v.as_str())
            .map(Self::parse_role)
            .unwrap_or(Role::Assistant);
        let mut delta = MessageDelta::new().with_role(role);

        let mut text_parts: Vec<PartDelta> = Vec::new();
        let mut tool_call_parts: Vec<PartDelta> = Vec::new();
        for block in message
            .pointer("/content")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            if let Some(text) = block.pointer("/text").and_then(|v| v.as_str()) {
                text_parts.push(PartDelta::Text { text: text.into() });
            } else if let Some(tool) = block.pointer("/toolUse") {
                let id = tool
                    .pointer("/toolUseId")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned);
                let name = tool.pointer("/name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = tool
                    .pointer("/input")
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();
                tool_call_parts.push(Self::tool_use_delta(id, name, arguments));
            } else if let Some(reasoning) = block.pointer("/reasoningContent/reasoningText") {
                if let Some(t) = reasoning.pointer("/text").and_then(|v| v.as_str()) {
                    delta.thinking = Some(t.to_owned());
                }
                if let Some(s) = reasoning.pointer("/signature").and_then(|v| v.as_str()) {
                    delta.signature = Some(s.to_owned());
                }
            }
        }
        if !text_parts.is_empty() {
            delta = delta.with_contents(text_parts);
        }
        if !tool_call_parts.is_empty() {
            delta = delta.with_tool_calls(tool_call_parts);
        }

        let finish_reason = val
            .pointer("/stopReason")
            .and_then(|v| v.as_str())
            .map(Self::parse_finish_reason);
        let usage = val.pointer("/usage").and_then(Self::parse_usage);

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
    use super::*;
    use crate::{
        datatype::Bytes,
        lang_model::{LangModel, LangModelOptions, ResponseFormat},
        message::Delta as _,
        tool::ToolDescBuilder,
    };

    fn request<'a>(
        provider: &'a LangModelProviderElem,
        messages: &'a [Message],
        tools: &'a [ToolDesc],
        options: &'a LangModelOptions,
        stream: bool,
    ) -> LangModelRequest<'a> {
        LangModelRequest {
            provider,
            model: "global.anthropic.claude-sonnet-5",
            messages,
            tools,
            options,
            stream,
        }
    }

    fn marshal(req: &LangModelRequest<'_>) -> serde_json::Value {
        BedrockMarshal.marshal(req).into()
    }

    #[test]
    fn envelope_targets_converse_with_bearer_token() {
        let provider = LangModelProvider::bedrock("ap-northeast-2", "KEY123".to_string());
        let messages = vec![Message::new(Role::User).with_contents([Part::text("hi")])];
        let options = LangModelOptions::default();

        let v = marshal(&request(&provider, &messages, &[], &options, false));
        assert_eq!(
            v["url"],
            "https://bedrock-runtime.ap-northeast-2.amazonaws.com/model/global.anthropic.claude-sonnet-5/converse"
        );
        assert_eq!(v["header"]["Authorization"], "Bearer KEY123");
        assert!(v["header"].get("accept").is_none());
        assert!(
            v["body"].get("inferenceConfig").is_none(),
            "no implicit maxTokens"
        );

        let v = marshal(&request(&provider, &messages, &[], &options, true));
        assert!(v["url"].as_str().unwrap().ends_with("/converse-stream"));
        assert_eq!(v["header"]["accept"], "application/vnd.amazon.eventstream");
        assert!(
            v["body"].get("stream").is_none(),
            "streaming is chosen by path"
        );
    }

    #[test]
    fn model_url_escapes_slashes_in_arns() {
        let url = model_url(
            &runtime_url("us-east-1"),
            "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc",
            "converse",
        );
        assert!(
            url.ends_with("application-inference-profile%2Fabc/converse"),
            "{url}"
        );
    }

    #[test]
    fn system_tools_and_options_are_mapped() {
        let provider = LangModelProvider::bedrock("us-east-1", "k".to_string());
        let messages = vec![
            Message::new(Role::System).with_contents([Part::text("Be terse.")]),
            Message::new(Role::User).with_contents([Part::text("hi")]),
        ];
        let tools = vec![
            ToolDescBuilder::new("get_weather")
                .description("Weather lookup")
                .parameters(
                    to_value!({"type": "object", "properties": {"city": {"type": "string"}}}),
                )
                .build(),
        ];
        let options = LangModelOptions {
            max_tokens: Some(100),
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: Some(5),
            response_format: None,
        };

        let v = marshal(&request(&provider, &messages, &tools, &options, false));
        assert_eq!(
            v["body"]["system"],
            serde_json::json!([{"text": "Be terse."}])
        );
        assert_eq!(v["body"]["messages"].as_array().unwrap().len(), 1);
        assert_eq!(
            v["body"]["toolConfig"]["toolChoice"],
            serde_json::json!({"auto": {}})
        );
        let spec = &v["body"]["toolConfig"]["tools"][0]["toolSpec"];
        assert_eq!(spec["name"], "get_weather");
        assert_eq!(spec["description"], "Weather lookup");
        assert_eq!(spec["inputSchema"]["json"]["type"], "object");
        assert_eq!(
            v["body"]["inferenceConfig"],
            serde_json::json!({"maxTokens": 100, "temperature": 0.2, "topP": 0.9})
        );
        assert_eq!(v["body"]["additionalModelRequestFields"]["top_k"], 5);
    }

    #[test]
    fn tool_results_fold_into_one_user_turn() {
        let messages = vec![
            Message::new(Role::User).with_contents([Part::text("weather in two cities")]),
            Message::new(Role::Assistant).with_tool_calls([
                Part::function("t1", "get_weather", to_value!({"city": "Seoul"})),
                Part::function("t2", "get_weather", to_value!({"city": "Busan"})),
            ]),
            Message::new(Role::Tool)
                .with_id("t1")
                .with_contents([Part::value(to_value!({"temp": 20}))]),
            Message::new(Role::Tool)
                .with_id("t2")
                .with_contents([Part::text("sunny")]),
        ];
        let v: serde_json::Value = marshal_messages(&messages).into();
        let msgs = v.as_array().unwrap();
        assert_eq!(msgs.len(), 3, "two tool results merge into one user turn");

        let assistant = &msgs[1]["content"];
        assert_eq!(assistant[0]["toolUse"]["toolUseId"], "t1");
        assert_eq!(assistant[0]["toolUse"]["input"]["city"], "Seoul");
        assert_eq!(assistant[1]["toolUse"]["name"], "get_weather");

        let results = &msgs[2];
        assert_eq!(results["role"], "user");
        assert_eq!(results["content"][0]["toolResult"]["toolUseId"], "t1");
        assert_eq!(
            results["content"][0]["toolResult"]["content"][0]["json"]["temp"],
            20
        );
        assert_eq!(results["content"][1]["toolResult"]["toolUseId"], "t2");
        assert_eq!(
            results["content"][1]["toolResult"]["content"][0]["text"],
            "sunny"
        );
    }

    #[test]
    fn images_and_empty_text_blocks() {
        let msg = Message::new(Role::User).with_contents([
            Part::text(""),
            Part::image_embedded("image/jpg", Bytes::from(vec![1, 2, 3])).unwrap(),
        ]);
        let v: serde_json::Value = marshal_message(&msg, true).into();
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), 1, "empty text is dropped");
        assert_eq!(content[0]["image"]["format"], "jpeg");
        assert_eq!(content[0]["image"]["source"]["bytes"], "AQID");
    }

    #[test]
    fn thinking_replays_only_after_last_user_turn() {
        let mut early = Message::new(Role::Assistant).with_contents([Part::text("a")]);
        early.thinking = Some("old".into());
        let mut late = Message::new(Role::Assistant).with_contents([Part::text("b")]);
        late.thinking = Some("new".into());
        late.signature = Some("sig".into());
        let messages = vec![
            Message::new(Role::User).with_contents([Part::text("q1")]),
            early,
            Message::new(Role::User).with_contents([Part::text("q2")]),
            late,
        ];
        let v: serde_json::Value = marshal_messages(&messages).into();
        assert!(v[1]["content"][0].get("reasoningContent").is_none());
        assert_eq!(
            v[3]["content"][0]["reasoningContent"]["reasoningText"],
            serde_json::json!({"text": "new", "signature": "sig"})
        );
    }

    #[test]
    fn validate_rejects_bedrock_only_gaps() {
        let converse = LangModelProvider::bedrock("us-east-1", "k".to_string());
        let messages = vec![Message::new(Role::User).with_contents([Part::text("hi")])];

        let with_format = LangModelOptions {
            response_format: Some(
                ResponseFormat::json_schema(to_value!({"type": "object"})).unwrap(),
            ),
            ..Default::default()
        };
        let err = validate_request(&request(&converse, &messages, &[], &with_format, false))
            .unwrap_err()
            .to_string();
        assert!(err.contains("response_format"), "{err}");

        let url_image = vec![
            Message::new(Role::User)
                .with_contents([Part::image_url("https://example.com/a.png".into()).unwrap()]),
        ];
        let options = LangModelOptions::default();
        let err = validate_request(&request(&converse, &url_image, &[], &options, false))
            .unwrap_err()
            .to_string();
        assert!(err.contains("image URL"), "{err}");

        assert!(validate_request(&request(&converse, &messages, &[], &options, false)).is_ok());
    }

    #[test]
    fn unmarshal_response_with_tool_use_and_usage() {
        let resp = to_value!({
            "output": {"message": {"role": "assistant", "content": [
                {"text": "Let me check."},
                {"toolUse": {"toolUseId": "t1", "name": "get_weather", "input": {"city": "Seoul"}}}
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 12, "outputTokens": 7, "totalTokens": 19, "cacheReadInputTokens": 3},
            "metrics": {"latencyMs": 100}
        });
        let out = BedrockUnmarshal.unmarshal(resp).unwrap().finish().unwrap();
        assert_eq!(out.finish_reason, FinishReason::ToolCall {});
        assert_eq!(out.message.contents[0].as_text(), Some("Let me check."));
        let (id, name, args) = out.message.tool_calls.as_ref().unwrap()[0]
            .as_function()
            .unwrap();
        assert_eq!((id, name), ("t1", "get_weather"));
        assert_eq!(
            args.pointer("/city").and_then(|v| v.as_str()),
            Some("Seoul")
        );
        let usage = out.usage.unwrap();
        assert_eq!((usage.input_tokens, usage.output_tokens), (12, 7));
        assert_eq!(usage.cache_read_input_tokens, Some(3));
        assert_eq!(usage.cache_creation_input_tokens, None);
    }

    #[test]
    fn unmarshal_stream_accumulates_tool_call() {
        let events = [
            r#"{"messageStart":{"role":"assistant"}}"#,
            r#"{"contentBlockStart":{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"t1","name":"get_weather"}}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"city\":"}}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"toolUse":{"input":"\"Seoul\"}"}}}}"#,
            r#"{"contentBlockStop":{"contentBlockIndex":0}}"#,
            r#"{"messageStop":{"stopReason":"tool_use"}}"#,
            r#"{"metadata":{"usage":{"inputTokens":5,"outputTokens":9,"totalTokens":14},"metrics":{"latencyMs":1}}}"#,
        ];
        let mut u = BedrockUnmarshal;
        let mut acc = MessageDeltaOutput::new();
        for e in events {
            if let Some(d) = u.unmarshal_event(e).unwrap() {
                acc = acc.accumulate(d).unwrap();
            }
        }
        let out = acc.finish().unwrap();
        assert_eq!(out.finish_reason, FinishReason::ToolCall {});
        let (id, name, args) = out.message.tool_calls.as_ref().unwrap()[0]
            .as_function()
            .unwrap();
        assert_eq!((id, name), ("t1", "get_weather"));
        assert_eq!(
            args.pointer("/city").and_then(|v| v.as_str()),
            Some("Seoul")
        );
        assert_eq!(out.usage.unwrap().output_tokens, 9);
    }

    #[test]
    fn unmarshal_stream_text_and_reasoning() {
        let events = [
            r#"{"messageStart":{"role":"assistant"}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"hmm"}}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"reasoningContent":{"signature":"sig"}}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":1,"delta":{"text":"Hel"}}}"#,
            r#"{"contentBlockDelta":{"contentBlockIndex":1,"delta":{"text":"lo"}}}"#,
            r#"{"messageStop":{"stopReason":"end_turn"}}"#,
        ];
        let mut u = BedrockUnmarshal;
        let mut acc = MessageDeltaOutput::new();
        for e in events {
            if let Some(d) = u.unmarshal_event(e).unwrap() {
                acc = acc.accumulate(d).unwrap();
            }
        }
        let out = acc.finish().unwrap();
        assert_eq!(out.finish_reason, FinishReason::Stop {});
        assert_eq!(out.message.contents[0].as_text(), Some("Hello"));
        assert_eq!(out.message.thinking.as_deref(), Some("hmm"));
        assert_eq!(out.message.signature.as_deref(), Some("sig"));
    }

    #[test]
    fn constructors_register_the_runtime_base() {
        let elem = LangModelProvider::bedrock("us-east-1", "k".to_string());
        let json = serde_json::to_value(&elem).unwrap();
        assert_eq!(json["type"], "api");
        assert_eq!(json["schema"], "bedrock");
        assert_eq!(
            json["url"],
            "https://bedrock-runtime.us-east-1.amazonaws.com/"
        );
    }

    /// Live Converse round trip through the env-seeded default provider
    /// (`AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION`).
    #[test_with::env(AWS_BEARER_TOKEN_BEDROCK)]
    #[tokio::test]
    async fn test_converse_run_text() {
        dotenvy::dotenv().ok();
        let model = LangModel::try_new("bedrock/global.anthropic.claude-sonnet-5".into()).unwrap();
        let messages = vec![
            Message::new(Role::User).with_contents([Part::text("Reply with a short greeting.")]),
        ];

        let resp = model
            .run(&messages, &[], &LangModelOptions::default())
            .await
            .unwrap();

        assert_eq!(resp.finish_reason, FinishReason::Stop {});
        assert!(
            resp.message
                .contents
                .iter()
                .any(|p| p.as_text().is_some_and(|t| !t.is_empty())),
            "expected non-empty text"
        );
        let usage = resp.usage.expect("expected usage");
        assert!(usage.input_tokens > 0 && usage.output_tokens > 0);
    }

    /// Live Converse tool call.
    #[test_with::env(AWS_BEARER_TOKEN_BEDROCK)]
    #[tokio::test]
    async fn test_converse_run_tool_call() {
        dotenvy::dotenv().ok();
        let model = LangModel::try_new("bedrock/global.anthropic.claude-sonnet-5".into()).unwrap();
        let messages = vec![Message::new(Role::User).with_contents([Part::text(
            "What is the weather in Seoul? Use the get_weather tool.",
        )])];
        let tools = vec![
            ToolDescBuilder::new("get_weather")
                .description("Get the current weather for a city.")
                .parameters(to_value!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }))
                .build(),
        ];

        let resp = model
            .run(&messages, &tools, &LangModelOptions::default())
            .await
            .unwrap();

        assert_eq!(resp.finish_reason, FinishReason::ToolCall {});
        let calls = resp.message.tool_calls.expect("expected tool calls");
        let (_, name, args) = calls[0].as_function().expect("expected function part");
        assert_eq!(name, "get_weather");
        let args: serde_json::Value = args.clone().into();
        assert!(
            args["city"]
                .as_str()
                .is_some_and(|c| c.to_lowercase().contains("seoul"))
        );
    }

    /// Live `ConverseStream`: the event-stream decoder plus the per-event
    /// unmarshal must accumulate into a complete message.
    #[test_with::env(AWS_BEARER_TOKEN_BEDROCK)]
    #[tokio::test]
    async fn test_converse_run_stream_text() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let model = LangModel::try_new("bedrock/global.anthropic.claude-sonnet-5".into()).unwrap();
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
        let out = acc.finish().unwrap();
        assert_eq!(out.finish_reason, FinishReason::Stop {});
        assert!(
            out.message
                .contents
                .iter()
                .any(|p| p.as_text().is_some_and(|t| !t.is_empty()))
        );
        assert!(out.usage.expect("usage from metadata event").output_tokens > 0);
    }
}
