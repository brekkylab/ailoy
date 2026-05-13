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
    pub fn grok(api_key: String) -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::ChatCompletion,
            url: Url::parse("https://api.x.ai/v1/chat/completions").unwrap(),
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
        if !item.contents.is_empty() {
            let contents: Vec<Value> = item
                .contents
                .iter()
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
            rv.as_object_mut()
                .unwrap()
                .insert("content".into(), contents.into());
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
        if let Some(temperature) = req.temperature {
            body.as_object_mut()
                .unwrap()
                .insert("temperature".to_owned(), temperature.into());
        }
        if let Some(top_p) = req.top_p {
            body.as_object_mut()
                .unwrap()
                .insert("top_p".to_owned(), top_p.into());
        }
        // top_k is not part of the OpenAI ChatCompletion spec; intentionally ignored.
        body.as_object_mut()
            .unwrap()
            .retain(|_key, value| !value.is_null());

        to_value!({
            "url": url,
            "header": header,
            "body": body,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct ChatCompletionUnmarshal;

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

        let mut delta = MessageDelta::new().with_role(role);
        if !contents.is_empty() {
            delta = delta.with_contents(contents);
        }
        if !tool_calls.is_empty() {
            delta = delta.with_tool_calls(tool_calls);
        }

        // Parse usage (Chat Completion: usage.prompt_tokens / completion_tokens)
        let usage = val
            .as_object()
            .and_then(|r| r.get("usage"))
            .and_then(|u| u.as_object())
            .map(|u| TokenUsage {
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
        lang_model::{LangModel, LangModelAPISchema, LangModelOptions, LangModelProviderElem},
        message::{FinishReason, Message, Part, Role},
        tool::ToolDesc,
    };

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

    /// Verifies that max_tokens is respected by the ChatCompletion API (finish_reason: length).
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
