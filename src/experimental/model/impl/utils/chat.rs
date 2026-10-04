//! The Chat Completions wire format that OpenRouter, DeepSeek, Kimi and GLM share. Only
//! the format lives here; each vendor's options, endpoint and credentials stay in its own
//! model. They differ in the field that carries thinking: `reasoning` on OpenRouter,
//! `reasoning_content` on the rest.

use serde_json::json;

use super::{EventParser, assistant_output, last_user_index, tokens, value_text};
use crate::{
    message::{
        FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, PartFunction, PartImage, Role, TokenUsage,
    },
    tool::ToolDesc,
};

/// `messages`, system ones included. Thinking goes in `reasoning_field` for assistant
/// turns after the last user message, where a tool-calling turn needs it back.
pub(crate) fn wire_messages(messages: &[Message], reasoning_field: &str) -> serde_json::Value {
    let last_user = last_user_index(messages);
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| wire_message(m, (i > last_user).then_some(reasoning_field)))
        .collect()
}

fn wire_message(msg: &Message, reasoning_field: Option<&str>) -> serde_json::Value {
    let mut wire = json!({"role": msg.role.to_string()});
    match msg.role {
        // Tool results are text here: these backends do not reliably take images in them,
        // so an image is named instead.
        Role::Tool => {
            wire["tool_call_id"] = msg.id.as_deref().unwrap_or_default().into();
            let text: Vec<_> = msg
                .contents
                .iter()
                .filter_map(|part| match part {
                    Part::Text { text } => Some(text.clone()),
                    Part::Value { value } => Some(value_text(value)),
                    Part::Image {
                        image: PartImage::Embedded { mime_type, .. },
                    } => Some(format!("[image: {mime_type}]")),
                    Part::Image {
                        image: PartImage::Url { url },
                    } => Some(format!("[image at {url}]")),
                    Part::Function { .. } => None,
                })
                .collect();
            wire["content"] = text.join("\n").into();
        }
        _ => {
            if let Some(content) = wire_content(&msg.contents) {
                wire["content"] = content;
            }
            if let Some(field) = reasoning_field
                && let Some(thinking) = msg.thinking.as_deref().filter(|t| !t.is_empty())
            {
                wire[field] = thinking.into();
            }
            let calls: Vec<_> = msg
                .tool_calls
                .iter()
                .flatten()
                .filter_map(|part| match part {
                    Part::Function { id, function } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": function.name,
                            "arguments": serde_json::to_string(&function.arguments)
                                .unwrap_or_default(),
                        },
                    })),
                    _ => None,
                })
                .collect();
            if !calls.is_empty() {
                wire["tool_calls"] = calls.into();
            }
        }
    }
    wire
}

/// All-text contents as one string, which every backend takes; otherwise an array of
/// parts. Empty text parts are dropped, as some backends (Kimi) reject them. `None` when
/// nothing is left.
fn wire_content(contents: &[Part]) -> Option<serde_json::Value> {
    let parts: Vec<_> = contents
        .iter()
        .filter(|p| !matches!(p, Part::Text { text } if text.is_empty()))
        .filter(|p| !p.is_function())
        .collect();
    if parts.is_empty() {
        return None;
    }
    if parts.iter().all(|p| p.is_text()) {
        let text: Vec<_> = parts.iter().filter_map(|p| p.as_text()).collect();
        return Some(text.concat().into());
    }
    Some(
        parts
            .into_iter()
            .map(|part| match part {
                Part::Text { text } => json!({"type": "text", "text": text}),
                Part::Value { value } => json!({"type": "text", "text": value_text(value)}),
                Part::Image { image } => {
                    let url = match image {
                        PartImage::Embedded { mime_type, data } => {
                            format!("data:{mime_type};base64,{}", data.base64())
                        }
                        PartImage::Url { url } => url.clone(),
                    };
                    json!({"type": "image_url", "image_url": {"url": url}})
                }
                Part::Function { .. } => unreachable!("function parts are filtered out"),
            })
            .collect(),
    )
}

pub(crate) fn wire_tool(tool: &ToolDesc) -> serde_json::Value {
    let mut function = json!({
        "name": tool.name,
        "parameters": serde_json::Value::from(tool.parameters.clone()),
    });
    if let Some(description) = &tool.description {
        function["description"] = description.as_str().into();
    }
    json!({"type": "function", "function": function})
}

/// The stream's `chat.completion.chunk`s. A tool call's first fragment carries its id,
/// which merges the fragments after it; a backend repeating the id on every fragment has
/// the repeats dropped.
pub(crate) struct ChatEvents {
    reasoning_field: &'static str,
    call_ids: Vec<String>,
}

impl ChatEvents {
    pub(crate) fn new(reasoning_field: &'static str) -> Self {
        Self {
            reasoning_field,
            call_ids: Vec::new(),
        }
    }
}

impl EventParser for ChatEvents {
    fn parse(&mut self, data: &str) -> anyhow::Result<Option<MessageDeltaOutput>> {
        if data == "[DONE]" {
            return Ok(None);
        }
        let chunk: serde_json::Value = serde_json::from_str(data)?;
        if chunk["error"].is_object() {
            anyhow::bail!("stream error: {}", chunk["error"]);
        }
        let mut out = MessageDeltaOutput::new();
        out.usage = parse_usage(&chunk["usage"]);
        let choice = &chunk["choices"][0];
        if choice.is_null() {
            // The usage-only last chunk.
            return Ok(out.usage.is_some().then_some(out));
        }

        // Some backends never send the role.
        out.delta = MessageDelta::new().with_role(Role::Assistant);
        let delta = &choice["delta"];
        if let Some(text) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.delta.contents = vec![PartDelta::Text {
                text: text.to_owned(),
            }];
        }
        if let Some(thinking) = delta[self.reasoning_field]
            .as_str()
            .filter(|t| !t.is_empty())
        {
            out.delta.thinking = Some(thinking.to_owned());
        }
        for call in delta["tool_calls"].as_array().into_iter().flatten() {
            let id = match call["id"].as_str().filter(|id| !id.is_empty()) {
                Some(id) if !self.call_ids.iter().any(|seen| seen == id) => {
                    self.call_ids.push(id.to_owned());
                    Some(id.to_owned())
                }
                _ => None,
            };
            out.delta.tool_calls.push(PartDelta::Function {
                id,
                function: PartDeltaFunction::WithStringArgs {
                    name: call["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                    arguments: call["function"]["arguments"]
                        .as_str()
                        .unwrap_or_default()
                        .to_owned(),
                },
            });
        }
        out.finish_reason = choice["finish_reason"]
            .as_str()
            .map(|r| parse_finish_reason(r, !self.call_ids.is_empty()));
        Ok(Some(out))
    }
}

/// A whole `chat.completion`, from its first choice.
pub(crate) fn parse_response(
    response: &serde_json::Value,
    reasoning_field: &str,
) -> anyhow::Result<MessageOutput> {
    let choice = &response["choices"][0];
    let message = &choice["message"];
    let contents = message["content"]
        .as_str()
        .filter(|t| !t.is_empty())
        .map(Part::text)
        .into_iter()
        .collect();
    let thinking = message[reasoning_field]
        .as_str()
        .filter(|t| !t.is_empty())
        .map(str::to_owned);
    let mut tool_calls = Vec::new();
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        let arguments = call["function"]["arguments"].as_str().unwrap_or("{}");
        let arguments = if arguments.trim().is_empty() {
            "{}"
        } else {
            arguments
        };
        tool_calls.push(Part::Function {
            id: call["id"].as_str().unwrap_or_default().to_owned(),
            function: PartFunction {
                name: call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                arguments: serde_json::from_str::<serde_json::Value>(arguments)?.into(),
            },
        });
    }
    let finish_reason = choice["finish_reason"]
        .as_str()
        .map(|r| parse_finish_reason(r, !tool_calls.is_empty()))
        .ok_or_else(|| anyhow::anyhow!("response has no finish_reason: {response}"))?;
    Ok(assistant_output(
        contents,
        thinking,
        None,
        tool_calls,
        finish_reason,
        parse_usage(&response["usage"]),
    ))
}

/// Some backends answer `stop` for a turn that called tools, so `called` makes it one.
fn parse_finish_reason(reason: &str, called: bool) -> FinishReason {
    match reason {
        "tool_calls" | "function_call" => FinishReason::ToolCall {},
        "stop" if called => FinishReason::ToolCall {},
        "stop" => FinishReason::Stop {},
        "length" => FinishReason::Length {},
        other => FinishReason::Refusal {
            reason: other.to_owned(),
        },
    }
}

/// Cache hits are `prompt_tokens_details.cached_tokens`, or DeepSeek's
/// `prompt_cache_hit_tokens`.
fn parse_usage(usage: &serde_json::Value) -> Option<TokenUsage> {
    usage.is_object().then(|| TokenUsage {
        input_tokens: tokens(&usage["prompt_tokens"]).unwrap_or(0),
        output_tokens: tokens(&usage["completion_tokens"]).unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: tokens(&usage["prompt_tokens_details"]["cached_tokens"])
            .or_else(|| tokens(&usage["prompt_cache_hit_tokens"])),
    })
}
