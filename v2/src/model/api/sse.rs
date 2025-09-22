use std::sync::Arc;

use futures::StreamExt as _;

use crate::{
    model::LanguageModel,
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageDelta, ToolDesc},
};

#[derive(Clone, Debug)]
pub struct ServerSentEvent {
    pub event: String,
    pub data: String,
}

fn drain_next_event(buf: &mut Vec<u8>) -> Option<ServerSentEvent> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
            let raw = buf.drain(..pos + 2).collect::<Vec<u8>>();
            let text = String::from_utf8_lossy(&raw);
            let mut event: String = String::new();
            let mut data_lines: Vec<String> = Vec::new();
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("event:") {
                    event = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.trim().to_string());
                }
            }
            return Some(ServerSentEvent {
                event,
                data: if data_lines.is_empty() {
                    String::new()
                } else {
                    data_lines.join("\n")
                },
            });
        }
        i += 1;
    }
    None
}

#[derive(Clone)]
pub struct SSELanguageModel {
    make_request:
        Arc<dyn Fn(Vec<Message>, Vec<ToolDesc>) -> reqwest::RequestBuilder + MaybeSend + MaybeSync>,
    handle_event: Arc<dyn Fn(ServerSentEvent) -> Vec<MessageDelta> + MaybeSend + MaybeSync>,
}

impl SSELanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model = model.into();
        let api_key = api_key.into();
        let (make_request, handle_event) = if model.starts_with("gpt") || model.starts_with("o") {
            // OpenAI models
            let model = model.clone();
            (
                Arc::new(move |msgs: Vec<Message>, tools: Vec<ToolDesc>| {
                    super::openai::make_request(&model, &api_key, msgs, tools)
                }),
                Arc::new(super::openai::handle_event),
            )
        } else if model.starts_with("claude") {
            // Anthropic models
            todo!()
        } else if model.starts_with("gemini") {
            // Gemini models
            todo!()
        } else if model.starts_with("grok") {
            todo!()
        } else {
            panic!()
        };

        SSELanguageModel {
            make_request,
            handle_event,
        }
    }
}

impl LanguageModel for SSELanguageModel {
    fn run<'a>(
        self: &'a mut Self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'a, Result<MessageDelta, String>> {
        // Initialize buffer
        let mut buf: Vec<u8> = Vec::with_capacity(8192);

        // Send request
        let resp = (self.make_request)(msgs, tools).send();

        let strm = async_stream::try_stream! {
            // Await response
            let resp = resp.await.map_err(|e| e.to_string())?;

            if resp.status().is_success() {
                // On success - read stream
                let mut strm = resp.bytes_stream();
                while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    while let Some(evt) = drain_next_event(&mut buf) {
                        for delta in (self.handle_event)(evt) {
                            yield delta;
                        }
                    };
                }
            } else {
                // On error
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
            yield MessageDelta::default();
        };
        Box::pin(strm)
    }
}
