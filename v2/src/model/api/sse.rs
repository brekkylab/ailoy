use std::sync::Arc;

use futures::StreamExt as _;

use crate::{
    model::LanguageModel,
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{FinishReason, LMConfig, Message, MessageOutput, ToolDesc},
};

#[derive(Clone, Debug)]
pub struct ServerSentEvent {
    pub event: String,
    pub data: String,
}

fn drain_next_event(buf: &mut Vec<u8>) -> Option<ServerSentEvent> {
    let mut i = 0;
    while i + 1 < buf.len() {
        if let Some(pos) = buf
            .windows(2)
            .position(|w| w == b"\n\n")
            .or_else(|| buf.windows(4).position(|w| w == b"\r\n\r\n"))
        {
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
    make_request: Arc<
        dyn Fn(Vec<Message>, Vec<ToolDesc>, LMConfig) -> reqwest::RequestBuilder
            + MaybeSend
            + MaybeSync,
    >,
    handle_event: Arc<dyn Fn(ServerSentEvent) -> MessageOutput + MaybeSend + MaybeSync>,
}

impl SSELanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model = model.into();
        let api_key = api_key.into();

        let ret = if model.starts_with("gpt") || model.starts_with("o") {
            // OpenAI models
            SSELanguageModel {
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, config: LMConfig| {
                        super::openai::make_request(
                            &api_key,
                            msgs,
                            tools,
                            config.with_model(model.as_str()),
                        )
                    },
                ),
                handle_event: Arc::new(super::openai::handle_event),
            }
        } else if model.starts_with("claude") {
            // Anthropic models
            todo!()
        } else if model.starts_with("gemini") {
            // Gemini models
            SSELanguageModel {
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, config: LMConfig| {
                        super::gemini::make_request(
                            &api_key,
                            msgs,
                            tools,
                            config.with_model(model.as_str()),
                        )
                    },
                ),
                handle_event: Arc::new(super::gemini::handle_event),
            }
        } else if model.starts_with("grok") {
            todo!()
        } else {
            panic!()
        };

        ret
    }
}

impl LanguageModel for SSELanguageModel {
    fn run<'a>(
        self: &'a mut Self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: LMConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        // Initialize buffer
        let mut buf: Vec<u8> = Vec::with_capacity(8192);

        // Send request
        let resp = (self.make_request)(msgs, tools, config).send();

        let strm = async_stream::try_stream! {
            // Await response
            let resp = resp.await.map_err(|e| e.to_string())?;

            if resp.status().is_success() {
                // println!("{:?}", resp.text().await);
                // On success - read stream
                let mut strm = resp.bytes_stream();
                'outer: while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    while let Some(evt) = drain_next_event(&mut buf) {
                        let message_output = (self.handle_event)(evt);
                        match message_output.finish_reason {
                            None | Some(FinishReason::Stop) => {
                                yield message_output;
                            }
                            Some(_) => {
                                break 'outer;
                            }
                        }
                    };
                }
            } else {
                // On error
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
        };
        Box::pin(strm)
    }
}
