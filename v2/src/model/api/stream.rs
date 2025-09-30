use std::sync::Arc;

use futures::StreamExt as _;

use crate::{
    model::{InferenceConfig, LangModelInference, api::RequestConfig},
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{FinishReason, Message, MessageOutput, Role, ToolDesc},
};

#[derive(Clone, Debug)]
pub(crate) struct ServerEvent {
    pub event: String,
    pub data: String,
}

fn drain_next_event(buf: &mut Vec<u8>) -> Option<ServerEvent> {
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
            return Some(ServerEvent {
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
pub(crate) struct StreamAPILangModel {
    name: String,
    make_request: Arc<
        dyn Fn(Vec<Message>, Vec<ToolDesc>, RequestConfig) -> reqwest::RequestBuilder
            + MaybeSend
            + MaybeSync,
    >,
    handle_event: Arc<dyn Fn(ServerEvent) -> MessageOutput + MaybeSend + MaybeSync>,
}

impl StreamAPILangModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        let model = model.into();
        let ret = if model.starts_with("gpt") || model.starts_with("o") {
            // OpenAI models
            StreamAPILangModel::with_url(model, api_key, "https://api.openai.com/v1/responses")
        } else if model.starts_with("claude") {
            // Anthropic models
            todo!()
        } else if model.starts_with("gemini") {
            // Gemini models
            StreamAPILangModel::with_url(
                model,
                api_key,
                "https://generativelanguage.googleapis.com/v1beta/models",
            )
        } else if model.starts_with("grok") {
            todo!()
        } else {
            panic!()
        };

        ret
    }

    pub fn with_url(
        model: impl Into<String>,
        api_key: impl Into<String>,
        url: impl Into<String>,
    ) -> Self {
        let model = model.into();
        let api_key = api_key.into();
        let url = url.into();

        let ret = if model.starts_with("gpt") || model.starts_with("o") {
            // OpenAI models
            StreamAPILangModel {
                name: model.into(),
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::openai::make_request(&url, &api_key, msgs, tools, req)
                    },
                ),
                handle_event: Arc::new(super::openai::handle_event),
            }
        } else if model.starts_with("claude") {
            // Anthropic models
            todo!()
        } else if model.starts_with("gemini") {
            // Gemini models
            StreamAPILangModel {
                name: model.into(),
                make_request: Arc::new(
                    move |msgs: Vec<Message>, tools: Vec<ToolDesc>, req: RequestConfig| {
                        super::gemini::make_request(&url, &api_key, msgs, tools, req)
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

impl LangModelInference for StreamAPILangModel {
    fn infer<'a>(
        self: &'a mut Self,
        mut msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
        // Initialize buffer
        let mut buf: Vec<u8> = Vec::with_capacity(8192);

        // Build RequestConfig
        let req = RequestConfig {
            model: Some(self.name.clone()),
            system_message: if let Some(msg) = msgs.get(0)
                && msg.role == Role::System
            {
                Some(
                    msgs.remove(0)
                        .contents
                        .get(0)
                        .unwrap()
                        .as_text()
                        .unwrap()
                        .to_owned(),
                )
            } else {
                None
            },
            stream: true,
            think_effort: config.think_effort,
            temperature: config.temperature,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
        };

        // Send request
        let resp = (self.make_request)(msgs, tools, req).send();

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
                            None | Some(FinishReason::Stop()) => {
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
