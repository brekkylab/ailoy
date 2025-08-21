use std::sync::Arc;

use futures::{Stream, StreamExt as _};
use reqwest::{Method, Url, header::HeaderMap};

use crate::{
    model::{LanguageModel, api::openai},
    value::{
        Message, MessageOutput, MessageStyle, OPENAI_FMT, StyledMessage, StyledMessageOutput,
        ToolDesc,
    },
};

#[cfg(not(target_family = "wasm"))]
type HandleResponseFn = dyn Fn(String) -> Vec<MessageOutput> + Send + Sync + 'static;
#[cfg(target_family = "wasm")]
type HandleResponseFn = dyn Fn(String) -> Vec<MessageOutput> + 'static;

#[derive(Clone)]
pub struct SSEResponseLanguageModel {
    model: String,
    style: MessageStyle,
    build_request:
        Arc<dyn Fn(Vec<StyledMessage>, Vec<ToolDesc>) -> (Url, Method, HeaderMap, String)>,
    handle_response: Arc<HandleResponseFn>,
}

impl SSEResponseLanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> SSEResponseLanguageModel {
        let model = model.into();
        let api_key = api_key.into();
        let (style, build_request, handle_response) =
            if model.starts_with("gpt") || model.starts_with("o") {
                let style = OPENAI_FMT.clone();
                let model = model.clone();
                let api_key = api_key.clone();
                (
                    style,
                    Arc::new(move |msgs: Vec<StyledMessage>, tools: Vec<ToolDesc>| {
                        openai::build_request(&model, &api_key, msgs, tools)
                    }),
                    Arc::new(|_| Vec::new()),
                )
            } else if model.starts_with("claude") {
                todo!()
            } else if model.starts_with("gemini") {
                todo!()
            } else if model.starts_with("grok") {
                todo!()
            } else {
                panic!()
            };

        SSEResponseLanguageModel {
            model,
            style,
            build_request,
            handle_response,
        }
    }

    fn run_inner(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> impl Stream<Item = Result<MessageOutput, String>> + 'static {
        let msgs = msgs
            .iter()
            .map(|v| StyledMessage {
                data: v.clone(),
                style: self.style.clone(),
            })
            .collect::<Vec<_>>();
        let (url, method, headers, body) = (self.build_request)(msgs, tools);
        let handle_response = self.handle_response.clone();

        async_stream::try_stream! {
            let req = reqwest::Client::new()
                .request(method, url)
                .headers(headers)
                .body(body)
                .send();
            let mut buf: Vec<u8> = Vec::with_capacity(8192);
            let resp = req.await.map_err(|e| e.to_string())?;
            if resp.status().is_success() {
                let mut strm = resp.bytes_stream();
                while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    while let Some(event_bytes) = drain_next_event(&mut buf) {
                        if let Some(payload) = parse_event_data(&event_bytes) {
                            let v = (handle_response)(payload);
                        }
                    }
                    // let outs = (parse_response)(&mut buf)?;
                    // for v in outs {
                    //     yield v;
                    // }
                    // yield MessageOutput::new();
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
            yield MessageOutput::new();
        }
    }
}

fn drain_next_event(buf: &mut Vec<u8>) -> Option<Vec<u8>> {
    let mut i = 0;
    while i + 1 < buf.len() {
        // LF LF
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            let ev = buf.drain(..i + 2).collect::<Vec<u8>>();
            return Some(ev);
        }
        // CRLF CRLF
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            let ev = buf.drain(..i + 4).collect::<Vec<u8>>();
            return Some(ev);
        }
        i += 1;
    }
    None
}

fn parse_event_data(event_bytes: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(event_bytes);
    let mut datas = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            datas.push(rest.trim_start().to_string());
        }
    }
    if datas.is_empty() {
        None
    } else {
        Some(datas.join("\n"))
    }
}

pub fn parse_response(buf: &mut Vec<u8>) -> Result<Vec<MessageOutput>, String> {
    let mut rv = Vec::new();
    while let Some(event_bytes) = drain_next_event(buf) {
        if let Some(payload) = parse_event_data(&event_bytes) {
            if payload == "[DONE]" {
                continue;
            }
            let evt: serde_json::Value = serde_json::from_str(&payload)
                .map_err(|e| format!("JSON deserialization failed: {}", e))?;
            let choice = evt
                .pointer("/choices/0")
                .cloned()
                .ok_or_else(|| "missing JSON pointer: /choices/0".to_string())?;
            println!("{:?}", choice);
            let out: StyledMessageOutput = serde_json::from_value(choice)
                .map_err(|e| format!("MessageOutput deserialization failed: {}", e.to_string()))?;
            rv.push(out.data);
        }
    }
    Ok(rv)
}

impl LanguageModel for SSEResponseLanguageModel {
    fn run_nonsend(
        self: Arc<Self>,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> futures::stream::LocalBoxStream<'static, Result<MessageOutput, String>> {
        self.run_inner(msg, tools).boxed_local()
    }

    #[cfg(not(target_family = "wasm"))]
    fn run(
        self: Arc<Self>,
        msg: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> futures::stream::BoxStream<'static, Result<MessageOutput, String>> {
        self.run_inner(msg, tools).boxed()
    }
}
