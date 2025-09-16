use std::{fmt::Debug, sync::Arc};

use futures::StreamExt;

use crate::{
    model::LanguageModel,
    utils::{BoxFuture, BoxStream},
    value::{
        Message, MessageOutput, MessageStyle, OPENAI_FMT, StyledMessage, StyledMessageOutput,
        ToolDesc,
    },
};

pub fn make_request(
    model_name: &str,
    api_key: &str,
    msgs: Vec<StyledMessage>,
    tools: Vec<ToolDesc>,
) -> BoxFuture<'static, Result<reqwest::Response, reqwest::Error>> {
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": msgs,
        "stream": true
    });
    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!("auto");
        body["tools"] = serde_json::to_value(tools).unwrap();
    }

    let req = reqwest::Client::new()
        .request(
            reqwest::Method::POST,
            "https://api.openai.com/v1/chat/completions",
        )
        .bearer_auth(api_key)
        .header("Content-Type", "application/json")
        .header(reqwest::header::ACCEPT, "text/event-stream")
        .body(body.to_string())
        .send();

    Box::pin(req)
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

pub fn handle_next_response(buf: &mut Vec<u8>) -> Result<Vec<MessageOutput>, String> {
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

#[derive(Clone)]
pub struct APILanguageModel {
    model: String,
    style: MessageStyle,
    make_request: Arc<
        dyn Fn(
                Vec<StyledMessage>,
                Vec<ToolDesc>,
            ) -> BoxFuture<'static, Result<reqwest::Response, reqwest::Error>>
            + Send
            + Sync,
    >,
    handle_response: Arc<dyn Fn(&mut Vec<u8>) -> Result<Vec<MessageOutput>, String> + Send + Sync>,
}

impl APILanguageModel {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> APILanguageModel {
        let model = model.into();
        let api_key = api_key.into();
        let (style, make_request, handle_response) =
            if model.starts_with("gpt") || model.starts_with("o") {
                let style = OPENAI_FMT.clone();
                let model = model.clone();
                let api_key = api_key.clone();
                (
                    style,
                    Arc::new(move |msgs: Vec<StyledMessage>, tools: Vec<ToolDesc>| {
                        make_request(&model, &api_key, msgs, tools)
                    }),
                    Arc::new(&handle_next_response),
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

        APILanguageModel {
            model,
            style,
            make_request,
            handle_response,
        }
    }
}

impl LanguageModel for APILanguageModel {
    fn run(
        self: Arc<Self>,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
    ) -> BoxStream<'static, Result<MessageOutput, String>> {
        let msgs = msgs
            .iter()
            .map(|v| StyledMessage {
                data: v.clone(),
                style: self.style.clone(),
            })
            .collect::<Vec<_>>();
        let req = (self.make_request)(msgs, tools);
        let strm = async_stream::try_stream! {
            let mut buf: Vec<u8> = Vec::with_capacity(8192);
            let resp = req.await.map_err(|e| e.to_string())?;
            if resp.status().is_success() {
                let mut strm = resp.bytes_stream();
                while let Some(chunk_res) = strm.next().await {
                    let chunk = chunk_res.map_err(|e| e.to_string())?;
                    buf.extend_from_slice(&chunk);
                    let outs = (self.handle_response)(&mut buf)?;
                    for v in outs {
                        yield v;
                    }
                }
            } else {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(format!("Request failed: {} - {}", status, text))?;
            }
        };
        Box::pin(strm)
    }
}

impl Debug for APILanguageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("APILanguageModel")
            .field("model", &self.model)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Part, Role};

        let model = Arc::new(APILanguageModel::new("gpt-4.1", OPENAI_API_KEY));

        let msgs = vec![
            Message::with_role(Role::System)
                .with_contents(vec![Part::Text("You are an assistant.".to_owned())]),
            Message::with_role(Role::User)
                .with_contents(vec![Part::Text("Hi what's your name?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
    }

    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn infer_tool_call() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Part, Role, ToolDesc, ToolDescArg};

        let model = Arc::new(APILanguageModel::new("gpt-4.1", OPENAI_API_KEY));
        let tools = vec![ToolDesc::new(
            "temperature",
            "Get current temperature",
            ToolDescArg::new_object().with_properties(
                [
                    (
                        "location",
                        ToolDescArg::new_string().with_desc("The city name"),
                    ),
                    (
                        "unit",
                        ToolDescArg::new_string()
                            .with_enum(["Celcius", "Fernheit"])
                            .with_desc("The unit of temperature"),
                    ),
                ],
                ["location", "unit"],
            ),
            Some(
                ToolDescArg::new_number().with_desc("Null if the given city name is unavailable."),
            ),
        )];
        let msgs = vec![
            Message::with_role(Role::User)
                .with_contents([Part::Text("How much hot currently in Dubai?".to_owned())]),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = model.run(msgs, tools);
        let mut assistant_msg: Option<Message> = None;
        while let Some(delta_opt) = strm.next().await {
            println!("{:?}", delta_opt);
            let delta = delta_opt.unwrap();
            if let Some(msg) = agg.update(delta) {
                assistant_msg = Some(msg);
            }
        }
        let assistant_msg = assistant_msg.unwrap();
        let tc = assistant_msg.tool_calls.get(0).unwrap();
        println!("Tool call: {:?}", tc);
    }
}
