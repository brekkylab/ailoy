use std::str::FromStr;

use reqwest::{
    Method, Url,
    header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue},
};

use crate::value::{MessageOutput, StyledMessage, StyledMessageOutput, ToolDesc};

pub fn build_request(
    model_name: &str,
    api_key: &str,
    msgs: Vec<StyledMessage>,
    tools: Vec<ToolDesc>,
) -> (Url, Method, HeaderMap, String) {
    let url = Url::from_str("https://api.openai.com/v1/chat/completions").unwrap();
    let method = Method::POST;
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".try_into().unwrap());
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", api_key).try_into().unwrap(),
    );
    headers.insert(ACCEPT, "text/event-stream".try_into().unwrap());
    let mut body = serde_json::json!({
        "model": model_name,
        "messages": msgs,
        "stream": true
    });
    if !tools.is_empty() {
        body["tool_choice"] = serde_json::json!("auto");
        body["tools"] = serde_json::to_value(tools).unwrap();
    }
    let body = body.to_string();
    (url, method, headers, body)
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
