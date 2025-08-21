use crate::{
    utils::BoxFuture,
    value::{MessageOutput, StyledMessage, StyledMessageOutput, ToolDesc},
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
