use async_stream::try_stream;
use futures::stream::BoxStream;

use crate::value::{Message, MessageOutput, ToolDescription};

pub fn run(
    model_name: &str,
    api_key: &str,
    msgs: Vec<Message>,
    tools: Vec<ToolDescription>,
) -> BoxStream<'static, Result<MessageOutput, String>> {
    use futures::StreamExt;
    use std::str::FromStr;

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

    Box::pin(try_stream! {
        let mut buf: Vec<u8> = Vec::with_capacity(8192);
        let resp = req.await.map_err(|e| e.to_string())?;
        if resp.status().is_success() {
            let mut strm = resp.bytes_stream();
            while let Some(chunk_res) = strm.next().await {
                let chunk = chunk_res.map_err(|e| e.to_string())?;
                buf.extend_from_slice(&chunk);

                while let Some(event_bytes) = drain_next_event(&mut buf) {
                    if let Some(payload) = parse_event_data(&event_bytes) {
                        if payload == "[DONE]" { return; }
                        let evt: serde_json::Value = serde_json::from_str(&payload).map_err(|e| format!("JSON deserialization failed: {}", e))?;
                        let choice = evt
                            .pointer("/choices/0")
                            .cloned()
                            .ok_or_else(|| "missing JSON pointer: /choices/0".to_string())?;
                        let out: MessageOutput = serde_json::from_value(choice)
                            .map_err(|e| format!("MessageOutput deserialization failed: {}", e.to_string()))?;
                        yield out;
                    }
                }
            }
        } else {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            Err(format!("OpenAI request failed: {} - {}", status, text))?;
        };
    })
}

#[cfg(test)]
mod tests {
    const OPENAI_API_KEY: &str = "";

    #[tokio::test]
    async fn infer_simple_chat() {
        use futures::StreamExt;

        use super::*;
        use crate::value::{MessageAggregator, Part, Role};

        let msgs = vec![
            Message::with_content(Role::System, Part::new_text("You are an assistant.")),
            Message::with_content(Role::User, Part::new_text("Hi what's your name?")),
        ];
        let mut agg = MessageAggregator::new();
        let mut strm = run("gpt-4.1", OPENAI_API_KEY, msgs, Vec::new());
        while let Some(delta_opt) = strm.next().await {
            let delta = delta_opt.unwrap();
            println!("{:?}", delta);
            if let Some(msg) = agg.update(delta) {
                println!("{:?}", msg);
            }
        }
    }
}
