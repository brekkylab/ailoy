//! The crate as a user takes it, on the platform this runs on -- as `node.mjs` and
//! `python.py` are for the bindings, and told what this platform can do the same way:
//!
//!   SMOKE_SERVER  1 if the console server is in $VIRTX_HOME/bin, else 0
//!   SMOKE_VM      1 if this machine can boot one (KVM or HVF), else 0
//!
//! A turn's model is served from this process, as `fake.mjs` serves it: no network, and no
//! API key.
use std::io::{BufRead as _, BufReader, Read as _, Write as _};

use ailoy::{
    agent::AgentBuilder,
    lang_model::{LangModelAPISchema, LangModelProviderElem, get_lm_providers_mut},
    message::{Message, MessageOutput, Part, Role},
};
use futures::StreamExt as _;
use serde_json::{Value, json};
use virtx::{console::ConsoleClient, image::ImageClient, image::Recipe};

fn want(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// The fake model's answer to one chat-completions request (see `fake.mjs`).
fn answer(body: &Value) -> Value {
    let text = |content: &Value| match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect(),
        _ => String::new(),
    };
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    let last = messages.last().cloned().unwrap_or_default();
    let message = if last["role"] == "tool" {
        json!({ "role": "assistant", "content": format!("the tool said {}", text(&last["content"])) })
    } else if body["tools"].as_array().is_some_and(|t| !t.is_empty()) {
        let user = messages.iter().find(|m| m["role"] == "user").map(|m| text(&m["content"])).unwrap_or_default();
        let (name, args) = user.split_once(' ').unwrap_or((&user, "{}"));
        json!({ "role": "assistant", "content": null, "tool_calls": [
            { "id": "call_1", "type": "function", "function": { "name": name, "arguments": args } }
        ] })
    } else {
        json!({ "role": "assistant", "content": "hello" })
    };
    let finish = if message["tool_calls"].is_null() { "stop" } else { "tool_calls" };
    json!({
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
        "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
    })
}

/// Serve the fake model on localhost, one request per connection, and return its URL.
fn fake_model() -> anyhow::Result<String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let url = format!("http://{}/v1/chat/completions", listener.local_addr()?);
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = (|| -> anyhow::Result<()> {
                let mut reader = BufReader::new(stream.try_clone()?);
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line)?;
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = line.split_once(':')
                        && k.eq_ignore_ascii_case("content-length")
                    {
                        length = v.trim().parse()?;
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body)?;
                let payload = answer(&serde_json::from_slice(&body)?).to_string();
                write!(
                    &stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
                    payload.len()
                )?;
                Ok(())
            })();
        }
    });
    Ok(url)
}

async fn run(agent: &mut ailoy::agent::Agent, query: &str) -> anyhow::Result<Vec<MessageOutput>> {
    let query = Message::new(Role::User).with_contents([Part::text(query)]);
    let mut stream = agent.run(query);
    let mut outputs = Vec::new();
    while let Some(output) = stream.next().await {
        outputs.push(output?);
    }
    Ok(outputs)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut failures = Vec::new();
    let mut check = |ok: bool, what: &str| {
        println!("{} {what}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures.push(what.to_string());
        }
    };

    let url = fake_model()?;
    get_lm_providers_mut()
        .get_mut("default")
        .ok_or_else(|| anyhow::anyhow!("no default model provider"))?
        .insert(
            "fake/*".into(),
            LangModelProviderElem::API { schema: LangModelAPISchema::ChatCompletion, url: url.parse()?, api_key: None },
        );

    let mut agent = AgentBuilder::new("fake/model").build().await?;
    let outputs = run(&mut agent, "hi").await?;
    let said: Vec<&str> = outputs.iter().flat_map(|o| o.message.contents.iter().filter_map(Part::as_text)).collect();
    check(said == ["hello"], "a turn against the model answers");

    if want("SMOKE_SERVER") {
        // The server runs on this machine, and answers -- no VM needed to ask its version.
        let mut images = ImageClient::try_new().await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let version = images.version().await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
        drop(images);
        check(!version.is_empty(), &format!("the console server answers (protocol {version})"));
    }

    if want("SMOKE_VM") {
        // An agent's shell tool, in a VM session that sees a host directory.
        let host = std::env::temp_dir().join(format!("ailoy-smoke-rust-{}", std::process::id()));
        std::fs::create_dir_all(&host)?;
        std::fs::write(host.join("from-host.txt"), "by path")?;
        let console = ConsoleClient::builder()
            .image(Recipe::new("alpine:latest"))
            .mount(host.clone(), "/host")
            .build()
            .await?;
        let mut agent = AgentBuilder::new("fake/model").shell_tool().console(console).build().await?;
        let cmd = "uname -m; cat /host/from-host.txt; echo written > /host/from-vm.txt";
        let outputs = run(&mut agent, &format!("shell {}", json!({ "cmd": cmd }))).await?;
        let said = outputs.get(1).map(|o| serde_json::to_string(&o.message.contents)).transpose()?.unwrap_or_default();
        println!("  vm: {said}");
        check(said.contains("by path"), "the agent's shell tool reads the session's mount");
        let wrote = std::fs::read_to_string(host.join("from-vm.txt")).unwrap_or_default();
        check(wrote.trim() == "written", "the host sees the VM's write");
        drop(agent);
        let _ = std::fs::remove_dir_all(&host);
    }

    if failures.is_empty() {
        println!("ALL PASS");
        Ok(())
    } else {
        anyhow::bail!("FAILED: {}", failures.join("; "))
    }
}
