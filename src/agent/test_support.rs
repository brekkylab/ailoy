//! Offline scaffolding for agent-loop tests: a scripted ChatCompletion SSE server and a
//! provider bundle pointing at it. No network beyond loopback, no keys.

use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use axum::{Router, body::Body, response::Response, routing::post};
use futures::StreamExt as _;

use crate::{
    agent::{AgentProvider, get_agent_providers_mut},
    lang_model::{LangModelProvider, get_lm_providers_mut},
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolFunc, ToolProvider, get_tool_providers_mut},
};

pub(crate) async fn spawn_sse_server(
    bodies: Vec<String>,
    chunk_delay: Option<Duration>,
) -> (SocketAddr, Arc<AtomicU32>) {
    let calls = Arc::new(AtomicU32::new(0));
    let bodies = Arc::new(bodies);
    let counter = calls.clone();
    let app = Router::new().route(
        "/",
        post(move || {
            let bodies = bodies.clone();
            let counter = counter.clone();
            async move {
                let i = counter.fetch_add(1, Ordering::SeqCst) as usize;
                let body = bodies[i.min(bodies.len() - 1)].clone();
                let events: Vec<String> = body
                    .split("\n\n")
                    .filter(|e| !e.trim().is_empty())
                    .map(|e| format!("{e}\n\n"))
                    .collect();
                let stream = futures::stream::iter(events).then(move |e| async move {
                    if let Some(d) = chunk_delay {
                        tokio::time::sleep(d).await;
                    }
                    Ok::<_, std::io::Error>(axum::body::Bytes::from(e))
                });
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr, calls)
}

/// A server that answers every request with `status` and `body` — for the paths that
/// turn an HTTP failure into a [`crate::lang_model::ModelError`]. [`spawn_sse_server`]
/// only ever answers 200, so a failure needs its own.
pub(crate) async fn spawn_status_server(status: u16, body: &'static str) -> SocketAddr {
    let app = Router::new().route(
        "/",
        post(move || async move {
            Response::builder()
                .status(status)
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

pub(crate) fn sse_text(text_chunks: &[&str]) -> String {
    let mut s = String::new();
    s.push_str("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n");
    for c in text_chunks {
        let c = c.replace('"', "\\\"");
        s.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{c}\"}}}}]}}\n\n"
        ));
    }
    s.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    s.push_str("data: [DONE]\n\n");
    s
}

pub(crate) fn sse_tool_call(id: &str, name: &str, args_json: &str) -> String {
    let args = args_json.replace('"', "\\\"");
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"{id}\",\"type\":\"function\",\"function\":{{\"name\":\"{name}\",\"arguments\":\"{args}\"}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

/// Register `name` as a lang-model provider (pattern `fake/*` → the server), a tool
/// provider holding exactly `tools`, and an agent-provider bundle. Idempotent per name;
/// the registries are process-global, so use a distinct name per test.
pub(crate) fn register_fake_provider(
    name: &'static str,
    addr: SocketAddr,
    tools: Vec<(&str, ToolDesc, ToolFunc)>,
) -> &'static str {
    {
        let mut lmps = get_lm_providers_mut();
        let mut lmp = LangModelProvider::new();
        lmp.insert(
            "fake/*".into(),
            LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap(),
        );
        lmps.insert(name.to_string(), lmp);
    }
    {
        let mut tps = get_tool_providers_mut();
        let mut tp = ToolProvider::empty();
        for (tool_name, _desc, func) in &tools {
            tp.insert_func(*tool_name, func.clone());
        }
        tps.insert(name.to_string(), tp);
    }
    get_agent_providers_mut().insert(name.to_string(), AgentProvider::new(name, name));
    name
}

pub(crate) fn user(text: &str) -> Message {
    Message::new(Role::User).with_contents([Part::text(text)])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scripted server hands out bodies in order, sticks on the last one once the
    /// script runs out, and `register_fake_provider` lands a bundle in the registry.
    #[tokio::test]
    async fn scripted_server_serves_bodies_in_order() {
        let expected = sse_text(&["hi"]);
        let (addr, calls) = spawn_sse_server(vec![expected.clone()], None).await;

        let client = reqwest::Client::new();
        let url = format!("http://{addr}/");

        let first = client
            .post(&url)
            .body("{}")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(first, expected);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Only one body was scripted, so every later call keeps serving the last one.
        let second = client
            .post(&url)
            .body("{}")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert_eq!(second, expected);
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        register_fake_provider("test_support_smoke", addr, vec![]);
        assert!(
            crate::agent::get_agent_providers()
                .get("test_support_smoke")
                .is_some()
        );
    }
}
