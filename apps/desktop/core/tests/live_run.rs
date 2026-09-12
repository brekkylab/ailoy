//! A run that calls the `shell` tool through a real `cortex-local-console`.
//!
//! Needs a built `cortex-local-console` (AILOY_CORTEX_BIN_DIR or ../cortex/target/debug).
//! Run: `AILOY_CORTEX_BIN_DIR=../cortex/target/debug cargo test -p ailoy-desktop-core --test live_run -- --ignored`
//!
//! The model is fake and the console is real: the point is the seam between them — a tool
//! call the engine routes into a console standing in the workspace, whose result comes back
//! through the assembler and lands in SQLite as a `Role::Tool` message.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ailoy::{
    agent::{AgentProvider, get_agent_providers_mut},
    lang_model::{LangModelProvider, get_lm_providers_mut},
    message::{Part, Role},
    tool::{ToolProvider, get_tool_providers_mut},
};
use ailoy_desktop_core::{Engine, EngineConfig, RunEvent};
use axum::{Router, body::Body, response::Response, routing::post};

/// The first turn: one `shell` tool call, no text. `cmd` is the parameter name in ailoy's
/// `shell` tool descriptor, and the console's cwd is the workspace root, so a bare relative
/// path is what reads the file written below.
const TOOL_CALL: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"cmd\\\":\\\"cat hello.txt\\\"}\"}}]}}]}\n\n\
                         data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n\
                         data: [DONE]\n\n";
/// The second turn, after the tool result comes back: the answer that ends the run.
const ANSWER: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"It says hi.\"},\"finish_reason\":\"stop\"}]}\n\n\
                      data: [DONE]\n\n";

#[tokio::test]
#[ignore]
async fn a_run_reads_a_workspace_file_through_the_shell_tool() {
    // One scripted stream per model call: the tool call first, then the answer. Every
    // request after the second replays the answer, so a retry cannot hang the run.
    let calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/",
        post(move || {
            let calls = calls.clone();
            async move {
                let body = if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    TOOL_CALL
                } else {
                    ANSWER
                };
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(body))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // The engine builds agents against the `"default"` bundle. Spelled out rather than
    // `or_default`: `LangModelProvider::default()` reads the ambient API keys out of the
    // environment, and this test wants only `fake/*`. `ToolProvider::new()` (not `empty()`)
    // carries the builtins — `shell` among them.
    {
        let mut lmps = get_lm_providers_mut();
        if !lmps.contains_key("default") {
            lmps.insert("default".into(), LangModelProvider::new());
        }
        lmps.get_mut("default")
            .expect("default lang-model registry")
            .insert(
                "fake/*".into(),
                LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap(),
            );
    }
    {
        let mut tps = get_tool_providers_mut();
        if !tps.contains_key("default") {
            tps.insert("default".into(), ToolProvider::new());
        }
    }
    get_agent_providers_mut()
        .entry("default".into())
        .or_insert_with(|| AgentProvider::new("default", "default"));

    let dir = tempfile::tempdir().unwrap();
    let mut cfg = EngineConfig::new(dir.path());
    cfg.mount_workspace = false; // the console stands in files/ directly
    cfg.catalog_refresh = false;
    let engine = Engine::start(cfg).await.unwrap();
    engine
        .fs_write("/hello.txt", "hi from the workspace")
        .await
        .unwrap();

    let s = engine.session_create(Some("fake/m".into())).await.unwrap();
    let mut h = engine
        .run_start(&s.id, vec![Part::text("what does hello.txt say?")])
        .await
        .unwrap();
    let mut tool_started = false;
    loop {
        match h.events.recv().await.unwrap() {
            RunEvent::ToolCallStarted { name, .. } => {
                assert_eq!(name, "shell");
                tool_started = true;
            }
            RunEvent::Done => break,
            RunEvent::Cancelled => panic!("cancelled unexpectedly"),
            RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
            _ => {}
        }
    }
    assert!(tool_started, "the run never announced the shell call");

    let msgs = engine.message_list(&s.id).await.unwrap();
    let tool = msgs
        .iter()
        .find(|m| m.message.role == Role::Tool)
        .expect("a tool result was persisted");
    let v = tool.message.contents[0].as_value().expect("a value part");
    let stdout = v
        .pointer("/stdout")
        .and_then(|s| s.as_str())
        .unwrap_or_default();
    assert!(
        stdout.contains("hi from the workspace"),
        "the console did not read the workspace file: {v:?}"
    );
    assert_eq!(
        msgs.last().unwrap().message.contents[0].as_text(),
        Some("It says hi.")
    );
    engine.shutdown().await;
}
