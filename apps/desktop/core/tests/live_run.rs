//! A run that calls the `shell` tool through a real `cortex-local-console`.
//!
//! Needs a built `cortex-local-console` (AILOY_CORTEX_BIN_DIR or ../cortex/target/debug);
//! the second test also needs FUSE-T installed.
//! Run: `AILOY_CORTEX_BIN_DIR=../cortex/target/debug cargo test -p ailoy-desktop-core --test live_run -- --ignored`
//!
//! The model is fake and the console is real: the point is the seam between them — a tool
//! call the engine routes into a console holding the workspace as its artifacts tree, whose
//! result comes back through the assembler and lands in SQLite as a `Role::Tool` message.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use ailoy::{
    agent::{AgentProvider, get_agent_providers_mut},
    lang_model::{LangModelProvider, get_lm_providers_mut},
    message::{Part, Role},
    tool::{ToolProvider, get_tool_providers_mut},
};
use ailoy_desktop_core::{Engine, EngineConfig, MountConfig, MountRequest, RunEvent};
use axum::{Router, body::Body, response::Response, routing::post};

/// The first turn: one `shell` tool call reading `path`, no text. `cmd` is the parameter
/// name in ailoy's `shell` tool descriptor. The path is absolute: the session starts in its
/// scratch tree, so a relative one would read the throwaway directory instead.
fn tool_call(path: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"role\":\"assistant\",\"tool_calls\":[{{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{{\"name\":\"shell\",\"arguments\":\"{{\\\"cmd\\\":\\\"cat {path}\\\"}}\"}}}}]}}}}]}}\n\n\
         data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\n\
         data: [DONE]\n\n"
    )
}

/// The second turn, after the tool result comes back: the answer that ends the run.
const ANSWER: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"It says hi.\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":3,\"total_tokens\":14}}\n\n\
                      data: [DONE]\n\n";

/// The `"default"` provider registry is process-global, and each test points it at a server
/// living on that test's own runtime. Overlapping would aim one test's agent at a listener
/// whose runtime has already gone — which reads as a model error, not as the race it is.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// No live test may hang the suite: a console that never answers, or a mount the kernel
/// stops serving, would otherwise block the recv loop forever with nothing to read.
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

/// Serve the tool call first, then the answer to every request after it — so a retry
/// cannot hang the run by replaying the tool call forever.
async fn scripted_model(reads: &str) -> std::net::SocketAddr {
    let first = Arc::new(tool_call(reads));
    let calls = Arc::new(AtomicUsize::new(0));
    let app = Router::new().route(
        "/",
        post(move || {
            let (calls, first) = (calls.clone(), first.clone());
            async move {
                let body = if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    first.as_str().to_string()
                } else {
                    ANSWER.to_string()
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
    addr
}

/// The engine builds agents against the `"default"` bundle. Spelled out rather than
/// `or_default`: `LangModelProvider::default()` reads the ambient API keys out of the
/// environment, and these tests want only `fake/*`. `ToolProvider::new()` (not `empty()`)
/// carries the builtins — `shell` among them.
fn point_default_at(addr: std::net::SocketAddr) {
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
}

/// Drive one run to `Done`, under a timeout, asserting the `shell` call was announced.
async fn run_to_done(engine: &Engine, session_id: &str, query: &str) {
    let mut h = engine
        .run_start(session_id, vec![Part::text(query)])
        .await
        .unwrap();
    let watch = async {
        let mut tool_started = false;
        loop {
            match h.events.recv().await.unwrap() {
                RunEvent::ToolCallStarted { name, .. } => {
                    assert_eq!(name, "shell");
                    tool_started = true;
                }
                RunEvent::Done => break tool_started,
                RunEvent::Cancelled => panic!("cancelled unexpectedly"),
                RunEvent::Error {
                    kind,
                    message,
                    status,
                    retryable,
                } => panic!("{kind}: {message} (status {status:?}, retryable {retryable})"),
                _ => {}
            }
        }
    };
    let tool_started = tokio::time::timeout(RUN_TIMEOUT, watch)
        .await
        .expect("the run did not reach a terminal event within the timeout");
    assert!(tool_started, "the run never announced the shell call");
}

/// The `stdout` the console reported for the run's one tool call.
async fn tool_stdout(engine: &Engine, session_id: &str) -> String {
    let msgs = engine.message_list(session_id).await.unwrap();
    let tool = msgs
        .iter()
        .find(|m| m.message.role == Role::Tool)
        .expect("a tool result was persisted");
    let v = tool.message.contents[0].as_value().expect("a value part");
    v.pointer("/stdout")
        .and_then(|s| s.as_str())
        .unwrap_or_default()
        .to_string()
}

/// The unmounted path: the workspace is `files/` on the host, handed to the console as its
/// artifacts tree, so the tool reads the root store straight off the disk.
#[tokio::test]
#[ignore]
async fn a_run_reads_a_workspace_file_through_the_shell_tool() {
    let _serial = SERIAL.lock().await;
    let dir = tempfile::tempdir().unwrap();
    // By its own path, not a relative one: the session starts in its scratch directory, so
    // `cat hello.txt` would read the throwaway tree and find nothing.
    let hello = dir.path().join("files").join("hello.txt");
    point_default_at(scripted_model(&hello.display().to_string()).await);

    let mut cfg = EngineConfig::new(dir.path());
    cfg.mount_workspace = false; // the console stands in files/ directly
    cfg.catalog_refresh = false;
    let engine = Engine::start(cfg).await.unwrap();
    engine
        .fs_write("/hello.txt", "hi from the workspace")
        .await
        .unwrap();

    let s = engine.session_create(Some("fake/m".into())).await.unwrap();
    run_to_done(&engine, &s.id, "what does hello.txt say?").await;

    let stdout = tool_stdout(&engine, &s.id).await;
    assert!(
        stdout.contains("hi from the workspace"),
        "the console did not read the workspace file: {stdout:?}"
    );
    assert_eq!(
        engine
            .message_list(&s.id)
            .await
            .unwrap()
            .last()
            .unwrap()
            .message
            .contents[0]
            .as_text(),
        Some("It says hi.")
    );
    engine.shutdown().await;
}

/// The mounted path, which is the one the app actually ships: the workspace is a FUSE-T
/// mount, the console's cwd is that mount point, and the file the tool reads lives in a
/// *connector* — a directory elsewhere on the disk, grafted in at `/docs`. Nothing but the
/// kernel mount makes that path exist for a separate process, so this is the test that
/// fails if `WorkFs`, the FUSE-T binding or the console's mount handoff regresses.
///
/// Needs FUSE-T. If it is killed mid-run it can leave a mount behind: `mount | grep
/// workspace`, then `umount` (or `diskutil unmount force`).
///
/// The data directory is deliberately *not* a `TempDir`. `TempDir::drop` runs
/// `remove_dir_all`, which would walk straight into the mount — and if the mount is still
/// up (this test panicking before `shutdown`, or an unmount that did not take) that walk
/// blocks in an uninterruptible syscall and hangs the whole test binary, with no assertion
/// message and nothing a `kill` can do about it. A plain directory removed by hand at the
/// end trades a leaked temp directory on failure for a test that always reports.
#[tokio::test]
#[ignore]
async fn a_mounted_run_reads_a_connector_through_the_kernel() {
    let _serial = SERIAL.lock().await;
    // The connector's backing directory, outside the workspace entirely. This one holds no
    // mount, so it can clean up after itself.
    let host = tempfile::tempdir().unwrap();
    std::fs::write(host.path().join("f.txt"), b"hi from docs").unwrap();

    let dir = tempfile::tempdir().unwrap().keep();
    // Through the mount point, by absolute path: the session stands in its scratch, and the
    // connector exists for a separate process only because the kernel answers for it here.
    let through_the_mount = dir.join("workspace").join("docs").join("f.txt");
    point_default_at(scripted_model(&through_the_mount.display().to_string()).await);

    let mut cfg = EngineConfig::new(&dir);
    cfg.mount_workspace = true;
    cfg.catalog_refresh = false;
    let engine = Engine::start(cfg).await.unwrap();
    assert!(
        matches!(
            engine.workspace_info().status,
            ailoy_desktop_core::WorkspaceStatus::Mounted
        ),
        "FUSE-T did not mount the workspace: {:?}",
        engine.workspace_info()
    );

    engine
        .mount_add(MountRequest {
            path: "/docs".into(),
            label: None,
            config: MountConfig::Local {
                host_root: host.path().to_path_buf(),
            },
        })
        .await
        .unwrap();

    let s = engine.session_create(Some("fake/m".into())).await.unwrap();
    run_to_done(&engine, &s.id, "what does docs/f.txt say?").await;

    let stdout = tool_stdout(&engine, &s.id).await;
    assert!(
        stdout.contains("hi from docs"),
        "the console did not read the connector through the mount: {stdout:?}"
    );

    // The mount comes down with the engine. `FuseTMount::drop` alone is not enough — the
    // console that just ran had its cwd inside the mount and takes a moment to die, so the
    // kernel answers the first `umount` with `EBUSY`; `WorkspaceManager::shutdown` is what
    // checks and escalates. Asserted here because a leaked FUSE mount outlives the process.
    engine.shutdown().await;
    let mountpoint = engine.config().mountpoint();
    let leaked = ailoy_desktop_core::workspace::is_mounted(&mountpoint);
    if !leaked {
        let _ = std::fs::remove_dir_all(&dir);
    }
    assert!(
        !leaked,
        "the workspace is still mounted after shutdown; clean it up with `umount {}` \
         (the data directory {} was left in place on purpose)",
        mountpoint.display(),
        dir.display()
    );
}
