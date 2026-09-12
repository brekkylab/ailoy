//! One actor per active run: assemble the agent, drive its stream, persist and broadcast.
//!
//! A run is started by [`RunManager::start`], which appends the user's message, registers
//! the run and spawns a detached task. From then on the only way in is the broadcast
//! channel ([`RunManager::attach`] for a window that reconnects) and the cancellation
//! token ([`RunManager::cancel`]). Every message the agent finalizes is written to the
//! store *before* it is broadcast, so a client that reloads instead of listening sees the
//! same conversation either way.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex},
};

use ailoy::{
    agent::{AgentBuilder, AgentError, RunControl},
    message::{Message, Part, Role},
};
use futures::{FutureExt as _, StreamExt as _};
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;

use crate::{
    assembler::{AssembledItem, MessageAssembler},
    catalog::Catalog,
    console::ConsoleFactory,
    error::{EngineError, Result},
    events::RunEvent,
    prompt, providers,
    store::{NewMessage, Store},
    usage,
    workspace::WorkspaceManager,
};

/// Everything a run needs from the engine around it. All `Arc`s, so cloning it per run is
/// cheap and a detached run borrows nothing.
#[derive(Clone)]
pub struct RunDeps {
    pub store: Arc<Store>,
    pub console: Arc<ConsoleFactory>,
    pub workspace: Arc<WorkspaceManager>,
    pub catalog: Arc<Catalog>,
}

/// A subscription to one run: its id, and the events it will emit from now on.
pub struct RunHandle {
    pub run_id: String,
    pub events: broadcast::Receiver<RunEvent>,
}

struct ActiveRun {
    run_id: String,
    cancel: CancellationToken,
    events: broadcast::Sender<RunEvent>,
    /// Assistant text streamed since the last finalized message — what a window that
    /// re-attaches mid-answer needs to paint before the next delta arrives.
    partial: Arc<StdMutex<String>>,
}

pub struct RunManager {
    deps: RunDeps,
    /// Behind an `Arc` so a detached run task can remove itself when it ends.
    runs: Arc<Mutex<HashMap<String, ActiveRun>>>,
}

/// Deep enough that a slow subscriber on a long tool-heavy run does not lag out: a
/// `broadcast` receiver that falls this far behind gets `RecvError::Lagged`, and the UI
/// would have to reload the session to recover.
const EVENT_BUFFER: usize = 4096;

impl RunManager {
    pub fn new(deps: RunDeps) -> Self {
        Self {
            deps,
            runs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn is_running(&self, session_id: &str) -> bool {
        self.runs.lock().await.contains_key(session_id)
    }

    /// Re-subscribe to a run already in flight, with the assistant text streamed so far.
    /// `None` when nothing is running for `session_id`.
    pub async fn attach(&self, session_id: &str) -> Option<(RunHandle, String)> {
        let runs = self.runs.lock().await;
        let run = runs.get(session_id)?;
        let partial = run.partial.lock().expect("partial mutex").clone();
        Some((
            RunHandle {
                run_id: run.run_id.clone(),
                events: run.events.subscribe(),
            },
            partial,
        ))
    }

    /// Ask a run to stop. `true` if one was running: the run still ends on its own terms
    /// (ailoy commits what it has and answers pending tool calls) and reports `Cancelled`.
    pub async fn cancel(&self, session_id: &str) -> bool {
        match self.runs.lock().await.get(session_id) {
            Some(run) => {
                run.cancel.cancel();
                true
            }
            None => false,
        }
    }

    pub async fn cancel_all(&self) {
        for run in self.runs.lock().await.values() {
            run.cancel.cancel();
        }
    }

    pub async fn start(&self, session_id: &str, parts: Vec<Part>) -> Result<RunHandle> {
        let session = self.deps.store.session_get(session_id)?;
        let mut runs = self.runs.lock().await;
        if runs.contains_key(session_id) {
            return Err(EngineError::AlreadyRunning);
        }

        // The user's message is persisted before anything can fail, so a refresh shows it.
        let user_msg = Message::new(Role::User).with_contents(parts);
        let seq = self.deps.store.message_append(
            session_id,
            NewMessage {
                depth: 0,
                source_agent: None,
                message: &user_msg,
                usage: None,
            },
        )?;
        self.deps.store.session_touch(session_id)?;

        let run_id = uuid::Uuid::new_v4().to_string();
        let (tx, rx) = broadcast::channel(EVENT_BUFFER);
        let cancel = CancellationToken::new();
        let partial = Arc::new(StdMutex::new(String::new()));
        runs.insert(
            session_id.to_string(),
            ActiveRun {
                run_id: run_id.clone(),
                cancel: cancel.clone(),
                events: tx.clone(),
                partial: partial.clone(),
            },
        );
        drop(runs);

        let _ = tx.send(RunEvent::Started {
            run_id: run_id.clone(),
        });
        let _ = tx.send(RunEvent::Message {
            seq,
            depth: 0,
            source_agent: None,
            message: user_msg.clone(),
            usage: None,
        });

        let deps = self.deps.clone();
        let sid = session_id.to_string();
        let model = session.model.clone();
        let finished = self.finisher(sid.clone());

        tokio::spawn(async move {
            // Caught rather than left to unwind the task: a panic anywhere in `drive` would
            // otherwise skip both the map removal and the terminal event, leaving the session
            // answering `AlreadyRunning` forever and `attach` handing out a receiver that
            // never yields again. Every exit — return, panic — now reaches the two lines below.
            let outcome = std::panic::AssertUnwindSafe(drive(
                deps,
                &sid,
                &model,
                user_msg,
                tx.clone(),
                cancel,
                partial,
            ))
            .catch_unwind()
            .await
            .unwrap_or_else(|payload| {
                let message = panic_message(&*payload);
                tracing::error!("the run task for {sid} panicked: {message}");
                Err(RunEnd::Failed {
                    kind: "internal".into(),
                    message,
                })
            });
            // The run leaves the active map before its terminal event goes out, so a
            // client that reacts to `Done` by asking `is_running` is told the truth.
            finished.await;
            match outcome {
                Ok(()) => {
                    let _ = tx.send(RunEvent::Done);
                }
                Err(RunEnd::Cancelled) => {
                    let _ = tx.send(RunEvent::Cancelled);
                }
                Err(RunEnd::Failed { kind, message }) => {
                    let _ = tx.send(RunEvent::Error { kind, message });
                }
            }
        });

        Ok(RunHandle { run_id, events: rx })
    }

    /// A future that removes `session_id` from the active map when awaited. Built while
    /// `self` is borrowed; awaited from the detached task through the map's `Arc`.
    fn finisher(
        &self,
        session_id: String,
    ) -> impl std::future::Future<Output = ()> + Send + 'static {
        let runs = self.runs.clone();
        async move {
            runs.lock().await.remove(&session_id);
        }
    }
}

/// How a run ended when it did not simply finish.
enum RunEnd {
    Cancelled,
    Failed { kind: String, message: String },
}

/// What a caught panic should say. `panic!` payloads are a `&'static str` or a `String`;
/// anything else (a panicking `Drop`, a custom payload) has no rendering, so say what
/// happened instead of nothing.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "run task panicked".to_string()
    }
}

/// An error rendered with its whole `#[source]` chain. `AgentError::Tool`/`Console` display
/// as fixed strings ("tool execution failed") and keep the real cause behind `source`, so
/// `to_string()` alone tells the user nothing about what actually went wrong.
fn with_causes(e: impl std::error::Error + Send + Sync + 'static) -> String {
    format!("{:#}", anyhow::Error::new(e))
}

async fn drive(
    deps: RunDeps,
    session_id: &str,
    model: &str,
    user_msg: Message,
    tx: broadcast::Sender<RunEvent>,
    cancel: CancellationToken,
    partial: Arc<StdMutex<String>>,
) -> std::result::Result<(), RunEnd> {
    let fail = |kind: &str, e: String| RunEnd::Failed {
        kind: kind.into(),
        message: e,
    };

    let settings =
        providers::read_settings(&deps.store).map_err(|e| fail("storage", e.to_string()))?;
    let mut history = deps
        .store
        .message_history(session_id)
        .map_err(|e| fail("storage", e.to_string()))?;
    // `start` appended the user message just now, and the agent pushes its own copy of
    // the query, so the replayed history stops one short of the end.
    history.pop();
    // A history restored from SQLite can end mid-tool-batch (the app died between the
    // assistant's tool calls and their results); ailoy only repairs histories it owns.
    ailoy::agent::close_dangling_tool_calls(&mut history, ailoy::agent::INTERRUPTED_BY_FAILURE);

    let mounts = deps.workspace.mounts().await;
    let ws_mount = deps.workspace.console_mount();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let extra = deps
        .store
        .setting_get("extra_instruction")
        .map_err(|e| fail("storage", e.to_string()))?;
    let preamble = prompt::build(&prompt::PromptInput {
        workfs_path: &ws_mount.0,
        mounts: &mounts,
        today: &today,
        os: std::env::consts::OS,
        extra: extra.as_deref(),
    });

    // No console binary means no console: the pure tools still work, and a tool that
    // needs a shell answers "needs a console" instead of the run failing outright.
    let console = if deps.console.is_disabled() {
        None
    } else {
        Some(
            deps.console
                .spawn(ws_mount)
                .await
                .map_err(|e| fail("console_unavailable", e.to_string()))?,
        )
    };

    let mut builder = AgentBuilder::new(model)
        .instruction(preamble)
        .system_tools()
        .web_search_tool(vec![])
        .web_fetch_tool()
        .max_tokens(settings.max_tokens)
        .history(history);
    if let Some(c) = console {
        builder = builder.console(c);
    }
    let mut agent = builder.build().map_err(|e| fail("model", e.to_string()))?;

    let ctl = RunControl {
        cancel,
        max_turns: Some(settings.max_turns),
        ..Default::default()
    };
    let catalog_model = deps.catalog.lookup(model);
    let context_limit = catalog_model.as_ref().and_then(|m| m.context);

    let mut assembler = MessageAssembler::new();
    let mut end: Option<RunEnd> = None;
    {
        let mut stream = agent.run_stream_controlled(user_msg, ctl);
        while let Some(item) = stream.next().await {
            let delta = match item {
                Ok(d) => d,
                Err(AgentError::Cancelled) => {
                    end = Some(RunEnd::Cancelled);
                    break;
                }
                Err(AgentError::MaxTurns { turns }) => {
                    end = Some(fail(
                        "max_turns",
                        format!("turn limit reached after {turns} model calls"),
                    ));
                    break;
                }
                Err(AgentError::Model(m)) => {
                    end = Some(fail("model", m.to_string()));
                    break;
                }
                Err(e @ AgentError::Console(_)) => {
                    end = Some(fail("console_unavailable", with_causes(e)));
                    break;
                }
                // `AgentError` is `#[non_exhaustive]`; everything else is tool-layer or
                // unclassified, and reads the same to the user. Bound whole (not as
                // `Tool(e)`) so the fixed variant text keeps its cause chain.
                Err(e) => {
                    end = Some(fail("tool", with_causes(e)));
                    break;
                }
            };
            // Usage is *not* reported per delta: providers split a turn's accounting across
            // the stream (Anthropic sends the input and cache counts first and only
            // `output_tokens` last), so a per-delta event would end every turn showing
            // input 0. The assembler's `accumulate` merges the turn's totals, so the
            // completed message below is the one place they are all present at once.
            let items = match assembler.push(delta) {
                Ok(items) => items,
                Err(e) => {
                    end = Some(fail("stream", e));
                    break;
                }
            };
            for item in items {
                match item {
                    AssembledItem::Text(t) => {
                        partial.lock().expect("partial mutex").push_str(&t);
                        let _ = tx.send(RunEvent::TextDelta { text: t });
                    }
                    AssembledItem::Thinking(t) => {
                        let _ = tx.send(RunEvent::ThinkingDelta { text: t });
                    }
                    AssembledItem::Completed(out) => {
                        partial.lock().expect("partial mutex").clear();
                        // Depth ≥ 1 is a sub-agent's own turn. Its usage is already inside
                        // the tool call the top-level turn will report, and `RunEvent` has
                        // no depth on `Usage`/`ToolCallStarted` to tell the two apart — so
                        // the top-level feed only ever carries depth 0.
                        let top_level = out.depth.unwrap_or(0) == 0;
                        if top_level && (out.usage.is_some() || out.rate_limit.is_some()) {
                            let _ = tx.send(RunEvent::Usage {
                                usage: out.usage.clone(),
                                rate_limit: out.rate_limit.clone(),
                                context_used: out.usage.as_ref().map(usage::context_used),
                                context_limit,
                            });
                        }
                        if top_level
                            && out.message.role == Role::Assistant
                            && let Some(calls) = &out.message.tool_calls
                        {
                            for c in calls {
                                if let Some((id, name, args)) = c.as_function() {
                                    let _ = tx.send(RunEvent::ToolCallStarted {
                                        id: id.into(),
                                        name: name.into(),
                                        arguments: args.clone(),
                                    });
                                }
                            }
                        }
                        persist(&deps.store, session_id, &tx, *out);
                    }
                }
            }
        }
    }
    // A stream that ended mid-message (cancel during the model phase) still has text
    // worth keeping — the agent committed it to its history, so mirror that here.
    match assembler.finish() {
        Ok(Some(out)) => {
            // Same as the in-loop `Completed` path: once the text is a stored message, a
            // client that re-attaches must not also be handed it as live partial text.
            partial.lock().expect("partial mutex").clear();
            persist(&deps.store, session_id, &tx, out)
        }
        Ok(None) => {}
        Err(e) => tracing::error!("finalizing the trailing message for {session_id}: {e}"),
    }
    let _ = deps.store.session_touch(session_id);
    match end {
        None => Ok(()),
        Some(e) => Err(e),
    }
}

/// Write one finalized message, then announce it with the sequence number it got. A store
/// failure is logged rather than ending the run: the answer is already on screen, and
/// failing the run would only lose the rest of it too.
fn persist(
    store: &Store,
    session_id: &str,
    tx: &broadcast::Sender<RunEvent>,
    out: ailoy::message::MessageOutput,
) {
    let depth = out.depth.unwrap_or(0);
    match store.message_append(
        session_id,
        NewMessage {
            depth,
            source_agent: out.source_agent.as_deref(),
            message: &out.message,
            usage: out.usage.as_ref(),
        },
    ) {
        Ok(seq) => {
            let _ = tx.send(RunEvent::Message {
                seq,
                depth,
                source_agent: out.source_agent,
                message: out.message,
                usage: out.usage,
            });
        }
        Err(e) => tracing::error!("persisting message for {session_id}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ailoy::{
        agent::{AgentProvider, get_agent_providers_mut},
        lang_model::{LangModelProvider, get_lm_providers_mut},
        message::Part,
        tool::{ToolProvider, get_tool_providers_mut},
    };
    use axum::{Router, body::Body, response::Response, routing::post};

    use super::*;
    use crate::{
        catalog::{Catalog, CatalogData},
        console::ConsoleFactory,
        store::Store,
        workspace::WorkspaceManager,
    };

    async fn fake_model_server(sse: &'static str) -> std::net::SocketAddr {
        let app = Router::new().route(
            "/",
            post(move || async move {
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
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

    /// The same, one SSE event at a time with `delay` between them, so a run can be
    /// cancelled while the model is still talking.
    async fn slow_model_server(
        sse: &'static str,
        delay: std::time::Duration,
    ) -> std::net::SocketAddr {
        let app = Router::new().route(
            "/",
            post(move || async move {
                let events: Vec<String> = sse
                    .split("\n\n")
                    .filter(|e| !e.trim().is_empty())
                    .map(|e| format!("{e}\n\n"))
                    .collect();
                let stream = futures::stream::iter(events).then(move |e| async move {
                    tokio::time::sleep(delay).await;
                    Ok::<_, std::io::Error>(axum::body::Bytes::from(e))
                });
                Response::builder()
                    .status(200)
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
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

    /// The engine builds agents against the `"default"` bundle; point its lang-model
    /// registry at the fake server under a `fake/*` pattern. All three `"default"`
    /// entries already exist in a running app — the inserts here are the safety net for a
    /// test binary that touches the registries in a different order.
    fn point_default_at(addr: std::net::SocketAddr) {
        let mut lmps = get_lm_providers_mut();
        // Spelled out rather than `or_default`: `LangModelProvider::default()` reads the
        // ambient API keys out of the environment, and this test wants only `fake/*`.
        if !lmps.contains_key("default") {
            lmps.insert("default".into(), LangModelProvider::new());
        }
        let def = lmps
            .get_mut("default")
            .expect("default lang-model registry");
        def.insert(
            "fake/*".into(),
            LangModelProvider::chat_completion(&format!("http://{addr}/"), None).unwrap(),
        );
        drop(lmps);
        // `ToolProvider::new()` (not `empty()`) carries the builtins — `system_tools()`
        // looks them up in this bundle at build time.
        let mut tps = get_tool_providers_mut();
        if !tps.contains_key("default") {
            tps.insert("default".into(), ToolProvider::new());
        }
        drop(tps);
        get_agent_providers_mut()
            .entry("default".into())
            .or_insert_with(|| AgentProvider::new("default", "default"));
    }

    async fn deps(dir: &std::path::Path) -> RunDeps {
        let store = Arc::new(Store::open_in_memory().unwrap());
        let workspace =
            Arc::new(WorkspaceManager::start(dir.join("files"), dir.join("ws"), false).await);
        // No console binary: a text-only run never needs one, and `drive` skips the spawn.
        let console = Arc::new(ConsoleFactory::disabled());
        RunDeps {
            store,
            console,
            workspace,
            catalog: Arc::new(Catalog::from_data(CatalogData::default())),
        }
    }

    // The registry lock is a `std::sync::Mutex` held across awaits on purpose: it is what
    // serializes this test against the other tests in this binary that write ailoy's
    // process-wide `"default"` registry, and `#[tokio::test]`'s current-thread runtime
    // cannot deadlock on it (no other task in this test takes it).
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_text_run_persists_user_and_assistant_and_ends_done() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // The last chunk carries the turn's accounting, the way a ChatCompletion provider
        // does: `prompt_tokens` 7 with no cache detail, so `context_used` is 7.
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n\
                           data: [DONE]\n\n";
        let addr = fake_model_server(SSE).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s1", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s1", vec![Part::text("hi")]).await.unwrap();
        assert!(!handle.run_id.is_empty());
        assert!(mgr.is_running("s1").await);
        assert!(matches!(
            mgr.start("s1", vec![Part::text("again")]).await,
            Err(EngineError::AlreadyRunning)
        ));

        let mut text = String::new();
        let mut done = false;
        let mut started = false;
        let mut usage_events = 0;
        let mut usage_before_any_text = 0;
        let mut last_usage: Option<(Option<u64>, Option<u64>)> = None;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::Started { .. } => started = true,
                RunEvent::TextDelta { text: t } => text.push_str(&t),
                RunEvent::Usage {
                    usage,
                    context_used,
                    ..
                } => {
                    usage_events += 1;
                    if text.is_empty() {
                        usage_before_any_text += 1;
                    }
                    last_usage = Some((usage.map(|u| u.output_tokens), context_used));
                }
                RunEvent::Done => {
                    done = true;
                    break;
                }
                RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
                RunEvent::Cancelled => panic!("cancelled unexpectedly"),
                _ => {}
            }
        }
        assert!(started);
        assert!(done);
        assert_eq!(text, "Hello");
        // Exactly one `Usage`, carrying the completed message's merged totals — not one per
        // delta, and so never ahead of the text it accounts for.
        assert_eq!(usage_events, 1);
        assert_eq!(usage_before_any_text, 0);
        let (output_tokens, context_used) = last_usage.expect("a usage event");
        assert!(output_tokens.unwrap_or(0) > 0, "{output_tokens:?}");
        assert_eq!(context_used, Some(7));
        let msgs = store.message_list("s1").unwrap();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert_eq!(msgs[0].message.role, Role::User);
        assert_eq!(msgs[1].message.contents[0].as_text(), Some("Hello"));
        // `Done` goes out after the run leaves the map, so this needs no grace period.
        assert!(!mgr.is_running("s1").await);
        assert!(mgr.attach("s1").await.is_none());
        assert!(!mgr.cancel("s1").await);
    }

    // Same reasoning as above for the registry lock.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cancelling_mid_answer_reports_cancelled_and_keeps_what_was_said() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Half \"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"an answer\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\" and the rest\"},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";
        let addr = slow_model_server(SSE, std::time::Duration::from_millis(150)).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s2", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s2", vec![Part::text("hi")]).await.unwrap();
        // Wait for the answer to start, so the cancel lands in the model phase.
        let first = loop {
            match handle.events.recv().await.unwrap() {
                RunEvent::TextDelta { text } => break text,
                RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
                RunEvent::Done | RunEvent::Cancelled => panic!("the run ended before it spoke"),
                _ => {}
            }
        };

        // A window that reconnects mid-answer gets the same run and the text so far.
        let (again, partial) = mgr.attach("s2").await.expect("a run to attach to");
        assert_eq!(again.run_id, handle.run_id);
        assert!(partial.starts_with(&first), "{partial:?} vs {first:?}");

        assert!(mgr.cancel("s2").await);
        let mut text = first;
        let mut cancelled = false;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::TextDelta { text: t } => text.push_str(&t),
                RunEvent::Cancelled => {
                    cancelled = true;
                    break;
                }
                RunEvent::Done => panic!("ran to completion despite the cancel"),
                RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
                _ => {}
            }
        }
        assert!(cancelled);
        assert!(!text.is_empty());
        // The partial answer is kept: ailoy commits it to its own history, and the
        // assembler's trailing flush mirrors that into the store.
        let msgs = store.message_list("s2").unwrap();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert_eq!(msgs[1].message.role, Role::Assistant);
        assert_eq!(msgs[1].message.contents[0].as_text(), Some(text.as_str()));
        assert!(!mgr.is_running("s2").await);
    }

    // Same reasoning as above for the registry lock.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_panicking_actor_frees_the_session_and_reports_an_error() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Half \"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"an answer\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\" and the rest\"},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";
        let addr = slow_model_server(SSE, std::time::Duration::from_millis(150)).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s3", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s3", vec![Part::text("hi")]).await.unwrap();
        // Wait until the run is inside its stream loop.
        loop {
            match handle.events.recv().await.unwrap() {
                RunEvent::TextDelta { .. } => break,
                RunEvent::Done | RunEvent::Cancelled => panic!("the run ended before it spoke"),
                RunEvent::Error { kind, message } => panic!("{kind}: {message}"),
                _ => {}
            }
        }

        // Poison the partial-text mutex from outside the actor: `drive` takes it on the
        // next text delta and `expect`s, which is the panic under test. Any panic in the
        // actor would do — this is the one reachable without touching another module.
        let partial = mgr
            .runs
            .lock()
            .await
            .get("s3")
            .expect("the run to still be active")
            .partial
            .clone();
        let _ = std::thread::spawn(move || {
            let _held = partial.lock().expect("partial mutex");
            panic!("poisoning the partial mutex on purpose");
        })
        .join();

        let mut kind = None;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::Error { kind: k, .. } => {
                    kind = Some(k);
                    break;
                }
                RunEvent::Done => panic!("ran to completion despite the panic"),
                RunEvent::Cancelled => panic!("reported a cancel that never happened"),
                _ => {}
            }
        }
        // The panic is reported as a run failure, and — the point of the test — the run
        // does not leak: the session is free again rather than stuck on `AlreadyRunning`.
        assert_eq!(kind.as_deref(), Some("internal"));
        assert!(!mgr.is_running("s3").await);
        assert!(mgr.attach("s3").await.is_none());
        assert!(mgr.start("s3", vec![Part::text("again")]).await.is_ok());
        mgr.cancel("s3").await;
    }
}
