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
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex},
};

use ailoy::{
    agent::{AgentBuilder, AgentError, RunControl},
    message::{Message, Part, RateLimitInfo, Role, TokenUsage},
};
use cortex::console::{Console, LocalBackend};
use futures::{FutureExt as _, StreamExt as _};
use tokio::sync::{Mutex, broadcast};
use tokio_util::sync::CancellationToken;

use crate::{
    assembler::{AssembledItem, MessageAssembler},
    catalog::Catalog,
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
    /// The console server each run starts its own console on; `None` runs without one, and
    /// only the tools that need a shell fail, saying so.
    pub console: Option<LocalBackend>,
    pub workspace: Arc<WorkspaceManager>,
    pub catalog: Arc<Catalog>,
    /// Where each run's scratch directory is made. Cortex starts the session in it, so a
    /// relative path a command writes lands there rather than among the user's files.
    pub scratch_root: PathBuf,
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
    /// One handle per spawned run task, so [`wait_idle`](Self::wait_idle) can join them at
    /// shutdown. Pruned of finished tasks on every `start`, so a long-lived process does
    /// not accumulate one handle per message ever sent.
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
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
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// Wait for every run task to finish, up to `timeout`. `true` when they all did.
    ///
    /// This is what makes `cancel_all` mean something at shutdown: cancelling only *asks*,
    /// and a run answers on its own terms — it flushes the assembler's trailing message
    /// into SQLite and touches the session before it returns. Exiting the process without
    /// joining loses exactly that last write, which is the partial answer the user was
    /// watching. A timeout is still a bounded shutdown: `false` means one run is past its
    /// grace period and the caller stops waiting for it.
    pub async fn wait_idle(&self, timeout: std::time::Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        let mut tasks = self.tasks.lock().await;
        while let Some(handle) = tasks.pop() {
            // A `JoinError` is a task that panicked or was aborted — finished either way,
            // and `run_task` already caught the panic and reported it.
            if tokio::time::timeout_at(deadline, handle).await.is_err() {
                return false;
            }
        }
        true
    }

    pub async fn is_running(&self, session_id: &str) -> bool {
        self.runs.lock().await.contains_key(session_id)
    }

    /// Re-subscribe to a run already in flight, with the assistant text streamed so far.
    /// `None` when nothing is running for `session_id`.
    pub async fn attach(&self, session_id: &str) -> Option<(RunHandle, String)> {
        let runs = self.runs.lock().await;
        let run = runs.get(session_id)?;
        let partial = lock_partial(&run.partial).clone();
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

        let task = tokio::spawn(async move {
            let actor = drive(deps, &sid, &model, user_msg, tx.clone(), cancel, partial);
            run_task(&sid, actor, tx, finished).await;
        });
        {
            let mut tasks = self.tasks.lock().await;
            tasks.retain(|h| !h.is_finished());
            tasks.push(task);
        }

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

/// The body of the detached run task: drive the run, survive a panic in it, leave the
/// active map, then send exactly one terminal event — in that order.
///
/// The panic is caught rather than left to unwind the task: a panic anywhere in `drive`
/// would otherwise skip both the map removal and the terminal event, leaving the session
/// answering `AlreadyRunning` forever and `attach` handing out a receiver that never
/// yields again. Every exit — return, panic — reaches the tail below.
///
/// Named (rather than inlined at the spawn site) so a test can run it over a future that
/// panics and check both halves of that tail.
async fn run_task(
    session_id: &str,
    actor: impl std::future::Future<Output = std::result::Result<(), RunEnd>>,
    tx: broadcast::Sender<RunEvent>,
    finished: impl std::future::Future<Output = ()>,
) {
    let outcome = std::panic::AssertUnwindSafe(actor)
        .catch_unwind()
        .await
        .unwrap_or_else(|payload| {
            let message = panic_message(&*payload);
            tracing::error!("the run task for {session_id} panicked: {message}");
            Err(fail("internal", message))
        });
    // The run leaves the active map before its terminal event goes out, so a client that
    // reacts to `Done` by asking `is_running` is told the truth.
    finished.await;
    match outcome {
        Ok(()) => {
            let _ = tx.send(RunEvent::Done);
        }
        Err(RunEnd::Cancelled) => {
            let _ = tx.send(RunEvent::Cancelled);
        }
        Err(RunEnd::Failed {
            kind,
            message,
            status,
            retryable,
        }) => {
            let _ = tx.send(RunEvent::Error {
                kind,
                message,
                status,
                retryable,
            });
        }
    }
}

/// How a run ended when it did not simply finish.
enum RunEnd {
    Cancelled,
    Failed {
        kind: String,
        message: String,
        /// The provider's HTTP status, when the failure was a model request that got a
        /// response. `None` for a transport failure and for every non-model failure.
        status: Option<u16>,
        /// Whether the same request may succeed if it is sent again — what the UI's
        /// "retry" button is allowed to be enabled by. Only a [`ModelError`] sets it.
        retryable: bool,
    },
}

/// A failure with nothing for a client to act on beyond its message: a storage write, a
/// console that would not start, a tool that threw. Only a model request answers `status`
/// and `retryable`, and it is built by hand at the one site that has a [`ModelError`].
fn fail(kind: &str, message: String) -> RunEnd {
    RunEnd::Failed {
        kind: kind.into(),
        message,
        status: None,
        retryable: false,
    }
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
///
/// The rule for the `fail(..)` sites below: an error whose `Display` hides its cause is
/// rendered this way (and an `anyhow::Error`, which this cannot take, as the equivalent
/// `{e:#}`); an error whose `Display` already embeds it — every `EngineError`, whose
/// storage/console/workspace variants interpolate their source, and `AgentError::Model`,
/// which is `#[error(transparent)]` — is rendered with plain `to_string()`, because
/// `{:#}` would print the same cause twice.
fn with_causes(e: impl std::error::Error + Send + Sync + 'static) -> String {
    format!("{:#}", anyhow::Error::new(e))
}

/// The live-text buffer, taken even when a panicking run poisoned its lock. The string is
/// at worst stale text nobody will read again; honouring the poison would instead panic
/// every later touch of it — including the `attach` of a window that only wants to read.
fn lock_partial(partial: &StdMutex<String>) -> std::sync::MutexGuard<'_, String> {
    partial.lock().unwrap_or_else(|e| e.into_inner())
}

/// Combine a completed message's accounting with a trailer's, field by field.
///
/// The two describe the *same* turn, so the answer is the larger of each pair, never their
/// sum: a provider that reports a field once leaves the other side at zero (or `None`), and
/// one that repeats it sends the same number twice.
fn merge_usage(a: Option<TokenUsage>, b: Option<TokenUsage>) -> Option<TokenUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(TokenUsage {
            input_tokens: a.input_tokens.max(b.input_tokens),
            output_tokens: a.output_tokens.max(b.output_tokens),
            cache_creation_input_tokens: max_opt(
                a.cache_creation_input_tokens,
                b.cache_creation_input_tokens,
            ),
            cache_read_input_tokens: max_opt(a.cache_read_input_tokens, b.cache_read_input_tokens),
        }),
        (Some(u), None) | (None, Some(u)) => Some(u),
        (None, None) => None,
    }
}

/// The larger of two optional counts; whichever is present when only one is.
fn max_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(v), None) | (None, Some(v)) => Some(v),
        (None, None) => None,
    }
}

/// Put a history read back from SQLite into a shape a provider will accept, after the
/// caller has dropped the turn the agent is about to push itself.
///
/// Two repairs, both for histories this engine wrote and then abandoned:
///
/// * A run can die between an assistant's tool calls and their results — the app was
///   killed, the process crashed — and come back as a batch nobody answered. ailoy only
///   repairs histories it owns, so the close-out happens here.
/// * A trailing `Role::User` row is a run that never got an answer at all: cancelled
///   before its first token, or failed at `build()` or at the console spawn, with the
///   user's message already persisted (`start` writes it before anything can fail).
///   Replaying it would put two consecutive user turns on the wire — which Anthropic
///   merges, Gemini rejects outright, and every provider bills for twice.
fn normalize_replay(history: &mut Vec<Message>) {
    ailoy::agent::close_dangling_tool_calls(history, ailoy::agent::INTERRUPTED_BY_FAILURE);
    while history.last().is_some_and(|m| m.role == Role::User) {
        history.pop();
    }
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
    let settings =
        providers::read_settings(&deps.store).map_err(|e| fail("storage", e.to_string()))?;
    let mut history = deps
        .store
        .message_history(session_id)
        .map_err(|e| fail("storage", e.to_string()))?;
    // `start` appended the user message just now, and the agent pushes its own copy of
    // the query, so the replayed history stops one short of the end.
    history.pop();
    normalize_replay(&mut history);

    let mounts = deps.workspace.mounts().await;
    let degraded = matches!(
        deps.workspace.info().status,
        crate::types::WorkspaceStatus::Degraded { .. }
    );
    let context = deps.workspace.console_context();
    let artifacts = deps.workspace.console_artifacts();
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let extra = deps
        .store
        .setting_get("extra_instruction")
        .map_err(|e| fail("storage", e.to_string()))?;
    let preamble = prompt::build(&prompt::PromptInput {
        workfs_path: &context,
        artifacts_path: &artifacts,
        mounts: &mounts,
        today: &today,
        os: std::env::consts::OS,
        model,
        degraded,
        extra: extra.as_deref(),
    });

    // No backend means no console: the pure tools still work, and a tool that needs a shell
    // answers "needs a console" instead of the run failing outright.
    //
    // The scratch directory is this run's alone and goes away with it: kept as a `TempDir`
    // for the length of this function so it is removed however the run ends — done,
    // cancelled, failed or panicking — rather than on a path only the happy ending reaches.
    let (console, _scratch) = match &deps.console {
        None => (None, None),
        Some(backend) => {
            std::fs::create_dir_all(&deps.scratch_root)
                .map_err(|e| fail("console_unavailable", e.to_string()))?;
            let scratch = tempfile::TempDir::new_in(&deps.scratch_root)
                .map_err(|e| fail("console_unavailable", e.to_string()))?;
            let console = open_console(
                backend.clone(),
                context,
                artifacts,
                scratch.path().to_path_buf(),
            )
            .await
            .map_err(|e| fail("console_unavailable", format!("{e:#}")))?;
            (Some(console), Some(scratch))
        }
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
    // `build()` answers `anyhow::Error`, whose `to_string()` is only the outermost
    // context ("unknown model") — `{:#}` is `with_causes` for a type that isn't `Error`.
    let mut agent = builder
        .build()
        .map_err(|e| fail("model", format!("{e:#}")))?;

    let ctl = RunControl {
        cancel,
        max_turns: Some(settings.max_turns),
        ..Default::default()
    };
    let catalog_model = deps.catalog.lookup(model);
    let context_limit = catalog_model.as_ref().and_then(|m| m.context);

    let mut assembler = MessageAssembler::new();
    // The last top-level message written, with the accounting it was written with — what a
    // usage trailer (see `AssembledItem::UsageTrailer`) belongs to. Replaced by the next
    // top-level message, and cleared if that one fails to persist.
    let mut last_top_level: Option<(i64, Option<TokenUsage>, Option<RateLimitInfo>)> = None;
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
                // The one failure a client can act on: the provider's own status and its
                // verdict on whether sending the same request again could work. Everything
                // else below is `status: None, retryable: false`.
                Err(AgentError::Model(m)) => {
                    end = Some(RunEnd::Failed {
                        kind: "model".into(),
                        message: m.to_string(),
                        status: m.status,
                        retryable: m.retryable,
                    });
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
                        lock_partial(&partial).push_str(&t);
                        let _ = tx.send(RunEvent::TextDelta { text: t });
                    }
                    AssembledItem::Thinking(t) => {
                        let _ = tx.send(RunEvent::ThinkingDelta { text: t });
                    }
                    AssembledItem::Completed(out) => {
                        lock_partial(&partial).clear();
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
                        let (u, rl) = (out.usage.clone(), out.rate_limit.clone());
                        let seq = persist(&deps.store, session_id, &tx, *out);
                        if top_level {
                            // `None` when the write failed: there is no row for a later
                            // trailer to amend, so it has nowhere to land either.
                            last_top_level = seq.map(|seq| (seq, u, rl));
                        }
                    }
                    // Accounting that arrived after the message it describes was already
                    // written — the ChatCompletion schema's `include_usage` frame. It is
                    // both re-announced and back-filled onto that row, so a reload shows
                    // the same totals the live feed did.
                    AssembledItem::UsageTrailer(trailer) => {
                        // A sub-agent's trailer (depth ≥ 1) has no top-level message to
                        // attribute it to, and neither does one that arrives before
                        // anything has closed.
                        let top_level = trailer.depth.unwrap_or(0) == 0;
                        let target = last_top_level.clone().filter(|_| top_level);
                        let Some((seq, prev_usage, prev_rl)) = target else {
                            tracing::debug!(
                                "a usage trailer for {session_id} at depth {:?} has no message to attribute it to",
                                trailer.depth
                            );
                            continue;
                        };
                        let merged = merge_usage(prev_usage, trailer.usage);
                        let rate_limit = trailer.rate_limit.or(prev_rl);
                        let _ = tx.send(RunEvent::Usage {
                            usage: merged.clone(),
                            rate_limit,
                            context_used: merged.as_ref().map(usage::context_used),
                            context_limit,
                        });
                        if let Some(u) = &merged
                            && let Err(e) = deps.store.message_set_usage(session_id, seq, u)
                        {
                            tracing::error!(
                                "recording the usage trailer on message {seq} of {session_id}: {e}"
                            );
                        }
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
            lock_partial(&partial).clear();
            // And the same accounting: a turn cut short still spent what it spent.
            if out.depth.unwrap_or(0) == 0 && (out.usage.is_some() || out.rate_limit.is_some()) {
                let _ = tx.send(RunEvent::Usage {
                    usage: out.usage.clone(),
                    rate_limit: out.rate_limit.clone(),
                    context_used: out.usage.as_ref().map(usage::context_used),
                    context_limit,
                });
            }
            persist(&deps.store, session_id, &tx, out);
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

/// Write one finalized message, then announce it with the sequence number it got — which
/// is also returned, so a usage trailer arriving later knows which row to amend. A store
/// failure is logged and answered with `None` rather than ending the run: the answer is
/// already on screen, and failing the run would only lose the rest of it too.
fn persist(
    store: &Store,
    session_id: &str,
    tx: &broadcast::Sender<RunEvent>,
    out: ailoy::message::MessageOutput,
) -> Option<i64> {
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
            Some(seq)
        }
        Err(e) => {
            tracing::error!("persisting message for {session_id}: {e}");
            None
        }
    }
}

/// One console over the three trees a run works with.
///
/// * `context` — the user's own files and their connectors. Cortex refuses a write that
///   lands here, which is the point: this tree is managed outside the agent's life and an
///   agent reads it.
/// * `artifacts` — what this agent produces. Part of the workspace the user sees, and the
///   one tree here the agent may write.
/// * `scratch` — the run's `/tmp`. The session *starts* here, so a relative path a command
///   writes lands in something thrown away rather than among the user's files.
///
/// Its own function only so that `tests::the_three_trees_have_the_access_each_is_meant_to`
/// can hold this arrangement to account: swapping two of these still starts a console, and
/// fails a whole run away, at the first write.
async fn open_console(
    backend: LocalBackend,
    context: PathBuf,
    artifacts: PathBuf,
    scratch: PathBuf,
) -> anyhow::Result<Console> {
    Console::builder()
        .backend(backend)
        .context(context)
        .artifacts(artifacts)
        .scratch(scratch)
        .build()
        .await
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
        let workspace = Arc::new(
            WorkspaceManager::start(
                dir.join("files"),
                dir.join("artifacts"),
                dir.join("ws"),
                false,
            )
            .await,
        );
        // No console: a text-only run never needs one, and `drive` skips the spawn.
        RunDeps {
            store,
            console: None,
            workspace,
            catalog: Arc::new(Catalog::from_data(CatalogData::default())),
            scratch_root: dir.join("scratch"),
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
        // The accounting rides on the same frame as the `finish_reason` — what
        // Anthropic/Gemini/Responses do, and what an OpenAI-compatible server that ignores
        // `stream_options.include_usage` sends. (The separate trailing frame the schema
        // normally uses is the next test.) `prompt_tokens` 7 with no cache detail, so
        // `context_used` is 7.
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
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
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
        // And the task behind it is joinable, which is what `Engine::shutdown` waits on
        // instead of sleeping and hoping. It is already finished here, so this returns at
        // once — the two seconds are slack for a loaded CI box, not an expected wait.
        assert!(
            mgr.wait_idle(std::time::Duration::from_secs(2)).await,
            "the run task outlived its own `Done`"
        );
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
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
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
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
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

    #[tokio::test]
    async fn a_panicking_actor_frees_the_session_and_reports_an_error() {
        // `run_task`'s tail is the invariant: however the actor ends — returning, or
        // panicking halfway — the run leaves the active map *and* sends exactly one
        // terminal event, in that order. Driven directly over a panicking future, because
        // no fake model can provoke a panic in `drive` now that its locks tolerate poison
        // (the test below covers that), and because a timing-free test is a stabler guard.
        async fn boom() -> std::result::Result<(), RunEnd> {
            panic!("boom inside the actor");
        }
        let dir = tempfile::tempdir().unwrap();
        let mgr = RunManager::new(deps(dir.path()).await);
        let (tx, mut rx) = broadcast::channel(EVENT_BUFFER);
        // Register the run the way `start` does, minus the agent.
        mgr.runs.lock().await.insert(
            "s3".into(),
            ActiveRun {
                run_id: "r3".into(),
                cancel: CancellationToken::new(),
                events: tx.clone(),
                partial: Arc::new(StdMutex::new(String::new())),
            },
        );
        assert!(mgr.is_running("s3").await);

        let finished = mgr.finisher("s3".into());
        run_task("s3", boom(), tx, finished).await;

        match rx.recv().await.expect("a terminal event") {
            RunEvent::Error { kind, message, .. } => {
                assert_eq!(kind, "internal");
                assert!(message.contains("boom inside the actor"), "{message}");
            }
            other => panic!("expected a terminal error, got {other:?}"),
        }
        // `run_task` owned the only sender: the channel closing here is "and nothing else".
        assert!(rx.recv().await.is_err(), "exactly one terminal event");
        // The point of the test: the session is free again rather than stuck answering
        // `AlreadyRunning` forever, with `attach` handing out a dead receiver.
        assert!(!mgr.is_running("s3").await);
        assert!(mgr.attach("s3").await.is_none());
    }

    // Same reasoning as above for the registry lock.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_poisoned_partial_lock_does_not_derail_the_run() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Half \"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"an answer\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\" and the rest\"},\"finish_reason\":\"stop\"}]}\n\n\
                           data: [DONE]\n\n";
        let addr = slow_model_server(SSE, std::time::Duration::from_millis(100)).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s4", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s4", vec![Part::text("hi")]).await.unwrap();
        loop {
            match handle.events.recv().await.unwrap() {
                RunEvent::TextDelta { .. } => break,
                RunEvent::Done | RunEvent::Cancelled => panic!("the run ended before it spoke"),
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
                _ => {}
            }
        }

        // Poison the live-text mutex from outside the actor, mid-answer. A run must not
        // die of someone else's panic: the buffer holds a copy of text that is also on its
        // way to the store, and the answer still has two deltas to go.
        let partial = mgr
            .runs
            .lock()
            .await
            .get("s4")
            .expect("the run to still be active")
            .partial
            .clone();
        let _ = std::thread::spawn(move || {
            let _held = partial.lock().expect("partial mutex");
            panic!("poisoning the partial mutex on purpose");
        })
        .join();

        let mut text = String::new();
        let mut done = false;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::TextDelta { text: t } => text.push_str(&t),
                RunEvent::Done => {
                    done = true;
                    break;
                }
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
                RunEvent::Cancelled => panic!("cancelled unexpectedly"),
                _ => {}
            }
        }
        assert!(done, "the run finished despite the poisoned lock");
        assert!(text.ends_with(" and the rest"), "{text:?}");
        let msgs = store.message_list("s4").unwrap();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert_eq!(
            msgs[1].message.contents[0].as_text(),
            Some("Half an answer and the rest")
        );
    }

    // Same reasoning as above for the registry lock.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn a_usage_trailer_is_billed_to_the_message_it_closes() {
        let _g = crate::providers::REGISTRY_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // The ChatCompletion schema's real shape: ailoy asks for
        // `stream_options.include_usage`, so the turn's accounting arrives in a frame of
        // its own — empty `choices`, no role, no content, no `finish_reason` — *after* the
        // frame that ended the message. It belongs to the message it follows.
        const SSE: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n\
                           data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
                           data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n\
                           data: [DONE]\n\n";
        let addr = fake_model_server(SSE).await;
        point_default_at(addr);
        let dir = tempfile::tempdir().unwrap();
        let d = deps(dir.path()).await;
        let store = d.store.clone();
        store.session_create("s5", "t", "fake/m").unwrap();
        let mgr = RunManager::new(d);

        let mut handle = mgr.start("s5", vec![Part::text("hi")]).await.unwrap();
        let mut usage_events = 0;
        let mut assistant_message_seq = None;
        let mut usage_after_the_message = 0;
        let mut last_usage = None;
        let mut done = false;
        while let Ok(ev) = handle.events.recv().await {
            match ev {
                RunEvent::Message { seq, message, .. } if message.role == Role::Assistant => {
                    assistant_message_seq = Some(seq);
                }
                RunEvent::Usage {
                    usage,
                    context_used,
                    ..
                } => {
                    usage_events += 1;
                    if assistant_message_seq.is_some() {
                        usage_after_the_message += 1;
                    }
                    last_usage = Some((usage, context_used));
                }
                RunEvent::Done => {
                    done = true;
                    break;
                }
                RunEvent::Error { kind, message, .. } => panic!("{kind}: {message}"),
                RunEvent::Cancelled => panic!("cancelled unexpectedly"),
                _ => {}
            }
        }
        assert!(done, "`Done` is the last event");
        // One event, not none (the trailer dropped) and not two (the tool-result message
        // billed for the model's turn as well).
        assert_eq!(usage_events, 1);
        assert_eq!(
            usage_after_the_message, 1,
            "the trailer follows its message"
        );
        let (usage, context_used) = last_usage.expect("a usage event");
        let usage = usage.expect("the trailer's counts");
        assert_eq!(usage.output_tokens, 2);
        assert_eq!(context_used, Some(7));

        // And it is written back onto the row that was persisted without it, so a reload
        // shows what the live feed showed.
        let msgs = store.message_list("s5").unwrap();
        assert_eq!(msgs.len(), 2, "{msgs:?}");
        assert_eq!(msgs[1].seq, assistant_message_seq.unwrap());
        assert_eq!(msgs[1].usage.as_ref().unwrap().input_tokens, 7);
        assert_eq!(msgs[1].usage.as_ref().unwrap().output_tokens, 2);
        assert_eq!(store.message_usages("s5").unwrap().len(), 1);
    }

    /// The replayed history must never end on a user turn. `start` persists the user's
    /// message before anything can fail, so a run cancelled before its first token — or
    /// one that died at `build()` or at the console spawn — leaves an unanswered user row
    /// behind. Replaying it and then pushing the new query puts two consecutive user turns
    /// on the wire.
    #[test]
    fn replay_normalization_drops_the_user_turns_nothing_answered() {
        let msg = |role: Role| Message::new(role).with_contents([Part::text("x")]);
        let roles = |h: &[Message]| h.iter().map(|m| m.role.clone()).collect::<Vec<_>>();
        let normalized = |input: Vec<Role>| {
            let mut h: Vec<Message> = input.into_iter().map(msg).collect();
            normalize_replay(&mut h);
            roles(&h)
        };

        // The ordinary case, after `drive` popped the turn the agent will push itself.
        assert_eq!(
            normalized(vec![Role::User, Role::Assistant, Role::User]),
            vec![Role::User, Role::Assistant]
        );
        // A first run that never got an answer: nothing is left to replay.
        assert_eq!(normalized(vec![Role::User]), Vec::<Role>::new());
        // Two of them in a row — a second attempt that also died before the model spoke.
        assert_eq!(normalized(vec![Role::User, Role::User]), Vec::<Role>::new());
        // An answered turn is left exactly as it was.
        assert_eq!(
            normalized(vec![Role::User, Role::Assistant]),
            vec![Role::User, Role::Assistant]
        );
    }

    #[test]
    fn merging_usage_takes_each_field_from_whichever_frame_reported_it() {
        let u = |i, o, cr| TokenUsage {
            input_tokens: i,
            output_tokens: o,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: cr,
        };
        // The pair describes one turn: the larger of each field, never the sum — a
        // message finalized with `output_tokens` 0 plus a trailer reporting 2 is 2.
        let merged = merge_usage(Some(u(7, 0, None)), Some(u(0, 2, Some(3)))).unwrap();
        assert_eq!((merged.input_tokens, merged.output_tokens), (7, 2));
        assert_eq!(merged.cache_read_input_tokens, Some(3));
        // A repeated report is idempotent, not doubled.
        let twice = merge_usage(Some(u(7, 2, Some(3))), Some(u(7, 2, Some(3)))).unwrap();
        assert_eq!((twice.input_tokens, twice.output_tokens), (7, 2));
        // One side missing entirely is the other side.
        assert_eq!(
            merge_usage(None, Some(u(1, 1, None))).unwrap().input_tokens,
            1
        );
        assert_eq!(
            merge_usage(Some(u(1, 1, None)), None)
                .unwrap()
                .output_tokens,
            1
        );
        assert!(merge_usage(None, None).is_none());
    }

    /// The three trees a run is given, and what each one is for.
    ///
    /// This is the arrangement the whole desktop rests on, and every part of it fails quietly if
    /// the roles are swapped: the user's workspace goes in as the *context*, which cortex refuses
    /// to let the agent write; what the agent produces goes in its *artifacts*; and the session
    /// stands in its *scratch*, so a relative path is a throwaway one. Getting context and
    /// artifacts the wrong way round still starts a console — it fails at the first write, which
    /// is a whole run away from here.
    #[tokio::test]
    async fn the_three_trees_have_the_access_each_is_meant_to() {
        let workspace = tempfile::tempdir().unwrap();
        let artifacts = tempfile::tempdir().unwrap();
        let scratch = tempfile::tempdir().unwrap();
        std::fs::write(workspace.path().join("theirs.txt"), b"the user's").unwrap();

        // A real console: the local server cortex carries, written out under a home of its own.
        let home = tempfile::tempdir().unwrap();
        let mut console = open_console(
            cortex::console::Backend::local().home(home.path()),
            workspace.path().to_path_buf(),
            artifacts.path().to_path_buf(),
            scratch.path().to_path_buf(),
        )
        .await
        .unwrap();

        // Where it stands: a relative path is the scratch, not the user's files.
        let out = console.exec(["pwd"], Some(5_000)).await.unwrap();
        let cwd = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert_eq!(
            std::fs::canonicalize(&cwd).unwrap(),
            std::fs::canonicalize(scratch.path()).unwrap(),
            "the session should start in its scratch"
        );

        // The workspace is readable.
        let theirs = workspace.path().join("theirs.txt");
        let out = console
            .exec(["cat", &theirs.display().to_string()], Some(5_000))
            .await
            .unwrap();
        assert_eq!(
            out.code,
            0,
            "stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(out.stdout, b"the user's".to_vec());

        // And not writable: the user's tree is managed outside the agent's life.
        let refused = console
            .write(
                workspace.path().join("mine.txt").display().to_string(),
                b"no".to_vec(),
                None,
            )
            .await;
        assert!(
            refused.is_err(),
            "a write into the context should be refused"
        );
        assert!(
            !workspace.path().join("mine.txt").exists(),
            "the refused write must not have landed"
        );

        // The artifacts tree is where the agent's own files go, and it takes the write.
        let mine = artifacts.path().join("mine.txt");
        console
            .write(
                mine.display().to_string(),
                b"from the session".to_vec(),
                None,
            )
            .await
            .expect("the artifacts tree takes a write");
        assert_eq!(std::fs::read_to_string(&mine).unwrap(), "from the session");
    }
}
