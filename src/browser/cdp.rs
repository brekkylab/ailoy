//! One CDP connection to a browser: calls answered by id, and the dialogs pages open.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

use futures::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio_tungstenite::tungstenite::{self, Message, http::StatusCode, protocol::WebSocketConfig};

/// How long one call may take. A page busy past this is the agent's to wait on, not ours.
const CALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Why a connection was not made.
pub(super) enum Refusal {
    /// Nothing listens, or the browser turned the connection away without asking.
    Unreachable,
    /// The person said no, or did not answer in time.
    Declined,
}

/// Why a call has no answer.
#[derive(Debug)]
pub(super) enum CallError {
    /// The browser answered with an error.
    Browser(String),
    /// A page dialog opened while the call was waiting, and holds the page until answered.
    Dialog,
    /// The connection is gone, or the call timed out.
    Gone(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Browser(e) | Self::Gone(e) => f.write_str(e),
            Self::Dialog => f.write_str("a dialog is open on the page"),
        }
    }
}

impl std::error::Error for CallError {}

/// A dialog a page opened: `alert`, `confirm`, `prompt` or `beforeunload`.
#[derive(Clone, Debug)]
pub(super) struct Dialog {
    pub kind: String,
    pub message: String,
}

/// A download the browser reported, and how far it has got.
#[derive(Clone, Debug, Default)]
pub(super) struct Download {
    /// The frame that began it.
    pub frame: String,
    pub name: String,
    pub received: u64,
    pub total: u64,
    /// `inProgress`, `completed` or `canceled`.
    pub state: String,
    /// Where it was saved, when the browser says.
    pub path: Option<String>,
}

#[derive(Default)]
struct Shared {
    pending: HashMap<u64, oneshot::Sender<Result<Value, CallError>>>,
    /// Dialogs open now, by the session of their page.
    dialogs: HashMap<String, Dialog>,
    /// Alerts accepted on the agent's behalf, by session, until it hears of them.
    accepted: HashMap<String, Vec<Dialog>>,
    /// Frames in a process of their own, attached under the session of the page holding them:
    /// their frame id and their own session.
    frames: HashMap<String, Vec<(String, String)>>,
    /// Downloads in the order they began, by guid.
    downloads: Vec<(String, Download)>,
}

pub(super) struct Cdp {
    outbox: mpsc::UnboundedSender<Message>,
    shared: Arc<Mutex<Shared>>,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    /// Woken when a dialog opens, so calls held up by it can give way.
    dialog_opened: Arc<Notify>,
}

pub(super) fn ws_config() -> WebSocketConfig {
    // Screenshots and accessibility trees run past tungstenite's 16 MiB frame default.
    WebSocketConfig::default()
        .max_message_size(Some(256 << 20))
        .max_frame_size(Some(256 << 20))
}

impl Cdp {
    /// Connects to the browser endpoint `url`, waiting up to `wait` for a person to allow it.
    pub(super) async fn connect(url: &str, wait: Duration) -> Result<Arc<Self>, Refusal> {
        let connecting = tokio_tungstenite::connect_async_with_config(url, Some(ws_config()), true);
        let socket = match tokio::time::timeout(wait, connecting).await {
            Ok(Ok((socket, _))) => socket,
            Ok(Err(tungstenite::Error::Io(_))) => return Err(Refusal::Unreachable),
            Ok(Err(tungstenite::Error::Http(response)))
                if response.status() == StatusCode::NOT_FOUND =>
            {
                return Err(Refusal::Unreachable);
            }
            Ok(Err(_)) | Err(_) => return Err(Refusal::Declined),
        };
        let (mut sink, mut stream) = socket.split();
        let (outbox, mut outgoing) = mpsc::unbounded_channel::<Message>();
        tokio::spawn(async move {
            while let Some(message) = outgoing.recv().await {
                if sink.send(message).await.is_err() {
                    break;
                }
            }
        });

        let cdp = Arc::new(Self {
            outbox,
            shared: Arc::default(),
            next_id: AtomicU64::new(1),
            alive: Arc::new(AtomicBool::new(true)),
            dialog_opened: Arc::default(),
        });
        let reader = Arc::downgrade(&cdp);
        let (shared, alive) = (cdp.shared.clone(), cdp.alive.clone());
        tokio::spawn(async move {
            while let Some(Ok(message)) = stream.next().await {
                let Message::Text(text) = message else {
                    continue;
                };
                let Ok(value) = serde_json::from_str::<Value>(&text) else {
                    continue;
                };
                let Some(cdp) = reader.upgrade() else { break };
                cdp.receive(value);
            }
            alive.store(false, Ordering::Relaxed);
            for (_, tx) in shared.lock().unwrap().pending.drain() {
                let _ = tx.send(Err(CallError::Gone(
                    "the browser closed the connection".into(),
                )));
            }
        });
        Ok(cdp)
    }

    pub(super) fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    fn receive(&self, mut value: Value) {
        if let Some(id) = value.get("id").and_then(Value::as_u64) {
            let Some(tx) = self.shared.lock().unwrap().pending.remove(&id) else {
                return;
            };
            let answer = match value.get("error") {
                Some(e) => Err(CallError::Browser(
                    e["message"].as_str().unwrap_or("error").to_string(),
                )),
                None => Ok(value["result"].take()),
            };
            let _ = tx.send(answer);
            return;
        }
        let session = value["sessionId"].as_str().unwrap_or_default().to_string();
        match value["method"].as_str() {
            Some("Page.javascriptDialogOpening") => {
                let dialog = Dialog {
                    kind: value["params"]["type"].as_str().unwrap_or("alert").into(),
                    message: value["params"]["message"].as_str().unwrap_or("").into(),
                };
                // An alert or a leave-page check asks nothing of the agent: accepted, and told.
                if matches!(dialog.kind.as_str(), "alert" | "beforeunload") {
                    self.send(
                        "Page.handleJavaScriptDialog",
                        json!({ "accept": true }),
                        Some(&session),
                    );
                    let mut shared = self.shared.lock().unwrap();
                    shared.accepted.entry(session).or_default().push(dialog);
                } else {
                    self.shared.lock().unwrap().dialogs.insert(session, dialog);
                    self.dialog_opened.notify_waiters();
                }
            }
            Some("Page.javascriptDialogClosed") => {
                self.shared.lock().unwrap().dialogs.remove(&session);
            }
            Some("Target.attachedToTarget")
                if value["params"]["targetInfo"]["type"] == "iframe" =>
            {
                let params = &value["params"];
                let (Some(frame), Some(child)) = (
                    params["targetInfo"]["targetId"].as_str(),
                    params["sessionId"].as_str(),
                ) else {
                    return;
                };
                let mut shared = self.shared.lock().unwrap();
                shared
                    .frames
                    .entry(session)
                    .or_default()
                    .push((frame.to_string(), child.to_string()));
            }
            Some("Browser.downloadWillBegin") => {
                let params = &value["params"];
                let download = Download {
                    frame: params["frameId"].as_str().unwrap_or_default().into(),
                    name: params["suggestedFilename"]
                        .as_str()
                        .unwrap_or("download")
                        .into(),
                    state: "inProgress".into(),
                    ..Download::default()
                };
                let guid = params["guid"].as_str().unwrap_or_default().to_string();
                self.shared.lock().unwrap().downloads.push((guid, download));
            }
            Some("Browser.downloadProgress") => {
                let params = &value["params"];
                let guid = params["guid"].as_str().unwrap_or_default();
                let mut shared = self.shared.lock().unwrap();
                if let Some((_, d)) = shared.downloads.iter_mut().find(|(g, _)| g == guid) {
                    d.received = params["receivedBytes"].as_f64().unwrap_or_default() as u64;
                    d.total = params["totalBytes"].as_f64().unwrap_or_default() as u64;
                    d.state = params["state"].as_str().unwrap_or_default().into();
                    if let Some(path) = params["filePath"].as_str() {
                        d.path = Some(path.into());
                    }
                }
            }
            Some("Target.detachedFromTarget") => {
                let child = value["params"]["sessionId"].as_str().unwrap_or_default();
                for frames in self.shared.lock().unwrap().frames.values_mut() {
                    frames.retain(|(_, s)| s != child);
                }
            }
            _ => {}
        }
    }

    /// Sends a call nobody waits on, returning its id.
    fn send(&self, method: &str, params: Value, session: Option<&str>) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut message = json!({ "id": id, "method": method, "params": params });
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        let _ = self.outbox.send(Message::text(message.to_string()));
        id
    }

    /// Calls `method`, on the page of `session` or on the browser.
    pub(super) async fn call(
        &self,
        method: &str,
        params: Value,
        session: Option<&str>,
    ) -> Result<Value, CallError> {
        if !self.is_alive() {
            return Err(CallError::Gone("the browser connection is closed".into()));
        }
        let (tx, rx) = oneshot::channel();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.shared.lock().unwrap().pending.insert(id, tx);
        let mut message = json!({ "id": id, "method": method, "params": params });
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        let _ = self.outbox.send(Message::text(message.to_string()));

        let answer = async {
            match rx.await {
                Ok(answer) => answer,
                Err(_) => Err(CallError::Gone("the browser connection is closed".into())),
            }
        };
        // A dialog this page opens holds the call; one open before it, which a call may be
        // answering, or one on another page does not.
        let open_before = session.is_some_and(|s| self.dialog(s).is_some());
        let blocked = async {
            loop {
                // Registered before the check, so a dialog opening in between still wakes it.
                let opened = self.dialog_opened.notified();
                tokio::pin!(opened);
                opened.as_mut().enable();
                if !open_before && session.is_some_and(|s| self.dialog(s).is_some()) {
                    return;
                }
                opened.await;
            }
        };
        let result = tokio::select! {
            answer = tokio::time::timeout(CALL_TIMEOUT, answer) => answer
                .unwrap_or_else(|_| Err(CallError::Gone(format!("{method} took over {}s", CALL_TIMEOUT.as_secs())))),
            () = blocked => Err(CallError::Dialog),
        };
        if result.is_err() {
            self.shared.lock().unwrap().pending.remove(&id);
        }
        result
    }

    /// The dialog open on `session`'s page, if any.
    pub(super) fn dialog(&self, session: &str) -> Option<Dialog> {
        self.shared.lock().unwrap().dialogs.get(session).cloned()
    }

    /// The out-of-process frames of `session`'s page: frame id and session.
    pub(super) fn frames(&self, session: &str) -> Vec<(String, String)> {
        self.shared
            .lock()
            .unwrap()
            .frames
            .get(session)
            .cloned()
            .unwrap_or_default()
    }

    /// The downloads the browser has reported, in the order they began.
    pub(super) fn downloads(&self) -> Vec<(String, Download)> {
        self.shared.lock().unwrap().downloads.clone()
    }

    /// Alerts accepted on `session`'s page since this was last asked.
    pub(super) fn take_accepted(&self, session: &str) -> Vec<Dialog> {
        self.shared
            .lock()
            .unwrap()
            .accepted
            .remove(session)
            .unwrap_or_default()
    }
}
