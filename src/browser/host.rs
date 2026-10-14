//! The person's own Chrome, lent with their consent: [`Browser::host`](super::Browser::host).
//!
//! Chrome 144 and later lets a client in only once the person allows it in a prompt, and only
//! while remote debugging is on at `chrome://inspect/#remote-debugging`. A console does not reach
//! this host's loopback, so the agent's agent-browser dials a listener of this process's there
//! that a virtx [`Door`] opens to consoles. This process then speaks CDP to both sides over one
//! connection to Chrome, kept for the agent's life, so the prompt comes once however often the
//! agent's console stops.
//!
//! ```text
//! agent-browser ─► 127.0.0.1:D (door) ─► Proxy ─► Chrome
//! ```

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex as SyncMutex},
    time::Duration,
};

use anyhow::Context as _;
use futures::{SinkExt as _, StreamExt as _};
use serde_json::{Value, json};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Mutex, OnceCell, mpsc, oneshot},
    task::JoinHandle,
};
use tokio_tungstenite::tungstenite::{
    self, Message,
    handshake::server::{ErrorResponse, Request, Response},
    http::StatusCode,
    protocol::WebSocketConfig,
};
use virtx::console::{ConsoleClient, Door};

use super::{Ready, run, token};

/// How long the person has to answer Chrome's prompt.
const CONSENT_WAIT: Duration = Duration::from_secs(120);

pub(super) const DECLINED: &str = "The user declined to let you use their Chrome. Do not try the \
     browser again unless they ask; `ask_user` asks them again.";

fn not_enabled(dir: &Path) -> String {
    format!(
        "The user's Chrome is not open to you: remote debugging is off, Chrome is closed, or it is \
         older than 144 (looked for {}). Ask the user to open Chrome and turn on remote debugging \
         at chrome://inspect/#remote-debugging.",
        dir.join("DevToolsActivePort").display()
    )
}

/// The default Chrome's user data directory on this OS.
pub(super) fn default_user_data_dir() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        Some(dirs::data_dir()?.join("Google/Chrome"))
    } else if cfg!(target_os = "windows") {
        Some(dirs::data_local_dir()?.join("Google/Chrome/User Data"))
    } else {
        Some(dirs::config_dir()?.join("google-chrome"))
    }
}

pub(super) struct Host {
    user_data_dir: Option<PathBuf>,
    link: Mutex<Link>,
    entrance: OnceCell<Entrance>,
}

#[derive(Default)]
struct Link {
    proxy: Option<Arc<Proxy>>,
    declined: bool,
    /// Whether a connection was ever up, so the next one is a reconnection.
    was_up: bool,
    /// The tab opened for the agent, in a window of its own.
    agent_tab: Option<String>,
}

impl Host {
    pub(super) fn new(user_data_dir: Option<PathBuf>) -> Self {
        Self {
            user_data_dir,
            link: Mutex::default(),
            entrance: OnceCell::new(),
        }
    }

    /// Connects to the person's Chrome if not connected, asking them in Chrome's prompt, and
    /// points `agent`'s agent-browser at it. After a refusal it asks again only when `asking`.
    pub(super) async fn ensure(
        &self,
        agent: &mut ConsoleClient,
        asking: bool,
    ) -> anyhow::Result<Ready> {
        let mut link = self.link.lock().await;
        if link.proxy.as_ref().is_some_and(|p| p.is_alive()) {
            return Ok(Ready::Go(None));
        }
        if link.declined && !asking {
            return Ok(Ready::Refused(DECLINED.into()));
        }
        let dir = self
            .user_data_dir
            .clone()
            .or_else(default_user_data_dir)
            .context("no Chrome user data directory on this OS: pass one to `Browser::host`")?;
        let Some(url) = devtools_url(&dir) else {
            return Ok(Ready::Refused(not_enabled(&dir)));
        };
        let proxy = match Proxy::connect(&url).await {
            Ok(proxy) => Arc::new(proxy),
            Err(Refusal::Unreachable) => return Ok(Ready::Refused(not_enabled(&dir))),
            Err(Refusal::Declined) => {
                link.declined = true;
                return Ok(Ready::Refused(DECLINED.into()));
            }
        };
        link.declined = false;

        let entrance = self.entrance.get_or_try_init(open_entrance).await?;
        *entrance.proxy.lock().unwrap() = Some(proxy.clone());

        let tab = match link.agent_tab.clone() {
            Some(tab) if proxy.has_page(&tab).await => tab,
            _ => {
                let created = proxy
                    .call(
                        "Target.createTarget",
                        json!({ "url": "about:blank", "newWindow": true }),
                    )
                    .await
                    .map_err(|e| anyhow::anyhow!("opening the agent's window: {e}"))?;
                created["targetId"]
                    .as_str()
                    .context("Chrome opened no tab")?
                    .to_string()
            }
        };
        point_agent(agent, entrance, &tab).await?;

        let reconnected = std::mem::replace(&mut link.was_up, true);
        let moved = link.agent_tab.as_deref().is_some_and(|old| old != tab);
        link.agent_tab = Some(tab);
        link.proxy = Some(proxy);
        Ok(Ready::Go(reconnected.then(|| {
            let window = if moved {
                " Your window was gone, so you are on a new blank tab: look before you act."
            } else {
                ""
            };
            format!("[browser] The connection to the user's Chrome dropped and is back.{window}")
        })))
    }
}

/// Chrome's browser endpoint from `DevToolsActivePort`: its port, then its path.
fn devtools_url(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("DevToolsActivePort")).ok()?;
    let mut lines = text.lines();
    let port: u16 = lines.next()?.trim().parse().ok()?;
    let path = lines.next()?.trim();
    path.starts_with('/')
        .then(|| format!("ws://127.0.0.1:{port}{path}"))
}

/// Binds agent-browser to `tab`, pinned, through the entrance; the secret path lets it in.
async fn point_agent(
    agent: &mut ConsoleClient,
    entrance: &Entrance,
    tab: &str,
) -> anyhow::Result<()> {
    let cdp = format!(
        "ws://127.0.0.1:{}/devtools/browser/{}",
        entrance.port, entrance.secret
    );
    let config = json!({ "cdp": cdp, "pinTab": true });
    let binding = json!({ "targetId": tab, "url": "about:blank", "pinned": true });
    run(
        agent,
        &format!(
            "mkdir -p \"$HOME/.agent-browser\" \
             && printf '%s' '{config}' > \"$HOME/.agent-browser/config.json\" \
             && printf '%s' '{binding}' > \"$HOME/.agent-browser/default.target\""
        ),
    )
    .await
}

// ---- the entrance ---------------------------------------------------------------------------

/// Where agent-browser dials: a listener on this host's loopback, which the door opens to
/// consoles. Every console reaches it, so the secret path is what lets the agent's in.
struct Entrance {
    _door: Door,
    port: u16,
    secret: String,
    /// Where an agent is served: the connection to Chrome now.
    proxy: Arc<SyncMutex<Option<Arc<Proxy>>>>,
    accepting: JoinHandle<()>,
}

impl Drop for Entrance {
    fn drop(&mut self) {
        self.accepting.abort();
    }
}

async fn open_entrance() -> anyhow::Result<Entrance> {
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    let door = Door::open(port).context("opening the browser's door to consoles")?;
    let secret = token();
    let proxy = Arc::new(SyncMutex::new(None));
    let accepting = tokio::spawn(accept(listener, secret.clone(), proxy.clone()));
    Ok(Entrance {
        _door: door,
        port,
        secret,
        proxy,
        accepting,
    })
}

async fn accept(listener: TcpListener, secret: String, proxy: Arc<SyncMutex<Option<Arc<Proxy>>>>) {
    while let Ok((stream, _)) = listener.accept().await {
        let (secret, proxy) = (secret.clone(), proxy.clone());
        tokio::spawn(async move { serve_agent(stream, &secret, &proxy).await });
    }
}

fn ws_config() -> WebSocketConfig {
    // A full-page screenshot is one message, and can be large.
    WebSocketConfig::default()
        .max_message_size(None)
        .max_frame_size(None)
}

/// An agent on `stream`: a WebSocket on the secret path, then CDP through the proxy.
// The handshake's refusal is tungstenite's type, large or not.
#[allow(clippy::result_large_err)]
async fn serve_agent(stream: TcpStream, secret: &str, proxy: &SyncMutex<Option<Arc<Proxy>>>) {
    let expected = format!("/devtools/browser/{secret}");
    let check = |request: &Request, response: Response| -> Result<Response, ErrorResponse> {
        if request.uri().path() == expected {
            Ok(response)
        } else {
            let mut refusal = ErrorResponse::new(None);
            *refusal.status_mut() = StatusCode::NOT_FOUND;
            Err(refusal)
        }
    };
    let Ok(socket) =
        tokio_tungstenite::accept_hdr_async_with_config(stream, check, Some(ws_config())).await
    else {
        return;
    };
    let proxy = proxy.lock().unwrap().clone();
    if let Some(proxy) = proxy {
        proxy.serve(socket).await;
    }
}

// ---- one connection to Chrome ---------------------------------------------------------------

enum Refusal {
    /// Nothing listens, or Chrome turned the connection away without asking.
    Unreachable,
    /// The person said no, or did not answer.
    Declined,
}

enum Pending {
    /// A call of this process's own; `None` when nobody waits for the answer.
    Own(Option<oneshot::Sender<Result<Value, String>>>),
    /// The agent's, by the id it sent, from the agent connection numbered `generation`.
    Agent { generation: u64, id: Value },
}

/// The agent connection being served: one at a time, the newest.
struct Agent {
    generation: u64,
    tx: mpsc::UnboundedSender<Message>,
    /// Sessions it attached, detached when it goes so the next one starts clean.
    sessions: HashSet<String>,
}

#[derive(Default)]
struct Shared {
    alive: bool,
    next_id: u64,
    next_generation: u64,
    pending: HashMap<u64, Pending>,
    agent: Option<Agent>,
}

pub(super) struct Proxy {
    to_chrome: mpsc::UnboundedSender<Message>,
    shared: Arc<SyncMutex<Shared>>,
}

impl Proxy {
    /// Connects to Chrome at `url`, which is when Chrome asks the person.
    async fn connect(url: &str) -> Result<Self, Refusal> {
        let connecting = tokio_tungstenite::connect_async_with_config(url, Some(ws_config()), true);
        let socket = match tokio::time::timeout(CONSENT_WAIT, connecting).await {
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
        let (to_chrome, mut outbox) = mpsc::unbounded_channel::<Message>();
        tokio::spawn(async move {
            while let Some(message) = outbox.recv().await {
                if sink.send(message).await.is_err() {
                    break;
                }
            }
        });
        let shared = Arc::new(SyncMutex::new(Shared {
            alive: true,
            next_id: 1,
            ..Shared::default()
        }));
        let reading = shared.clone();
        tokio::spawn(async move {
            while let Some(Ok(message)) = stream.next().await {
                if let Message::Text(text) = message
                    && let Ok(value) = serde_json::from_str::<Value>(&text)
                {
                    from_chrome(&reading, value);
                }
            }
            let mut shared = reading.lock().unwrap();
            shared.alive = false;
            // Dropping the agent's sender closes its connection; its next one finds us gone.
            shared.agent = None;
            for (_, pending) in shared.pending.drain() {
                if let Pending::Own(Some(tx)) = pending {
                    let _ = tx.send(Err("Chrome closed the connection".into()));
                }
            }
        });
        Ok(Self { to_chrome, shared })
    }

    fn is_alive(&self) -> bool {
        self.shared.lock().unwrap().alive
    }

    /// Sends `method` to the browser for this process, answered or not.
    fn send_own(&self, method: &str, params: Value, session: Option<&str>) -> u64 {
        let id = {
            let mut shared = self.shared.lock().unwrap();
            let id = shared.next_id;
            shared.next_id += 1;
            shared.pending.insert(id, Pending::Own(None));
            id
        };
        let mut message = json!({ "id": id, "method": method, "params": params });
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        let _ = self.to_chrome.send(Message::text(message.to_string()));
        id
    }

    async fn call(&self, method: &str, params: Value) -> Result<Value, String> {
        let (tx, rx) = oneshot::channel();
        let id = self.send_own(method, params, None);
        if let Some(pending) = self.shared.lock().unwrap().pending.get_mut(&id) {
            *pending = Pending::Own(Some(tx));
        }
        match tokio::time::timeout(Duration::from_secs(30), rx).await {
            Ok(Ok(result)) => result.map_err(|e| format!("{method}: {e}")),
            Ok(Err(_)) => Err(format!("{method}: the connection closed")),
            Err(_) => Err(format!("{method}: no answer in 30s")),
        }
    }

    async fn has_page(&self, target: &str) -> bool {
        let Ok(targets) = self.call("Target.getTargets", json!({})).await else {
            return false;
        };
        targets["targetInfos"]
            .as_array()
            .is_some_and(|all| all.iter().any(|t| t["targetId"] == target))
    }

    /// Serves an agent connection until it ends, replacing whichever came before it.
    async fn serve(&self, socket: tokio_tungstenite::WebSocketStream<TcpStream>) {
        let (mut sink, mut stream) = socket.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
        let generation = {
            let mut shared = self.shared.lock().unwrap();
            if !shared.alive {
                return;
            }
            shared.next_generation += 1;
            let generation = shared.next_generation;
            let before = shared.agent.replace(Agent {
                generation,
                tx,
                sessions: HashSet::new(),
            });
            drop(shared);
            if let Some(before) = before {
                self.forget(before);
            }
            generation
        };
        let writer = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                if sink.send(message).await.is_err() {
                    break;
                }
            }
            let _ = sink.close().await;
        });
        while let Some(Ok(message)) = stream.next().await {
            let Message::Text(text) = message else {
                continue;
            };
            let Ok(mut value) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            let id = {
                let mut shared = self.shared.lock().unwrap();
                if shared.agent.as_ref().map(|a| a.generation) != Some(generation) {
                    break;
                }
                let id = shared.next_id;
                shared.next_id += 1;
                shared.pending.insert(
                    id,
                    Pending::Agent {
                        generation,
                        id: value["id"].take(),
                    },
                );
                id
            };
            value["id"] = json!(id);
            let _ = self.to_chrome.send(Message::text(value.to_string()));
        }
        let gone = {
            let mut shared = self.shared.lock().unwrap();
            match &shared.agent {
                Some(agent) if agent.generation == generation => shared.agent.take(),
                _ => None,
            }
        };
        if let Some(gone) = gone {
            self.forget(gone);
        }
        writer.abort();
    }

    /// Undoes what an agent connection left behind: its sessions, discovery and auto-attach,
    /// and the answers still owed to it.
    fn forget(&self, agent: Agent) {
        self.shared.lock().unwrap().pending.retain(|_, pending| {
            !matches!(pending, Pending::Agent { generation, .. } if *generation == agent.generation)
        });
        for session in &agent.sessions {
            self.send_own(
                "Target.detachFromTarget",
                json!({ "sessionId": session }),
                None,
            );
        }
        self.send_own(
            "Target.setDiscoverTargets",
            json!({ "discover": false }),
            None,
        );
        self.send_own(
            "Target.setAutoAttach",
            json!({ "autoAttach": false, "waitForDebuggerOnStart": false }),
            None,
        );
    }
}

/// A message from Chrome: an answer goes to whoever asked, an event to the agent.
fn from_chrome(shared: &SyncMutex<Shared>, mut value: Value) {
    let mut shared = shared.lock().unwrap();
    if let Some(id) = value.get("id").and_then(Value::as_u64) {
        match shared.pending.remove(&id) {
            Some(Pending::Own(Some(tx))) => {
                let result = match value.get("error") {
                    Some(e) => Err(e["message"].as_str().unwrap_or("error").to_string()),
                    None => Ok(value["result"].take()),
                };
                let _ = tx.send(result);
            }
            Some(Pending::Agent { generation, id }) => {
                let Some(agent) = shared.agent.as_mut().filter(|a| a.generation == generation)
                else {
                    return;
                };
                if let Some(session) = value["result"]["sessionId"].as_str() {
                    agent.sessions.insert(session.to_string());
                }
                value["id"] = id;
                let _ = agent.tx.send(Message::text(value.to_string()));
            }
            Some(Pending::Own(None)) | None => {}
        }
        return;
    }
    let Some(agent) = shared.agent.as_mut() else {
        return;
    };
    // An earlier connection's session, still winding down: not this agent's to hear.
    if let Some(session) = value["sessionId"].as_str()
        && !agent.sessions.contains(session)
    {
        return;
    }
    // Sessions Chrome attached for it, through auto-attach, count as its own.
    match value["method"].as_str() {
        Some("Target.attachedToTarget") => {
            if let Some(session) = value["params"]["sessionId"].as_str() {
                agent.sessions.insert(session.to_string());
            }
        }
        Some("Target.detachedFromTarget") => {
            let session = value["params"]["sessionId"].as_str().unwrap_or_default();
            if !agent.sessions.remove(session) {
                return;
            }
        }
        _ => {}
    }
    let _ = agent.tx.send(Message::text(value.to_string()));
}

#[cfg(test)]
mod tests {
    use virtx::image::Recipe;

    use super::*;

    /// A headless Chrome on its own profile, which asks nobody: `AILOY_TEST_CHROME` names the
    /// binary, macOS's Chrome by default.
    struct TestChrome {
        child: std::process::Child,
        dir: tempfile::TempDir,
    }

    impl TestChrome {
        fn start(dir: Option<tempfile::TempDir>) -> Self {
            let binary = std::env::var("AILOY_TEST_CHROME").unwrap_or_else(|_| {
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into()
            });
            let dir = dir.unwrap_or_else(|| tempfile::tempdir().unwrap());
            let _ = std::fs::remove_file(dir.path().join("DevToolsActivePort"));
            let child = std::process::Command::new(binary)
                .args([
                    "--headless=new",
                    "--remote-debugging-port=0",
                    "--no-first-run",
                ])
                .arg(format!("--user-data-dir={}", dir.path().display()))
                .arg("about:blank")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let chrome = Self { child, dir };
            for _ in 0..100 {
                if devtools_url(chrome.dir.path()).is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            chrome
        }

        fn stop(mut self) -> tempfile::TempDir {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.dir
        }
    }

    async fn sh(console: &mut ConsoleClient, cmd: &str) -> (i32, String) {
        let out = console.exec(["sh", "-c", cmd], Some(60_000)).await.unwrap();
        let text = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        (out.code, text)
    }

    async fn pages(chrome: &TestChrome) -> Vec<String> {
        let url = devtools_url(chrome.dir.path()).unwrap();
        let port = url.split(':').nth(2).unwrap().split('/').next().unwrap();
        let list: Value = reqwest::get(format!("http://127.0.0.1:{port}/json/list"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        list.as_array()
            .unwrap()
            .iter()
            .filter(|t| t["type"] == "page")
            .map(|t| t["url"].as_str().unwrap().to_string())
            .collect()
    }

    /// Lends a Chrome for real: needs a virtx whose `/abin` carries `agent-browser`
    /// (`VIRTX_HOME` naming one), the network, and Chrome.
    #[tokio::test]
    #[ignore = "needs a virtx with agent-browser in /abin, and Chrome"]
    async fn lends_a_chrome() {
        let mut agent = ConsoleClient::builder()
            .image(Recipe::new("debian:trixie-slim"))
            .network(true)
            .build()
            .await
            .unwrap();

        // Nothing to connect to yet: refused, with the way to fix it, and nothing started.
        let nowhere = tempfile::tempdir().unwrap();
        let host = Host::new(Some(nowhere.path().to_path_buf()));
        let Ready::Refused(why) = host.ensure(&mut agent, false).await.unwrap() else {
            panic!("connected to nothing");
        };
        assert!(why.contains("chrome://inspect/#remote-debugging"), "{why}");
        assert!(host.entrance.get().is_none());

        let chrome = TestChrome::start(None);
        let host = Host::new(Some(chrome.dir.path().to_path_buf()));
        let Ready::Go(None) = host.ensure(&mut agent, false).await.unwrap() else {
            panic!("not a first connection");
        };
        // A window of the agent's own beside the one Chrome started with.
        assert_eq!(pages(&chrome).await.len(), 2);

        let (code, out) = sh(
            &mut agent,
            "agent-browser open https://example.com && agent-browser get title",
        )
        .await;
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("Example Domain"), "{out}");
        // On its own tab: the person's stays where it was.
        let open = pages(&chrome).await;
        assert_eq!(open.len(), 2, "{open:?}");
        assert!(open.iter().any(|u| u == "about:blank"), "{open:?}");

        // The console stopping between batches takes agent-browser's connection with it; the
        // next one goes through the same connection to Chrome, onto the same tab.
        agent.stop().await.unwrap();
        let Ready::Go(None) = host.ensure(&mut agent, false).await.unwrap() else {
            panic!("reconnected when nothing dropped");
        };
        let (code, out) = sh(
            &mut agent,
            "agent-browser snapshot -i && agent-browser get url",
        )
        .await;
        assert_eq!(code, 0, "{out}");
        assert!(
            out.contains("Learn more") && out.contains("https://example.com/"),
            "{out}"
        );
        assert_eq!(pages(&chrome).await.len(), 2);

        // Chrome going away and coming back is a reconnection, onto a new window.
        let dir = chrome.stop();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let chrome = TestChrome::start(Some(dir));
        let Ready::Go(Some(note)) = host.ensure(&mut agent, false).await.unwrap() else {
            panic!("no word of the reconnection");
        };
        assert!(note.contains("new blank tab"), "{note}");
        let (code, out) = sh(
            &mut agent,
            "agent-browser open https://example.org && agent-browser get title",
        )
        .await;
        assert_eq!(code, 0, "{out}");
        assert!(out.contains("Example Domain"), "{out}");
        chrome.stop();
    }

    /// [`BrowserSession::start`](super::super::BrowserSession::start), the app's: it asks again
    /// after a decline and fails with why, and a reconnect it makes is told on the agent's next
    /// turn. Needs what [`lends_a_chrome`] needs.
    #[tokio::test]
    #[ignore = "needs a virtx with agent-browser in /abin, and Chrome"]
    async fn the_apps_start_asks_again_and_passes_a_reconnect_on() {
        use std::{
            io::{Read as _, Write as _},
            sync::atomic::{AtomicUsize, Ordering::SeqCst},
        };

        use super::super::{Browser, BrowserSession};

        let mut agent = ConsoleClient::builder()
            .image(Recipe::new("debian:trixie-slim"))
            .network(true)
            .build()
            .await
            .unwrap();

        // Answers as Chrome does when the person declines.
        let rejecting = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = rejecting.local_addr().unwrap().port();
        let tries = Arc::new(AtomicUsize::new(0));
        let counted = tries.clone();
        std::thread::spawn(move || {
            for mut stream in rejecting.incoming().flatten() {
                counted.fetch_add(1, SeqCst);
                let _ = stream.read(&mut [0; 4096]);
                let _ = stream.write_all(
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 19\r\n\r\nConnection rejected",
                );
            }
        });
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("DevToolsActivePort"),
            format!("{port}\n/devtools/browser/x\n"),
        )
        .unwrap();
        let session = BrowserSession::new(Browser::host().user_data_dir(dir.path()));
        let Ready::Refused(why) = session.ensure(&mut agent).await.unwrap() else {
            panic!("connected through a decline");
        };
        assert_eq!(why, DECLINED);
        // The agent does not ask again; the app does.
        assert!(matches!(
            session.ensure(&mut agent).await.unwrap(),
            Ready::Refused(_)
        ));
        assert_eq!(tries.load(SeqCst), 1);
        let err = session.start(&mut agent).await.unwrap_err();
        assert_eq!(err.to_string(), DECLINED);
        assert_eq!(tries.load(SeqCst), 2);

        let chrome = TestChrome::start(None);
        let session = BrowserSession::new(Browser::host().user_data_dir(chrome.dir.path()));
        assert_eq!(session.start(&mut agent).await.unwrap(), None);
        assert_eq!(session.start(&mut agent).await.unwrap(), None);
        assert_eq!(pages(&chrome).await.len(), 2);
        assert_eq!(session.catch_up().await.unwrap(), None);

        let dir = chrome.stop();
        tokio::time::sleep(Duration::from_millis(500)).await;
        let chrome = TestChrome::start(Some(dir));
        assert_eq!(session.start(&mut agent).await.unwrap(), None);
        let note = session
            .catch_up()
            .await
            .unwrap()
            .expect("no word of the reconnection");
        assert!(note.contains("new blank tab"), "{note}");
        assert_eq!(session.catch_up().await.unwrap(), None);
        chrome.stop();
    }

    #[test]
    fn devtools_active_port_is_a_port_then_a_path() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(devtools_url(dir.path()), None);
        std::fs::write(
            dir.path().join("DevToolsActivePort"),
            "9555\n/devtools/browser/abc\n",
        )
        .unwrap();
        assert_eq!(
            devtools_url(dir.path()).as_deref(),
            Some("ws://127.0.0.1:9555/devtools/browser/abc")
        );
        std::fs::write(dir.path().join("DevToolsActivePort"), "x\n/y\n").unwrap();
        assert_eq!(devtools_url(dir.path()), None);
    }

    /// `ask_user` on a host browser with `Wait`: the app's handler answers. Needs what
    /// [`lends_a_chrome`] needs.
    #[tokio::test]
    #[ignore = "needs a virtx with agent-browser in /abin, and Chrome"]
    async fn the_apps_handler_answers_ask_user() {
        use super::super::{AskResult, AskUser, Browser, BrowserSession};

        let mut agent = ConsoleClient::builder()
            .image(Recipe::new("debian:trixie-slim"))
            .network(true)
            .build()
            .await
            .unwrap();
        let chrome = TestChrome::start(None);
        let asked = Arc::new(SyncMutex::new(Vec::<String>::new()));
        let heard = asked.clone();
        let session = BrowserSession::new(
            Browser::host()
                .user_data_dir(chrome.dir.path())
                .ask_user(AskUser::Wait)
                .on_ask(move |message| {
                    heard.lock().unwrap().push(message.clone());
                    async move {
                        if message.contains("pay") {
                            AskResult::Declined
                        } else {
                            AskResult::Note("signed in as kim".into())
                        }
                    }
                }),
        );
        let answer = session.ask(&mut agent, "Sign in, please").await;
        assert_eq!(
            answer.pointer("/note").and_then(|v| v.as_str()),
            Some("signed in as kim")
        );
        let answer = session.ask(&mut agent, "May I pay 10,000 won?").await;
        assert_eq!(
            answer.pointer("/declined").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(asked.lock().unwrap().len(), 2);
        assert_eq!(session.url(), None);
        chrome.stop();
    }
}
