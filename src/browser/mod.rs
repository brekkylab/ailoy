//! A browser the agent drives with `/abin/agent-browser` from its console, over CDP.
//!
//! [`Browser::sandboxed`] is Chromium in a console of its own, which a person watches and uses
//! from their browser through a viewer. [`Browser::host`] is the person's own Chrome, lent with
//! their consent. Either way the browser outlives the agent's console stopping between tool
//! batches, nothing starts until the agent first runs `agent-browser` or asks the person
//! something, and what ailoy started ends with the agent.

mod host;
mod sandboxed;

use std::{
    fmt, future::Future, net::TcpListener, path::PathBuf, pin::Pin, sync::Arc, time::Duration,
};

use anyhow::bail;
use virtx::console::ConsoleClient;

use crate::tool::{ToolDesc, ToolDescBuilder, ToolFunc};

/// How `ask_user` waits for the person.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AskUser {
    /// Show the request and end the turn; the next run picks up from whatever the person did.
    #[default]
    EndTurn,
    /// Wait in the tool call until the person answers: Done in the viewer, or what
    /// [`Browser::on_ask`] returns.
    Wait,
}

/// How the person answered a request, from an [`on_ask`](Browser::on_ask) handler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AskResult {
    Done,
    Declined,
    /// Done, with something to tell the agent.
    Note(String),
}

/// Asks the person `message`, however the app does that, and returns their answer.
pub type AskHandler =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = AskResult> + Send>> + Send + Sync>;

#[derive(Clone, Debug, Default)]
enum Mode {
    #[default]
    Sandboxed,
    Host {
        user_data_dir: Option<PathBuf>,
    },
}

/// A browser for an agent, set with [`AgentBuilder::browser`](crate::agent::AgentBuilder::browser).
#[derive(Clone, Default)]
pub struct Browser {
    mode: Mode,
    ask_user: AskUser,
    /// A BCP 47 tag; this host's language when unset. Sandboxed only.
    locale: Option<String>,
    on_ask: Option<AskHandler>,
    ask_timeout: Option<Duration>,
}

impl fmt::Debug for Browser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Browser")
            .field("mode", &self.mode)
            .field("ask_user", &self.ask_user)
            .field("locale", &self.locale)
            .field("on_ask", &self.on_ask.is_some())
            .field("ask_timeout", &self.ask_timeout)
            .finish()
    }
}

impl Browser {
    /// Chromium in a console of its own, which the person sees and uses through the viewer.
    pub fn sandboxed() -> Self {
        Self::default()
    }

    /// The person's own Chrome (144 or later, with remote debugging on at
    /// `chrome://inspect/#remote-debugging`), which asks them before it lets the agent in. The
    /// agent works in a window of its own but is not kept from the person's other tabs, and
    /// is signed in wherever they are.
    pub fn host() -> Self {
        Self {
            mode: Mode::Host {
                user_data_dir: None,
            },
            ..Self::default()
        }
    }

    /// For [`host`](Self::host): the Chrome whose user data directory this is, for Beta,
    /// Canary or Chromium. The default is this OS's Chrome.
    pub fn user_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        if let Mode::Host { user_data_dir } = &mut self.mode {
            *user_data_dir = Some(dir.into());
        }
        self
    }

    pub fn ask_user(mut self, mode: AskUser) -> Self {
        self.ask_user = mode;
        self
    }

    /// The language pages are asked in and the browser speaks, such as `"ko-KR"`. This host's
    /// language by default, and `en-US` when that is unknown. A [`host`](Self::host) browser
    /// keeps its own and ignores this.
    pub fn locale(mut self, locale: impl Into<String>) -> Self {
        self.locale = Some(locale.into());
        self
    }

    /// How [`AskUser::Wait`] asks: `handler` gets the request and returns the answer. A
    /// [`host`](Self::host) browser has no viewer to press Done in, so it needs one.
    pub fn on_ask<F, Fut>(mut self, handler: F) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = AskResult> + Send + 'static,
    {
        self.on_ask = Some(Arc::new(move |message| Box::pin(handler(message))));
        self
    }

    /// How long [`on_ask`](Self::on_ask) may take before the agent hears it timed out; no limit
    /// by default.
    pub fn ask_timeout(mut self, timeout: Duration) -> Self {
        self.ask_timeout = Some(timeout);
        self
    }

    fn is_host(&self) -> bool {
        matches!(self.mode, Mode::Host { .. })
    }

    /// Refuses what cannot work, before anything starts.
    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.is_host() && self.ask_user == AskUser::Wait && self.on_ask.is_none() {
            bail!(
                "`Browser::host` with `AskUser::Wait` needs `Browser::on_ask`: there is no viewer \
                 to press Done in"
            );
        }
        Ok(())
    }
}

/// Whether a browser command may go ahead.
pub(crate) enum Ready {
    /// Go ahead, telling the agent this alongside the command's result, if anything.
    Go(Option<String>),
    /// Do not; this is the command's result instead.
    Refused(String),
}

enum Backend {
    Sandboxed(sandboxed::Sandboxed),
    Host(host::Host),
}

/// An agent's browser, from [`Agent::browser`](crate::agent::Agent::browser): started on first
/// use, and ended when the last handle drops. Its [`url`](Self::url) can be read while the
/// agent is in the middle of a turn.
pub struct BrowserSession {
    config: Browser,
    backend: Backend,
    /// What [`Self::start`] heard for the agent, told on its next turn.
    unheard: std::sync::Mutex<Option<String>>,
}

impl BrowserSession {
    pub(crate) fn new(config: Browser) -> Arc<Self> {
        let backend = match &config.mode {
            Mode::Sandboxed => Backend::Sandboxed(sandboxed::Sandboxed::default()),
            Mode::Host { user_data_dir } => Backend::Host(host::Host::new(user_data_dir.clone())),
        };
        Arc::new(Self {
            config,
            backend,
            unheard: Default::default(),
        })
    }

    pub(crate) fn config(&self) -> &Browser {
        &self.config
    }

    /// Where a person watches and uses a [sandboxed](Browser::sandboxed) browser, token
    /// included, once it has booted. `None` for a host one: the person has it in front of them.
    pub fn url(&self) -> Option<String> {
        match &self.backend {
            Backend::Sandboxed(sandboxed) => sandboxed.url(),
            Backend::Host(_) => None,
        }
    }

    /// Starts the browser, or connects to it, if that has not happened, and points `agent`'s
    /// agent-browser at it.
    pub(crate) async fn ensure(&self, agent: &mut ConsoleClient) -> anyhow::Result<Ready> {
        self.ensure_for(agent, false).await
    }

    /// [`Agent::start_browser`](crate::agent::Agent::start_browser): as [`Self::ensure`], but a
    /// refusal is the caller's error, and one after a decline asks again.
    pub(crate) async fn start(&self, agent: &mut ConsoleClient) -> anyhow::Result<Option<String>> {
        match self.ensure_for(agent, true).await? {
            Ready::Go(said) => {
                if said.is_some() {
                    *self.unheard.lock().unwrap() = said;
                }
                Ok(self.url())
            }
            Ready::Refused(why) => bail!(why),
        }
    }

    /// As [`Self::ensure`]; `asking` is `ask_user`'s, which asks again after a refusal.
    async fn ensure_for(&self, agent: &mut ConsoleClient, asking: bool) -> anyhow::Result<Ready> {
        match &self.backend {
            Backend::Sandboxed(sandboxed) => {
                sandboxed
                    .ensure(agent, self.config.locale.as_deref())
                    .await?;
                Ok(Ready::Go(None))
            }
            Backend::Host(host) => host.ensure(agent, asking).await,
        }
    }

    /// What the person did in the viewer since the agent last heard, as one note for its next
    /// message; and, for [`AskUser::EndTurn`], the request it was answering taken down. A host
    /// browser tells only a reconnect [`Self::start`] made: the agent reads its pages again.
    pub(crate) async fn catch_up(&self) -> anyhow::Result<Option<String>> {
        match &self.backend {
            Backend::Sandboxed(sandboxed) => {
                sandboxed
                    .catch_up(self.config.ask_user == AskUser::EndTurn)
                    .await
            }
            Backend::Host(_) => Ok(self.unheard.lock().unwrap().take()),
        }
    }

    /// `ask_user`: the tool result for asking the person `message`.
    async fn ask(&self, agent: &mut ConsoleClient, message: &str) -> crate::datatype::Value {
        match self.ensure_for(agent, true).await {
            Ok(Ready::Go(_)) => {}
            Ok(Ready::Refused(why)) => {
                return crate::to_value!({ "error": why, "phase": "browser" });
            }
            Err(e) => return crate::to_value!({ "error": format!("{e:#}"), "phase": "browser" }),
        }
        let sandboxed = match &self.backend {
            Backend::Sandboxed(sandboxed) => Some(sandboxed),
            Backend::Host(_) => None,
        };
        match (self.config.ask_user, &self.config.on_ask, sandboxed) {
            (AskUser::Wait, Some(handler), _) => {
                // Shown in the viewer too, without a Done: the app answers.
                if let Some(sandboxed) = sandboxed {
                    let _ = sandboxed.request(message, false).await;
                }
                let asked = handler(message.to_string());
                let answer = match self.config.ask_timeout {
                    Some(limit) => tokio::time::timeout(limit, asked).await.ok(),
                    None => Some(asked.await),
                };
                if let Some(sandboxed) = sandboxed {
                    let _ = sandboxed.clear().await;
                }
                match answer {
                    Some(AskResult::Done) => crate::to_value!({ "done": true }),
                    Some(AskResult::Declined) => crate::to_value!({ "declined": true }),
                    Some(AskResult::Note(note)) => crate::to_value!({ "done": true, "note": note }),
                    None => crate::to_value!({ "timeout": true }),
                }
            }
            (mode, _, Some(sandboxed)) => {
                let wait = mode == AskUser::Wait;
                match sandboxed.request(message, wait).await {
                    Ok(()) if wait => crate::to_value!({ "done": true }),
                    Ok(()) => crate::to_value!({
                        "shown": true,
                        "next": "The request is in the user's viewer. End your turn now; their next message is when it is done."
                    }),
                    Err(e) => crate::to_value!({ "error": format!("{e:#}"), "phase": "viewer" }),
                }
            }
            // Host and EndTurn (`check` refuses host and Wait without a handler).
            _ => crate::to_value!({
                "next": "Nothing is shown to the user: tell them in your reply exactly what to do in their Chrome, then end your turn. Their next message is when it is done."
            }),
        }
    }
}

async fn run(console: &mut ConsoleClient, cmd: &str) -> anyhow::Result<()> {
    let out = console.exec(["sh", "-c", cmd], Some(60_000)).await?;
    if out.code != 0 {
        bail!(
            "`{cmd}` exited {}: {}",
            out.code,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

async fn require(console: &mut ConsoleClient, path: &str) -> anyhow::Result<()> {
    let out = console.exec(["test", "-x", path], Some(10_000)).await?;
    if out.code != 0 {
        bail!(
            "this virtx has no {path}: the browser needs a console server whose /abin carries it"
        );
    }
    Ok(())
}

async fn wait_for(http: &reqwest::Client, url: &str) -> anyhow::Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let answer = http.get(url).timeout(Duration::from_secs(2)).send().await;
        if answer.is_ok_and(|a| a.status().is_success()) {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            bail!("{url} did not answer in 60s");
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
    }
}

/// A port this host is not using now. Another process may take it before the console does;
/// the console then fails to boot and says so.
fn free_port() -> anyhow::Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

/// 122 random bits from the OS, which is what a v4 UUID is.
fn token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// What the agent is told about its browser, beside its instruction.
pub(crate) fn instruction(browser: &Browser) -> String {
    let asking = match (browser.ask_user, browser.is_host()) {
        (AskUser::EndTurn, false) => {
            "call `ask_user` saying exactly what to do in the viewer, then end your turn: \
             their next message is when it is done"
        }
        (AskUser::EndTurn, true) => {
            "call `ask_user`, then tell them exactly what to do in their Chrome and end your \
             turn: their next message is when it is done"
        }
        (AskUser::Wait, _) => {
            "call `ask_user` saying exactly what to do; it returns once they have answered"
        }
    };
    let whose = if browser.is_host() {
        "This is the user's own Chrome, signed in wherever they are. It is already connected \
         and you have a window of your own: work only in your own tabs, and never read or touch \
         the user's other tabs. Before anything that cannot be undone (paying, sending, \
         deleting, posting), ask the user with `ask_user`. Never launch another browser."
    } else {
        "It is already connected and on a tab of its own; never launch another browser. The \
         user watches this browser live and can use it too."
    };
    format!(
        "\n\n# Browser\n\n\
         A web browser is attached to your machine: drive it with the `agent-browser` CLI \
         through your shell. {whose} Use its default session, which your tab is bound to: \
         never pass `--session` or set `AGENT_BROWSER_SESSION`. Run `agent-browser skills get \
         core` before your first browser command.\n\n\
         Read pages as text, not images: `agent-browser snapshot -i` lists the interactive \
         elements with refs, which `click @e3` and `fill @e5 \"...\"` act on; take contents \
         with `get text @e1` or `eval`. A screenshot costs far more: take one only when the \
         layout itself matters, and then prefer `screenshot --annotate`.\n\n\
         When a site needs something only the user can do (signing in, a CAPTCHA, a second \
         factor, a consent), never ask for credentials and never get around it: {asking}. \
         Look at the page again afterwards. Never close or navigate a tab you did not open."
    )
}

pub(crate) fn ask_user_tool_desc(browser: &Browser) -> ToolDesc {
    let (place, returns) = match (
        browser.ask_user,
        browser.is_host(),
        browser.on_ask.is_some(),
    ) {
        (AskUser::EndTurn, false, _) => (
            "in the browser viewer",
            "It shows the request and returns at once: end your turn after it.",
        ),
        (AskUser::EndTurn, true, _) => (
            "in their Chrome",
            "It returns at once: then tell them what to do and end your turn.",
        ),
        (AskUser::Wait, false, false) => (
            "in the browser viewer",
            "It returns when the user presses Done in the viewer.",
        ),
        (AskUser::Wait, host, true) => (
            if host {
                "in their Chrome"
            } else {
                "in the browser viewer"
            },
            "It returns with their answer: done, declined (with a note, perhaps), or timed out.",
        ),
        (AskUser::Wait, true, false) => ("in their Chrome", "It returns with their answer."),
    };
    ToolDescBuilder::new("ask_user")
        .description(format!(
            "Ask the user to do something {place} that only they can do, such as signing in, or \
             to confirm something. Say exactly what to do, in the user's language. {returns}"
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "What the user should do." }
            },
            "required": ["message"]
        }))
        .build()
}

pub(crate) fn ask_user_tool_func(session: Arc<BrowserSession>) -> ToolFunc {
    crate::tool_func!(async |args: Value, console: &mut ConsoleClient| -> Value
        with [session = session.clone()]
    {
        let Some(message) = args.pointer("/message").and_then(|v| v.as_str()).filter(|m| !m.trim().is_empty()) else {
            return crate::to_value!({ "error": "missing required parameter: message", "phase": "validation" });
        };
        session.ask(console, message).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_is_32_hex_digits() {
        let t = token();
        assert_eq!(t.len(), 32);
        assert!(t.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(t, token());
    }

    #[test]
    fn host_and_wait_needs_a_handler() {
        assert!(Browser::host().ask_user(AskUser::Wait).check().is_err());
        assert!(Browser::host().check().is_ok());
        assert!(Browser::sandboxed().ask_user(AskUser::Wait).check().is_ok());
        let answered = Browser::host()
            .ask_user(AskUser::Wait)
            .on_ask(async |_| AskResult::Done);
        assert!(answered.check().is_ok());
    }

    #[test]
    fn each_mode_tells_the_agent_its_own_browser() {
        let host = instruction(&Browser::host());
        assert!(host.contains("the user's own Chrome") && !host.contains("viewer"));
        let sandboxed = instruction(&Browser::sandboxed());
        assert!(sandboxed.contains("viewer") && !sandboxed.contains("own Chrome"));
        let desc = ask_user_tool_desc(&Browser::host());
        assert!(
            desc.description
                .as_deref()
                .unwrap_or_default()
                .contains("their Chrome")
        );
    }
}
