//! The person's own browser, a Chromium-based one that speaks CDP (Chrome by default), which the
//! agent drives with its `browser_*` tools over CDP from this process; the agent's console takes
//! no part. Nothing connects until the agent first uses the
//! browser or asks the person something.

mod cdp;
mod host;
mod page;
mod snapshot;

use std::{fmt, future::Future, path::PathBuf, pin::Pin, sync::Arc, time::Duration};

use anyhow::bail;
use futures::future::BoxFuture;

use crate::{
    datatype::{Bytes, Value},
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// How `ask_user` waits for the person.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AskUser {
    /// The agent tells the person in its reply and ends the turn; their next message goes on.
    #[default]
    EndTurn,
    /// Wait in the tool call for what [`Browser::on_ask`] returns.
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

/// A browser for an agent, set with [`AgentBuilder::browser`](crate::agent::AgentBuilder::browser).
#[derive(Clone)]
pub struct Browser {
    user_data_dir: Option<PathBuf>,
    ask_user: AskUser,
    on_ask: Option<AskHandler>,
    ask_timeout: Option<Duration>,
    careful: bool,
    start_url: Option<String>,
}

impl Default for Browser {
    fn default() -> Self {
        Self {
            user_data_dir: None,
            ask_user: AskUser::default(),
            on_ask: None,
            ask_timeout: None,
            careful: true,
            start_url: None,
        }
    }
}

impl fmt::Debug for Browser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Browser")
            .field("user_data_dir", &self.user_data_dir)
            .field("ask_user", &self.ask_user)
            .field("on_ask", &self.on_ask.is_some())
            .field("ask_timeout", &self.ask_timeout)
            .field("careful", &self.careful)
            .field("start_url", &self.start_url)
            .finish()
    }
}

impl Browser {
    /// The person's own browser: a Chromium-based one with remote debugging on (Chrome by default:
    /// 144 or later, at `chrome://inspect/#remote-debugging`), which asks them before it lets the
    /// agent in. The agent works in a window of its own, signed in wherever they are; its tools
    /// reach only the tabs it opened.
    pub fn host() -> Self {
        Self::default()
    }

    /// The browser whose user data directory this is: another Chromium-based one, such as Chrome
    /// Beta, Canary or Chromium. The default is this OS's Chrome.
    pub fn user_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.user_data_dir = Some(dir.into());
        self
    }

    pub fn ask_user(mut self, mode: AskUser) -> Self {
        self.ask_user = mode;
        self
    }

    /// How [`AskUser::Wait`] asks: `handler` gets the request and returns the answer.
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

    /// Whether the agent is told to use sites as a person would, so that sites checking for bots
    /// are less likely to turn it away. On by default.
    pub fn careful(mut self, careful: bool) -> Self {
        self.careful = careful;
        self
    }

    /// The page the agent's tab opens on when the browser starts; a blank one by default.
    pub fn start_url(mut self, url: impl Into<String>) -> Self {
        self.start_url = Some(url.into());
        self
    }

    /// Refuses what cannot work, before anything starts.
    pub(crate) fn check(&self) -> anyhow::Result<()> {
        if self.ask_user == AskUser::Wait && self.on_ask.is_none() {
            bail!("`AskUser::Wait` needs `Browser::on_ask`, which is how it waits");
        }
        Ok(())
    }
}

/// Whether the browser may be used now.
enum Ready {
    /// Go ahead, telling the agent this alongside the result, if anything.
    Go(Option<String>),
    /// Do not; this is the result instead.
    Refused(String),
}

/// The connection and what outlives it.
#[derive(Default)]
struct Live {
    tabs: Option<page::Tabs>,
    /// The agent's tabs as last known, to take up again after a reconnection.
    targets: Vec<String>,
    /// Whether the person declined, so the agent is not asked again until it asks.
    declined: bool,
    /// Whether a connection was ever up, so the next one is a reconnection.
    was_up: bool,
}

/// An agent's browser: connected on first use, and let go of when the agent drops.
pub(crate) struct BrowserSession {
    config: Browser,
    live: tokio::sync::Mutex<Live>,
    /// What [`Self::start`] heard for the agent, told on its next turn.
    unheard: std::sync::Mutex<Option<String>>,
}

impl BrowserSession {
    pub(crate) fn new(config: Browser) -> Arc<Self> {
        Arc::new(Self {
            config,
            live: Default::default(),
            unheard: Default::default(),
        })
    }

    pub(crate) fn config(&self) -> &Browser {
        &self.config
    }

    /// [`Agent::start_browser`](crate::agent::Agent::start_browser): connects, and a refusal is
    /// the caller's error; one after a decline asks again.
    pub(crate) async fn start(&self) -> anyhow::Result<()> {
        let mut live = self.live.lock().await;
        match self.ensure(&mut live, true).await? {
            Ready::Go(said) => {
                if said.is_some() {
                    *self.unheard.lock().unwrap() = said;
                }
                Ok(())
            }
            Ready::Refused(why) => bail!(why),
        }
    }

    /// Connects if not connected. `asking` is `ask_user`'s and the app's, which ask again after
    /// the person declined.
    async fn ensure(&self, live: &mut Live, asking: bool) -> anyhow::Result<Ready> {
        if live.tabs.as_ref().is_some_and(page::Tabs::is_alive) {
            return Ok(Ready::Go(None));
        }
        if let Some(tabs) = live.tabs.take() {
            live.targets = tabs.targets();
        }
        if live.declined && !asking {
            return Ok(Ready::Refused(host::DECLINED.into()));
        }
        let cdp = match host::connect(self.config.user_data_dir.as_deref()).await? {
            Ok(cdp) => cdp,
            Err(host::Closed::NotEnabled(why)) => return Ok(Ready::Refused(why)),
            Err(host::Closed::Declined) => {
                live.declined = true;
                return Ok(Ready::Refused(host::DECLINED.into()));
            }
        };
        live.declined = false;
        let targets = host::agent_tabs(&cdp, &live.targets).await?;
        let moved = !live.targets.is_empty() && live.targets != targets;
        let mut tabs = page::Tabs::new(cdp, &targets).await?;
        let reconnected = std::mem::replace(&mut live.was_up, true);
        if !reconnected && let Some(url) = &self.config.start_url {
            tabs.open(url).await?;
        }
        live.targets = tabs.targets();
        live.tabs = Some(tabs);
        Ok(Ready::Go(reconnected.then(|| {
            let window = if moved {
                " Your tabs were gone, so you are on a new blank one: look before you act."
            } else {
                ""
            };
            format!("[browser] The connection to the user's browser dropped and is back.{window}")
        })))
    }

    /// Runs one tool's work on the agent's tabs, connecting first, as the tool's result.
    async fn act<F>(&self, id: String, work: F) -> Message
    where
        F: for<'a> FnOnce(&'a mut page::Tabs) -> BoxFuture<'a, anyhow::Result<page::Said>>,
    {
        let mut live = self.live.lock().await;
        let result = match self.ensure(&mut live, false).await {
            Ok(Ready::Go(note)) => {
                let tabs = live.tabs.as_mut().expect("connected");
                let said = work(tabs).await;
                if !tabs.is_alive() {
                    live.targets = tabs.targets();
                }
                said.map(|mut said| {
                    if let Some(note) = note {
                        said.text = format!("{note}\n{}", said.text);
                    }
                    said
                })
            }
            Ok(Ready::Refused(why)) => Ok(why.into()),
            Err(e) => Err(e),
        };
        let mut message = Message::new(Role::Tool).with_id(id);
        match result {
            Ok(said) => {
                message = message.with_contents([Part::text(said.text)]);
                if let Some(image) = said.image
                    && let Ok(part) = Part::image_embedded("image/jpeg", Bytes::from(image))
                {
                    message.contents.push(part);
                }
            }
            Err(e) => {
                let gone = live.tabs.as_ref().is_some_and(|t| !t.is_alive());
                let hint = if gone {
                    " The browser connection dropped; the next browser tool connects again."
                } else {
                    ""
                };
                message = message.with_contents([Part::text(format!("Error: {e:#}{hint}"))]);
            }
        }
        message
    }

    /// A reconnection [`Self::start`] made, as a note for the agent's next message.
    pub(crate) fn catch_up(&self) -> Option<String> {
        self.unheard.lock().unwrap().take()
    }

    async fn ask(&self, message: &str) -> Value {
        {
            let mut live = self.live.lock().await;
            match self.ensure(&mut live, true).await {
                Ok(Ready::Go(_)) => {}
                Ok(Ready::Refused(why)) => {
                    return crate::to_value!({ "error": why, "phase": "browser" });
                }
                Err(e) => {
                    return crate::to_value!({ "error": format!("{e:#}"), "phase": "browser" });
                }
            }
        }
        match (self.config.ask_user, &self.config.on_ask) {
            (AskUser::Wait, Some(handler)) => {
                let asked = handler(message.to_string());
                let answer = match self.config.ask_timeout {
                    Some(limit) => tokio::time::timeout(limit, asked).await.ok(),
                    None => Some(asked.await),
                };
                match answer {
                    Some(AskResult::Done) => crate::to_value!({ "done": true }),
                    Some(AskResult::Declined) => crate::to_value!({ "declined": true }),
                    Some(AskResult::Note(note)) => crate::to_value!({ "done": true, "note": note }),
                    None => crate::to_value!({ "timeout": true }),
                }
            }
            // EndTurn (`check` refuses Wait without a handler).
            _ => crate::to_value!({
                "next": "Nothing is shown to the user: tell them in your reply exactly what to do in their browser, then end your turn. Their next message is when it is done."
            }),
        }
    }
}

// ---- the tools ----------------------------------------------------------------------------

fn desc(name: &str, description: &str, parameters: Value) -> ToolDesc {
    ToolDescBuilder::new(name)
        .description(description)
        .parameters(parameters)
        .build()
}

fn no_params() -> Value {
    crate::to_value!({ "type": "object", "properties": {} })
}

fn ref_param() -> Value {
    crate::to_value!({ "type": "string", "description": "A ref from the last `browser_snapshot`, such as `e3`." })
}

fn str_arg<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.pointer(&format!("/{key}")).and_then(|v| v.as_str())
}

fn bool_arg(args: &Value, key: &str) -> Option<bool> {
    args.pointer(&format!("/{key}")).and_then(|v| v.as_bool())
}

fn num_arg(args: &Value, key: &str) -> Option<f64> {
    let v = args.pointer(&format!("/{key}"))?;
    v.as_float().or_else(|| v.as_integer().map(|n| n as f64))
}

/// A browser tool: `run` takes the arguments and works on the agent's tabs, or names the
/// missing parameter.
macro_rules! browser_tool {
    ($session:expr, |$args:ident, $tabs:ident| $body:expr) => {{
        let session: Arc<BrowserSession> = $session.clone();
        crate::tool_func!(async |args: Value, id: String| -> Message
            with [session = session.clone()]
        {
            let $args = args;
            session.act(id, move |$tabs: &mut page::Tabs| Box::pin(async move { $body })).await
        })
    }};
}

/// The browser tools, bound to `session`, with `ask_user`.
pub(crate) fn tools(session: &Arc<BrowserSession>) -> Vec<(ToolDesc, ToolFunc)> {
    let config = session.config();
    let navigate = format!(
        "Open a URL in your current tab. The browser is the user's own, signed in as \
         them; you work in a window of your own and reach only the tabs you opened.{}",
        if config.careful { CAREFUL } else { "" }
    );
    vec![
        (
            desc(
                "browser_navigate",
                &navigate,
                crate::to_value!({
                    "type": "object",
                    "properties": { "url": { "type": "string", "description": "The address to open." } },
                    "required": ["url"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "url") {
                    Some(url) => tabs.navigate(url).await,
                    None => bail!("missing required parameter: url"),
                }
            }),
        ),
        (
            desc(
                "browser_back",
                "Go back one page in your current tab.",
                no_params(),
            ),
            browser_tool!(session, |_args, tabs| tabs.back().await),
        ),
        (
            desc(
                "browser_snapshot",
                "The current page as text, with refs (`[ref=e3]`) on what the other tools act on. \
                 Refs hold until the page navigates; then take a new snapshot. Read pages this way \
                 rather than by screenshot; for a long page, narrow with `interactive` or read a \
                 part with `browser_eval`.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "interactive": { "type": "boolean", "description": "Only what can be acted on, with the headings it sits under. Default false." },
                        "max_chars": { "type": "integer", "description": "Where to cut a long page. Default 20000." }
                    }
                }),
            ),
            browser_tool!(session, |args, tabs| {
                let interactive = bool_arg(&args, "interactive").unwrap_or(false);
                let max = num_arg(&args, "max_chars").map_or(20_000, |n| n.max(500.0) as usize);
                tabs.snapshot(interactive, max).await
            }),
        ),
        (
            desc(
                "browser_click",
                "Click an element, as a person would with the mouse.",
                crate::to_value!({
                    "type": "object",
                    "properties": { "ref": ref_param() },
                    "required": ["ref"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "ref") {
                    Some(r) => tabs.click(r).await,
                    None => bail!("missing required parameter: ref"),
                }
            }),
        ),
        (
            desc(
                "browser_type",
                "Type text into a field, replacing what it holds unless `clear` is false.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "ref": ref_param(),
                        "text": { "type": "string" },
                        "clear": { "type": "boolean", "description": "Replace what the field holds. Default true." },
                        "submit": { "type": "boolean", "description": "Press Enter after. Default false." }
                    },
                    "required": ["ref", "text"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match (str_arg(&args, "ref"), str_arg(&args, "text")) {
                    (Some(r), Some(text)) => {
                        let clear = bool_arg(&args, "clear").unwrap_or(true);
                        let submit = bool_arg(&args, "submit").unwrap_or(false);
                        tabs.type_text(r, text, clear, submit).await
                    }
                    _ => bail!("missing required parameters: ref and text"),
                }
            }),
        ),
        (
            desc(
                "browser_select",
                "Pick an option of a <select> (a combobox in the snapshot) by its text or value.",
                crate::to_value!({
                    "type": "object",
                    "properties": { "ref": ref_param(), "option": { "type": "string" } },
                    "required": ["ref", "option"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match (str_arg(&args, "ref"), str_arg(&args, "option")) {
                    (Some(r), Some(option)) => tabs.select(r, option).await,
                    _ => bail!("missing required parameters: ref and option"),
                }
            }),
        ),
        (
            desc(
                "browser_press",
                "Press a key in the focused element: `Enter`, `Escape`, `Tab`, `ArrowDown`, a \
                 character, or a combination such as `Control+a`.",
                crate::to_value!({
                    "type": "object",
                    "properties": { "key": { "type": "string" } },
                    "required": ["key"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "key") {
                    Some(key) => tabs.press(key).await,
                    None => bail!("missing required parameter: key"),
                }
            }),
        ),
        (
            desc(
                "browser_scroll",
                "Scroll the page a screen up or down, or bring an element into view.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "direction": { "type": "string", "enum": ["down", "up"], "description": "Default down." },
                        "ref": ref_param()
                    }
                }),
            ),
            browser_tool!(session, |args, tabs| {
                let down = str_arg(&args, "direction") != Some("up");
                tabs.scroll(down, str_arg(&args, "ref")).await
            }),
        ),
        (
            desc(
                "browser_screenshot",
                "A picture of what the current tab shows. Costly: use it only when the layout \
                 itself matters.",
                no_params(),
            ),
            browser_tool!(session, |_args, tabs| tabs.screenshot().await),
        ),
        (
            desc(
                "browser_eval",
                "Run JavaScript in the current page and return its value as JSON, such as \
                 `[...document.querySelectorAll('h3')].map(e => e.innerText)`. Statements run as \
                 a function body: `return` what you need.",
                crate::to_value!({
                    "type": "object",
                    "properties": { "expression": { "type": "string" } },
                    "required": ["expression"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "expression") {
                    Some(expression) => tabs.evaluate(expression).await,
                    None => bail!("missing required parameter: expression"),
                }
            }),
        ),
        (
            desc(
                "browser_wait",
                "Wait for some text to appear on the page, or for a number of seconds.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "Text to wait for." },
                        "seconds": { "type": "number", "description": "How long, at most 30. Default 10 with text, else 2." }
                    }
                }),
            ),
            browser_tool!(session, |args, tabs| {
                let text = str_arg(&args, "text");
                let seconds =
                    num_arg(&args, "seconds").unwrap_or(if text.is_some() { 10.0 } else { 2.0 });
                tabs.wait(seconds, text).await
            }),
        ),
        (
            desc(
                "browser_dialog",
                "Answer the dialog a page is waiting on: accept or dismiss a confirm, or answer a \
                 prompt.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "accept": { "type": "boolean" },
                        "text": { "type": "string", "description": "For a prompt: what to answer." }
                    },
                    "required": ["accept"]
                }),
            ),
            browser_tool!(session, |args, tabs| {
                let accept = bool_arg(&args, "accept").unwrap_or(false);
                tabs.dialog(accept, str_arg(&args, "text")).await
            }),
        ),
        (
            desc(
                "browser_tabs",
                "List your tabs, open a new one, switch to one, or close one.",
                crate::to_value!({
                    "type": "object",
                    "properties": {
                        "action": { "type": "string", "enum": ["list", "new", "switch", "close"], "description": "Default list." },
                        "index": { "type": "integer", "description": "For switch and close: the tab's number in the list." }
                    }
                }),
            ),
            browser_tool!(session, |args, tabs| {
                let action = str_arg(&args, "action").unwrap_or("list").to_string();
                let index = num_arg(&args, "index").map(|n| n as usize);
                tabs.tabs(&action, index).await
            }),
        ),
        (
            ask_user_tool_desc(config),
            ask_user_tool_func(session.clone()),
        ),
    ]
}

/// Appended to `browser_navigate` unless [`Browser::careful`] is off.
const CAREFUL: &str = " Sites may turn away traffic that looks automated: start from a site's \
     home page or search and go through its links and forms rather than typed-in query URLs, act \
     at a person's pace, and never retry in a loop. On a robot check, stop and use `ask_user`.";

pub(crate) fn ask_user_tool_desc(browser: &Browser) -> ToolDesc {
    let returns = match browser.ask_user {
        AskUser::EndTurn => {
            "It returns at once: then tell them what to do and end your turn; their next message \
             is when it is done."
        }
        AskUser::Wait => {
            "It returns with their answer: done, declined (with a note, perhaps), or timed out."
        }
    };
    desc(
        "ask_user",
        &format!(
            "Ask the user to do something in their browser that only they can do (signing in, a \
             CAPTCHA, a second factor), or to confirm before anything that cannot be undone \
             (paying, sending, deleting, posting). Never ask for credentials or get around a check \
             yourself. Say exactly what to do, in the user's language. {returns}"
        ),
        crate::to_value!({
            "type": "object",
            "properties": {
                "message": { "type": "string", "description": "What the user should do." }
            },
            "required": ["message"]
        }),
    )
}

fn ask_user_tool_func(session: Arc<BrowserSession>) -> ToolFunc {
    crate::tool_func!(async |args: Value| -> Value
        with [session = session.clone()]
    {
        let Some(message) = args.pointer("/message").and_then(|v| v.as_str()).filter(|m| !m.trim().is_empty()) else {
            return crate::to_value!({ "error": "missing required parameter: message", "phase": "validation" });
        };
        session.ask(message).await
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_needs_a_handler() {
        assert!(Browser::host().ask_user(AskUser::Wait).check().is_err());
        assert!(Browser::host().check().is_ok());
        let answered = Browser::host()
            .ask_user(AskUser::Wait)
            .on_ask(async |_| AskResult::Done);
        assert!(answered.check().is_ok());
    }

    fn description(browser: Browser, name: &str) -> String {
        let session = BrowserSession::new(browser);
        tools(&session)
            .into_iter()
            .find(|(d, _)| d.name == name)
            .and_then(|(d, _)| d.description)
            .unwrap()
    }

    #[test]
    fn ask_user_says_how_it_returns() {
        let ends = description(Browser::host(), "ask_user");
        assert!(ends.contains("Never ask for credentials") && ends.contains("end your turn"));
        let waits = description(
            Browser::host()
                .ask_user(AskUser::Wait)
                .on_ask(async |_| AskResult::Done),
            "ask_user",
        );
        assert!(waits.contains("returns with their answer"));
    }

    #[test]
    fn careful_is_on_unless_turned_off() {
        assert!(description(Browser::host(), "browser_navigate").contains(CAREFUL));
        let careless = description(Browser::host().careful(false), "browser_navigate");
        assert!(careless.contains("user's own") && !careless.contains(CAREFUL));
    }

    #[test]
    fn every_tool_is_named_once() {
        let session = BrowserSession::new(Browser::host());
        let names: Vec<String> = tools(&session).into_iter().map(|(d, _)| d.name).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "{names:?}");
        assert!(
            names
                .iter()
                .all(|n| n.starts_with("browser_") || n == "ask_user")
        );
    }

    /// A headless Chrome on its own profile, which asks nobody: `AILOY_TEST_CHROME` names the
    /// binary, macOS's Chrome by default.
    struct TestChrome {
        child: std::process::Child,
        dir: tempfile::TempDir,
    }

    impl TestChrome {
        fn start() -> Self {
            let binary = std::env::var("AILOY_TEST_CHROME").unwrap_or_else(|_| {
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome".into()
            });
            let dir = tempfile::tempdir().unwrap();
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
            for _ in 0..100 {
                if dir.path().join("DevToolsActivePort").exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Self { child, dir }
        }
    }

    impl Drop for TestChrome {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    /// Calls the tool `name` of `tools` with `args`, returning its text and whether it had an
    /// image.
    async fn call(
        tools: &[(ToolDesc, ToolFunc)],
        name: &str,
        args: serde_json::Value,
    ) -> (String, bool) {
        use futures::StreamExt as _;
        let (_, func) = tools.iter().find(|(d, _)| d.name == name).unwrap();
        let args: Value = serde_json::from_value(args).unwrap();
        let mut stream = func.call_pure(args, "1").expect("browser tools are pure");
        let mut last = None;
        while let Some(output) = stream.next().await {
            last = Some(output.message);
        }
        let message = last.unwrap();
        let text = message
            .contents
            .iter()
            .filter_map(|p| p.as_text())
            .collect::<Vec<_>>()
            .join("\n");
        let image = message
            .contents
            .iter()
            .any(|p| matches!(p, Part::Image { .. }));
        (text, image)
    }

    /// The ref of the first snapshot line holding `needle`.
    fn ref_of(snapshot: &str, needle: &str) -> String {
        let line = snapshot
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no {needle:?} in\n{snapshot}"));
        let at = line.find("[ref=").unwrap() + 5;
        line[at..].trim_end_matches(']').to_string()
    }

    /// Every tool, on a real Chrome lent as a host browser.
    #[tokio::test]
    #[ignore = "needs Chrome"]
    async fn the_tools_drive_a_chrome() {
        let chrome = TestChrome::start();
        let html = "<title>Form</title>\
            <form onsubmit=\"document.title='sent:'+q.value;return false\">\
            <input id=q aria-label=Query><button>Search</button></form>\
            <select aria-label=Size><option value=s>Small</option><option value=l>Large</option></select>\
            <button onclick=\"this.textContent=confirm('Sure?')?'yes':'no'\">Ask</button>\
            <button onclick=\"window.open('about:blank#popup')\">Pop</button>\
            <div style=height:3000px></div><button onclick=\"this.textContent='pressed'\">Far</button>";
        let page = format!("data:text/html,{}", urlencoding::encode(html));
        let session = BrowserSession::new(
            Browser::host()
                .user_data_dir(chrome.dir.path())
                .start_url(&page),
        );
        session.start().await.unwrap();
        let tools = tools(&session);

        let (snapshot, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        assert!(snapshot.contains("\"Form\""), "{snapshot}");
        let query = ref_of(&snapshot, "textbox \"Query\"");
        let (typed, _) = call(
            &tools,
            "browser_type",
            serde_json::json!({ "ref": query, "text": "hello 한글", "submit": true }),
        )
        .await;
        assert!(typed.contains("sent:hello 한글"), "{typed}");

        let (picked, _) = call(
            &tools,
            "browser_select",
            serde_json::json!({ "ref": ref_of(&snapshot, "combobox \"Size\""), "option": "Large" }),
        )
        .await;
        assert!(picked.contains("Selected \"Large\""), "{picked}");

        let (far, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Far\"") }),
        )
        .await;
        assert!(far.starts_with("Clicked."), "{far}");
        let (text, _) = call(&tools, "browser_eval", serde_json::json!({ "expression": "return [...document.querySelectorAll('button')].pop().textContent" })).await;
        assert!(text.starts_with("pressed"), "{text}");

        // A confirm holds the page until the agent answers it.
        let (asked, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Ask\"") }),
        )
        .await;
        assert!(
            asked.contains("confirm dialog") && asked.contains("Sure?"),
            "{asked}"
        );
        let (refused, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        assert!(refused.contains("browser_dialog"), "{refused}");
        let (answered, _) = call(
            &tools,
            "browser_dialog",
            serde_json::json!({ "accept": true }),
        )
        .await;
        assert!(answered.starts_with("Accepted the confirm"), "{answered}");
        let (after, _) = call(
            &tools,
            "browser_snapshot",
            serde_json::json!({ "interactive": true }),
        )
        .await;
        assert!(after.contains("button \"yes\""), "{after}");

        // A tab the agent's page opens becomes its own, and current.
        let (popped, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": ref_of(&after, "button \"Pop\"") }),
        )
        .await;
        assert!(popped.contains("new tab"), "{popped}");
        let (listed, _) = call(&tools, "browser_tabs", serde_json::json!({})).await;
        assert!(
            listed.contains("1 (current)") && listed.contains("#popup"),
            "{listed}"
        );
        let (closed, _) = call(
            &tools,
            "browser_tabs",
            serde_json::json!({ "action": "close", "index": 1 }),
        )
        .await;
        assert!(
            closed.contains("0 (current)") && !closed.contains("#popup"),
            "{closed}"
        );

        let (stale, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": "e999" }),
        )
        .await;
        assert!(stale.contains("take a new `browser_snapshot`"), "{stale}");
        let (_, image) = call(&tools, "browser_screenshot", serde_json::json!({})).await;
        assert!(image);
        let (scrolled, _) = call(
            &tools,
            "browser_scroll",
            serde_json::json!({ "direction": "down" }),
        )
        .await;
        assert!(scrolled.starts_with("Scrolled down"), "{scrolled}");
        let (waited, _) = call(
            &tools,
            "browser_wait",
            serde_json::json!({ "text": "pressed", "seconds": 2 }),
        )
        .await;
        assert!(waited.contains("is on the page"), "{waited}");

        // Only the agent's tabs: the page Chrome started with is not among them.
        let (listed, _) = call(&tools, "browser_tabs", serde_json::json!({})).await;
        assert_eq!(listed.lines().count(), 1, "{listed}");
        drop(chrome);
    }

    /// A turn a real model plays on a real Chrome: needs Chrome and `ANTHROPIC_API_KEY`.
    #[tokio::test]
    #[ignore = "needs Chrome and ANTHROPIC_API_KEY"]
    async fn a_model_reads_and_acts_on_a_page() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        crate::model::set_lang_model_provider("anthropic", crate::model::Backend::Api);
        let chrome = TestChrome::start();
        let html = "<title>Shop</title><h1>Mice</h1>\
            <ul><li>Glide $29</li><li>Pro $79</li><li>Lite $19</li></ul>\
            <input aria-label=Coupon><button onclick=\"document.title='code:'+document.querySelector('input').value\">Apply</button>";
        let page = format!("data:text/html,{}", urlencoding::encode(html));
        let mut agent = crate::agent::AgentBuilder::new("anthropic/claude-haiku-4-5")
            .browser(
                Browser::host()
                    .user_data_dir(chrome.dir.path())
                    .start_url(&page),
            )
            .build()
            .await
            .unwrap();
        let query = Message::new(Role::User).with_contents([Part::text(
            "In the browser tab you have, list the mice under $50, then type SAVE10 into the \
             coupon field and apply it.",
        )]);
        let mut called = Vec::new();
        let mut answer = String::new();
        let mut turn = agent.run(query);
        while let Some(output) = turn.next().await {
            let message = output.unwrap().message;
            for call in message.tool_calls.iter().flatten() {
                if let Some((_, name, _)) = call.as_function() {
                    called.push(name.to_string());
                }
            }
            if message.role == Role::Assistant
                && let Some(text) = message.contents.iter().find_map(|p| p.as_text())
            {
                answer = text.to_string();
            }
        }
        drop(turn);
        eprintln!("tools: {called:?}\nanswer: {answer}");
        assert!(called.iter().any(|n| n == "browser_snapshot"), "{called:?}");
        assert!(called.iter().any(|n| n == "browser_type"), "{called:?}");
        assert!(
            answer.contains("Glide") && answer.contains("Lite") && !answer.contains("Pro"),
            "{answer}"
        );
        // Applied: the button wrote the coupon into the title.
        let session = agent.state.browser.clone().unwrap();
        let (title, _) = call(
            &tools(&session),
            "browser_eval",
            serde_json::json!({ "expression": "document.title" }),
        )
        .await;
        assert!(title.starts_with("code:SAVE10"), "{title}");
        drop(chrome);
    }
}
