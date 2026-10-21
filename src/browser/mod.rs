//! The person's own browser, a Chromium-based one that speaks CDP (Chrome by default), which the
//! agent drives with its `browser_*` tools over CDP from this process; the agent's console takes
//! no part. Nothing connects until the agent first uses the
//! browser or asks the person something.

mod cdp;
mod isolated;
mod page;
mod snapshot;
mod user;

pub use user::UserBrowser;

use std::{
    fmt,
    sync::{Arc, atomic::Ordering},
    time::Duration,
};

use anyhow::{Context as _, bail};
use futures::future::BoxFuture;
use virtx::console::ConsoleClient;

use crate::{
    datatype::{Bytes, Value},
    message::{Message, Part, Role},
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// A browser for an agent, set with [`AgentBuilder::browser`](crate::agent::AgentBuilder::browser).
#[derive(Clone)]
pub struct Browser {
    /// The agent's own browser rather than the person's.
    isolated: bool,
    user_browser: Option<UserBrowser>,
    headless: bool,
    executable: Option<std::path::PathBuf>,
    profile: Option<std::path::PathBuf>,
    careful: bool,
    start_url: Option<String>,
}

impl Default for Browser {
    fn default() -> Self {
        Self {
            isolated: false,
            user_browser: None,
            headless: false,
            executable: None,
            profile: None,
            careful: true,
            start_url: None,
        }
    }
}

impl fmt::Debug for Browser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Browser")
            .field("isolated", &self.isolated)
            .field("user_browser", &self.user_browser)
            .field("headless", &self.headless)
            .field("executable", &self.executable)
            .field("profile", &self.profile)
            .field("careful", &self.careful)
            .field("start_url", &self.start_url)
            .finish()
    }
}

impl Browser {
    /// The person's own browser: a Chromium-based one with remote debugging on (in Chrome 144 or
    /// later, at `chrome://inspect/#remote-debugging`), which asks them before it lets the agent
    /// in. With none named by [`user_browser`](Self::user_browser), it is whichever of
    /// [`UserBrowser::NAMED`] turned remote debugging on last. The agent works in a window of its
    /// own, signed in wherever they are; its tools reach only the tabs it opened.
    pub fn user() -> Self {
        Self::default()
    }

    /// A browser of the agent's own, apart from the person's: started on first use and closed
    /// with the agent, in a window the person can see and act in unless
    /// [`headless`](Self::headless). The browser is [`executable`](Self::executable) if named,
    /// else a Chromium-based one installed here (Chrome, Chromium, Edge or Brave); its profile is
    /// a fresh one each time unless [`profile`](Self::profile) keeps one.
    pub fn isolated() -> Self {
        Self {
            isolated: true,
            ..Self::default()
        }
    }

    /// For [`isolated`](Self::isolated): no window, as on a server. Off by default.
    pub fn headless(mut self, headless: bool) -> Self {
        self.headless = headless;
        self
    }

    /// For [`isolated`](Self::isolated): the Chromium-based browser to start, by its executable.
    pub fn executable(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.executable = Some(path.into());
        self
    }

    /// For [`isolated`](Self::isolated): the user data directory it keeps, so sign-ins last
    /// from one run to the next.
    pub fn profile(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.profile = Some(dir.into());
        self
    }

    /// The browser to use, rather than the one found: a named one, or any other by its user
    /// data directory ([`UserBrowser::Custom`]).
    pub fn user_browser(mut self, browser: UserBrowser) -> Self {
        self.user_browser = Some(browser);
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
        if self.isolated && self.user_browser.is_some() {
            bail!("`Browser::user_browser` is for `Browser::user`, not `Browser::isolated`");
        }
        if !self.isolated && (self.headless || self.executable.is_some() || self.profile.is_some())
        {
            bail!("`headless`, `executable` and `profile` are for `Browser::isolated`");
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
    /// Whether a connection was ever up, so the next one is a reconnection.
    was_up: bool,
    /// The isolated browser started for the agent; dropped before its profile.
    launched: Option<isolated::Launched>,
    /// Its profile when none is kept, for as long as the session lasts.
    profile: Option<tempfile::TempDir>,
}

/// An agent's browser: connected on first use, and let go of when the agent drops.
pub(crate) struct BrowserSession {
    config: Browser,
    live: tokio::sync::Mutex<Live>,
    /// What [`Self::start`] heard for the agent, told on its next turn.
    unheard: std::sync::Mutex<Option<String>>,
    /// Whether the person declined the browser, so it does not ask them again before their next
    /// message.
    declined: std::sync::atomic::AtomicBool,
    /// Files taken from the console for an upload, on this host where the browser reads them.
    uploads: std::sync::OnceLock<tempfile::TempDir>,
    /// Whether the agent has the console tools, to tell it a download can go there.
    console: std::sync::atomic::AtomicBool,
}

impl BrowserSession {
    pub(crate) fn new(config: Browser) -> Arc<Self> {
        Arc::new(Self {
            config,
            live: Default::default(),
            unheard: Default::default(),
            declined: Default::default(),
            uploads: Default::default(),
            console: Default::default(),
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

    /// Connects if not connected. `asking` is the app's, which asks again after the person
    /// declined.
    async fn ensure(&self, live: &mut Live, asking: bool) -> anyhow::Result<Ready> {
        if live.tabs.as_ref().is_some_and(page::Tabs::is_alive) {
            return Ok(Ready::Go(None));
        }
        if let Some(tabs) = live.tabs.take() {
            live.targets = tabs.targets();
        }
        let mut restarted = false;
        let cdp = if self.config.isolated {
            restarted = self.launch(live).await?;
            self.connect_launched(live).await?
        } else {
            if self.declined.load(Ordering::Relaxed) && !asking {
                return Ok(Ready::Refused(user::DECLINED.into()));
            }
            match user::connect(self.config.user_browser.as_ref()).await? {
                Ok(cdp) => cdp,
                Err(user::Closed::NotEnabled(why)) => return Ok(Ready::Refused(why)),
                Err(user::Closed::Declined) => {
                    self.declined.store(true, Ordering::Relaxed);
                    return Ok(Ready::Refused(user::DECLINED.into()));
                }
            }
        };
        self.declined.store(false, Ordering::Relaxed);
        let targets = if restarted {
            isolated::first_tab(&cdp).await?
        } else {
            user::agent_tabs(&cdp, &live.targets).await?
        };
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
            if self.config.isolated && restarted {
                "[browser] Your browser had closed and is started again: its tabs are gone, so you \
                 are on a new blank one."
                    .to_string()
            } else {
                let whose = if self.config.isolated {
                    "your"
                } else {
                    "the user's"
                };
                format!("[browser] The connection to {whose} browser dropped and is back.{window}")
            }
        })))
    }

    /// Starts the isolated browser unless it is running; says whether it started it.
    async fn launch(&self, live: &mut Live) -> anyhow::Result<bool> {
        if live
            .launched
            .as_mut()
            .is_some_and(isolated::Launched::alive)
        {
            return Ok(false);
        }
        live.launched = None;
        let Some(executable) = self.config.executable.clone().or_else(isolated::installed) else {
            bail!(
                "no Chromium-based browser is installed here (Chrome, Chromium, Edge or Brave): \
                 install one, or name one with `Browser::executable`"
            );
        };
        let dir = match &self.config.profile {
            Some(dir) => dir.clone(),
            None => match &live.profile {
                Some(dir) => dir.path().to_path_buf(),
                None => live
                    .profile
                    .insert(
                        tempfile::Builder::new()
                            .prefix("ailoy-browser-")
                            .tempdir()?,
                    )
                    .path()
                    .to_path_buf(),
            },
        };
        live.launched = Some(isolated::launch(&executable, &dir, self.config.headless)?);
        Ok(true)
    }

    /// Connects to the isolated browser, waiting while it opens its port.
    async fn connect_launched(&self, live: &mut Live) -> anyhow::Result<Arc<cdp::Cdp>> {
        let launched = live.launched.as_mut().expect("launched");
        let browser = UserBrowser::Custom(launched.dir.clone());
        let started = tokio::time::Instant::now();
        loop {
            if let Ok(cdp) = user::connect(Some(&browser)).await? {
                return Ok(cdp);
            }
            if !launched.alive() {
                bail!("the agent's browser quit as it started");
            }
            if started.elapsed() > Duration::from_secs(30) {
                bail!("the agent's browser did not open its debugging port within 30 s");
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Runs `work` on the agent's tabs, connecting first; a refusal is its error.
    async fn run<T>(
        &self,
        work: impl for<'a> FnOnce(&'a mut page::Tabs) -> BoxFuture<'a, anyhow::Result<T>>,
    ) -> anyhow::Result<T> {
        let mut live = self.live.lock().await;
        match self.ensure(&mut live, false).await? {
            Ready::Go(_) => work(live.tabs.as_mut().expect("connected")).await,
            Ready::Refused(why) => bail!(why),
        }
    }

    /// `bytes`, the console's file at `path`, kept on this host under the same name for the
    /// browser to upload, for as long as the session lasts.
    fn hold_upload(&self, path: &str, bytes: &[u8]) -> anyhow::Result<std::path::PathBuf> {
        let dir = match self.uploads.get() {
            Some(dir) => dir,
            None => {
                let made = tempfile::Builder::new()
                    .prefix("ailoy-uploads-")
                    .tempdir()?;
                self.uploads.get_or_init(|| made)
            }
        };
        let name = path
            .rsplit('/')
            .next()
            .filter(|n| !n.is_empty())
            .unwrap_or("upload");
        // A directory each, so two files of one name do not overwrite each other.
        let place = tempfile::Builder::new().tempdir_in(dir.path())?.keep();
        let file = place.join(name);
        std::fs::write(&file, bytes)?;
        Ok(file)
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
                    if self.console.load(std::sync::atomic::Ordering::Relaxed)
                        && said.text.contains("Downloaded file:")
                    {
                        said.text
                            .push_str(" `browser_save_download` copies it into your console.");
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
                if let Some((kind, image)) = said.image
                    && let Ok(part) = Part::image_embedded(kind, Bytes::from(image))
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

    /// A reconnection [`Self::start`] made, as a note for the agent's next message. A new message
    /// from the person may ask for the browser after they declined it, so it may ask them again.
    pub(crate) fn catch_up(&self) -> Option<String> {
        self.declined.store(false, Ordering::Relaxed);
        self.unheard.lock().unwrap().take()
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

fn obj(properties: Value, required: &[&str]) -> Value {
    let required = Value::array(required.iter().map(|r| Value::string(*r)));
    crate::to_value!({ "type": "object", "properties": properties, "required": required })
}

fn int_arg(args: &Value, key: &str) -> Option<usize> {
    num_arg(args, key).map(|n| n.max(0.0) as usize)
}

/// The browser tools, bound to `session`; with `console`, also those that keep or take a file
/// there.
pub(crate) fn tools(session: &Arc<BrowserSession>, console: bool) -> Vec<(ToolDesc, ToolFunc)> {
    let config = session.config();
    let whose = if config.isolated {
        "The browser is your own, apart from the user's, and signed in nowhere they have not \
         signed in within it."
    } else {
        "The browser is the user's own, signed in as them; you work in a window of your own and \
         reach only the tabs you opened."
    };
    let act_in = if config.isolated && config.headless {
        "tell the user what it needs; the browser has no window for them to act in"
    } else {
        "tell the user exactly what to do in the browser"
    };
    let navigate = format!(
        "Open a URL in your current tab, or in a new tab with `new_tab`. {whose} When a site needs \
         what only the user can do (signing in, a CAPTCHA, a second factor), {act_in}, and end \
         your turn; never ask for credentials or get around a check yourself. Before anything \
         that cannot be undone (paying, sending, deleting, posting), ask the user and end your \
         turn.{}",
        if config.careful { CAREFUL } else { "" }
    );
    let mut tools = vec![
        (
            desc(
                "browser_navigate",
                &navigate,
                obj(
                    crate::to_value!({
                        "url": { "type": "string" },
                        "new_tab": { "type": "boolean", "description": "Open it in a new tab. Default false." }
                    }),
                    &["url"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "url") {
                    Some(url) => {
                        let new_tab = bool_arg(&args, "new_tab").unwrap_or(false);
                        tabs.navigate(url, new_tab).await
                    }
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
                "The current page as text: its accessibility tree in document order, frames \
                 included, with refs (`[ref=e3]`) on what the other tools act on, after a line on \
                 the page as a whole. What is hidden or covered is left out; a box that scrolls \
                 says how far. A ref names the same element until the page navigates, and one \
                 that appeared since the last snapshot is marked `[new]`. Read pages this way \
                 rather than by screenshot, and narrow a long page with the options.",
                obj(
                    crate::to_value!({
                        "interactive": { "type": "boolean", "description": "Only what can be acted on, with the headings it sits under. Default false." },
                        "depth": { "type": "integer", "description": "How deep the tree goes." },
                        "selector": { "type": "string", "description": "A CSS selector: only that element and what is inside it." },
                        "urls": { "type": "boolean", "description": "Show where each link goes. Default false." },
                        "max_chars": { "type": "integer", "description": "Where to cut a long page. Default 20000." }
                    }),
                    &[],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                let look = page::Look {
                    interactive: bool_arg(&args, "interactive").unwrap_or(false),
                    depth: int_arg(&args, "depth"),
                    selector: str_arg(&args, "selector"),
                    urls: bool_arg(&args, "urls").unwrap_or(false),
                    max_chars: int_arg(&args, "max_chars").map_or(20_000, |n| n.max(500)),
                };
                tabs.snapshot(&look).await
            }),
        ),
        (
            desc(
                "browser_click",
                "Click an element by ref, as a person would with the mouse. Only when what you \
                 want has no ref (a canvas, a map, a custom widget), click at x and y instead: \
                 pixels in your last `browser_screenshot`. The result says what was there.",
                obj(
                    crate::to_value!({
                        "ref": ref_param(),
                        "x": { "type": "number" },
                        "y": { "type": "number" }
                    }),
                    &[],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                let at = num_arg(&args, "x").zip(num_arg(&args, "y"));
                tabs.click(str_arg(&args, "ref"), at).await
            }),
        ),
        (
            desc(
                "browser_input",
                "Input text into a field by ref, typed a key at a time. Clears existing text by \
                 default; pass text=\"\" to clear only, or clear=false to append.",
                obj(
                    crate::to_value!({
                        "ref": ref_param(),
                        "text": { "type": "string" },
                        "clear": { "type": "boolean", "description": "Clear the field first. Default true." }
                    }),
                    &["ref", "text"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match (str_arg(&args, "ref"), str_arg(&args, "text")) {
                    (Some(r), Some(text)) => {
                        let clear = bool_arg(&args, "clear").unwrap_or(true);
                        tabs.input(r, text, clear).await
                    }
                    _ => bail!("missing required parameters: ref and text"),
                }
            }),
        ),
        (
            desc(
                "browser_send_keys",
                "Send keys to what has focus: a key (`Escape`, `Enter`, `PageDown`, `F5`), a \
                 shortcut (`Control+o`), or text, typed a key at a time.",
                obj(
                    crate::to_value!({ "keys": { "type": "string" } }),
                    &["keys"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "keys") {
                    Some(keys) => tabs.send_keys(keys).await,
                    None => bail!("missing required parameter: keys"),
                }
            }),
        ),
        (
            desc(
                "browser_scroll",
                "Scroll by pages: the page, or with `ref` inside that element (a dropdown, a \
                 panel), each page by the viewport's height. 10 pages reaches the end.",
                obj(
                    crate::to_value!({
                        "down": { "type": "boolean", "description": "Down, or up when false. Default true." },
                        "pages": { "type": "number", "description": "0.5 to 10. Default 1." },
                        "ref": ref_param()
                    }),
                    &[],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                let down = bool_arg(&args, "down").unwrap_or(true);
                let pages = num_arg(&args, "pages").unwrap_or(1.0);
                tabs.scroll(down, pages, str_arg(&args, "ref")).await
            }),
        ),
        (
            desc(
                "browser_find_text",
                "Scroll to the first element whose text (or an attribute) holds the given text, \
                 case and all.",
                obj(
                    crate::to_value!({ "text": { "type": "string" } }),
                    &["text"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "text") {
                    Some(text) => tabs.find_text(text).await,
                    None => bail!("missing required parameter: text"),
                }
            }),
        ),
        (
            desc(
                "browser_dropdown_options",
                "The options of a dropdown by ref: a <select>, an ARIA combobox, menu or listbox, or a \
                 Semantic UI dropdown, or the first one inside the element. Use an option's exact \
                 text or value in browser_select_dropdown.",
                obj(crate::to_value!({ "ref": ref_param() }), &["ref"]),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "ref") {
                    Some(r) => tabs.dropdown_options(r).await,
                    None => bail!("missing required parameter: ref"),
                }
            }),
        ),
        (
            desc(
                "browser_select_dropdown",
                "Pick an option of a dropdown by ref, by its text or value (case ignored): a <select> \
                 is set, a menu or listbox has the option clicked.",
                obj(
                    crate::to_value!({ "ref": ref_param(), "text": { "type": "string" } }),
                    &["ref", "text"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match (str_arg(&args, "ref"), str_arg(&args, "text")) {
                    (Some(r), Some(text)) => tabs.select_dropdown(r, text).await,
                    _ => bail!("missing required parameters: ref and text"),
                }
            }),
        ),
        (
            desc(
                "browser_search_page",
                "Search the page's text, like grep: the number of matches and each with the text \
                 around it. Instant and cheap: use it to find or check something on a long page.",
                obj(
                    crate::to_value!({
                        "pattern": { "type": "string" },
                        "regex": { "type": "boolean", "description": "A regular expression. Default false." },
                        "case_sensitive": { "type": "boolean", "description": "Default false." },
                        "context_chars": { "type": "integer", "description": "Characters of text shown around each match. Default 150." },
                        "css_scope": { "type": "string", "description": "A CSS selector to search within." },
                        "max_results": { "type": "integer", "description": "Default 25." }
                    }),
                    &["pattern"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "pattern") {
                    Some(pattern) => {
                        let search = page::Search {
                            regex: bool_arg(&args, "regex").unwrap_or(false),
                            case_sensitive: bool_arg(&args, "case_sensitive").unwrap_or(false),
                            context_chars: int_arg(&args, "context_chars").unwrap_or(150),
                            css_scope: str_arg(&args, "css_scope"),
                            max_results: int_arg(&args, "max_results").unwrap_or(25),
                        };
                        tabs.search_page(pattern, &search).await
                    }
                    None => bail!("missing required parameter: pattern"),
                }
            }),
        ),
        (
            desc(
                "browser_find_elements",
                "Query elements by CSS selector: how many match, and each one's tag, text and the \
                 attributes asked for (`href`, `src`, …). Instant and cheap.",
                obj(
                    crate::to_value!({
                        "selector": { "type": "string" },
                        "attributes": { "type": "array", "items": { "type": "string" }, "description": "Attributes to show, such as href and src. None by default." },
                        "max_results": { "type": "integer", "description": "Default 50." },
                        "include_text": { "type": "boolean", "description": "Show each one's text. Default true." }
                    }),
                    &["selector"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "selector") {
                    Some(selector) => {
                        let attributes: Option<Vec<String>> = args
                            .pointer("/attributes")
                            .and_then(|v| v.as_array())
                            .map(|list| {
                                list.iter()
                                    .filter_map(|v| v.as_str().map(str::to_string))
                                    .collect()
                            });
                        let max = int_arg(&args, "max_results").unwrap_or(50);
                        let text = bool_arg(&args, "include_text").unwrap_or(true);
                        tabs.find_elements(selector, attributes.as_deref(), max, text)
                            .await
                    }
                    None => bail!("missing required parameter: selector"),
                }
            }),
        ),
        (
            desc(
                "browser_evaluate",
                "Run JavaScript in the page and return its value as JSON, such as \
                 `[...document.querySelectorAll('h3')].map(e => e.innerText)`. Statements run as \
                 a function body: `return` what you need. Browser APIs only.",
                obj(
                    crate::to_value!({ "code": { "type": "string" } }),
                    &["code"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "code") {
                    Some(code) => tabs.evaluate(code).await,
                    None => bail!("missing required parameter: code"),
                }
            }),
        ),
        (
            desc(
                "browser_screenshot",
                "A picture of what the current tab shows. Costly: use it when the layout itself \
                 matters, or to click by coordinates.",
                no_params(),
            ),
            browser_tool!(session, |_args, tabs| tabs.screenshot().await),
        ),
        (
            desc(
                "browser_wait",
                "Wait for some text to appear on the page, or for a number of seconds.",
                obj(
                    crate::to_value!({
                        "text": { "type": "string", "description": "Text to wait for." },
                        "seconds": { "type": "number", "description": "How long, at most 30. Default 10 with text, else 3." }
                    }),
                    &[],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                let text = str_arg(&args, "text");
                let seconds =
                    num_arg(&args, "seconds").unwrap_or(if text.is_some() { 10.0 } else { 3.0 });
                tabs.wait(seconds, text).await
            }),
        ),
        (
            desc(
                "browser_dialog",
                "Answer the dialog a page is waiting on: accept or dismiss a confirm, or answer a \
                 prompt. Alerts are accepted on their own, and you are told.",
                obj(
                    crate::to_value!({
                        "accept": { "type": "boolean" },
                        "text": { "type": "string", "description": "For a prompt: what to answer." }
                    }),
                    &["accept"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                let accept = bool_arg(&args, "accept").unwrap_or(false);
                tabs.dialog(accept, str_arg(&args, "text")).await
            }),
        ),
        (
            desc(
                "browser_tabs",
                "List your tabs: each one's id, title and address, and which is current.",
                no_params(),
            ),
            browser_tool!(session, |_args, tabs| tabs.list_tabs().await),
        ),
        (
            desc(
                "browser_switch_tab",
                "Switch to another of your tabs, by its id from `browser_tabs`.",
                obj(
                    crate::to_value!({ "tab_id": { "type": "string" } }),
                    &["tab_id"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "tab_id") {
                    Some(id) => tabs.switch_tab(id).await,
                    None => bail!("missing required parameter: tab_id"),
                }
            }),
        ),
        (
            desc(
                "browser_close_tab",
                "Close one of your tabs, by its id from `browser_tabs`.",
                obj(
                    crate::to_value!({ "tab_id": { "type": "string" } }),
                    &["tab_id"],
                ),
            ),
            browser_tool!(session, |args, tabs| {
                match str_arg(&args, "tab_id") {
                    Some(id) => tabs.close_tab(id).await,
                    None => bail!("missing required parameter: tab_id"),
                }
            }),
        ),
    ];
    session
        .console
        .store(console, std::sync::atomic::Ordering::Relaxed);
    if console {
        tools.extend(console_tools(session));
    }
    tools
}

/// The tools that keep a file in the console, or take one from it.
fn console_tools(session: &Arc<BrowserSession>) -> Vec<(ToolDesc, ToolFunc)> {
    let file_param = crate::to_value!({ "file_name": { "type": "string", "description": "A path in your console; a bare name goes in its current directory." } });
    vec![
        (
            desc(
                "browser_save_screenshot",
                "Save a screenshot of the current tab as a PNG in your console, and say where.",
                obj(file_param.clone(), &["file_name"]),
            ),
            save_tool(session, "screenshot", |tabs| {
                Box::pin(tabs.screenshot_png())
            }),
        ),
        (
            desc(
                "browser_save_as_pdf",
                "Save the current page, all of it, as a PDF in your console, and say where.",
                obj(
                    crate::to_value!({
                        "file_name": { "type": "string", "description": "A path in your console; a bare name goes in its current directory. Defaults to the page's title; .pdf is added when missing." },
                        "print_background": { "type": "boolean", "description": "Include background graphics and colors. Default true." },
                        "landscape": { "type": "boolean", "description": "Default false." },
                        "scale": { "type": "number", "description": "0.1 to 2. Default 1." },
                        "paper_format": { "type": "string", "description": "Letter, Legal, A4, A3 or Tabloid. Default Letter." },
                        "display_header_footer": { "type": "boolean", "description": "The date in the header, the address and page numbers in the footer, as the print dialog has them. Default true." },
                        "header_template": { "type": "string", "description": "HTML for the header; spans of class date, title, url, pageNumber or totalPages are filled in. Give it a font-size." },
                        "footer_template": { "type": "string", "description": "HTML for the footer, as for the header." }
                    }),
                    &[],
                ),
            ),
            pdf_tool(session),
        ),
        (
            desc(
                "browser_upload_file",
                "Put a file from your console into a file input by ref.",
                obj(
                    crate::to_value!({
                        "ref": ref_param(),
                        "path": { "type": "string", "description": "The file's path in your console." }
                    }),
                    &["ref", "path"],
                ),
            ),
            upload_tool(session),
        ),
        (
            desc(
                "browser_save_download",
                "Copy a file the browser downloaded for you (it says so after the click) into your \
                 console, and say where. Up to 512 MiB.",
                obj(
                    crate::to_value!({
                        "name": { "type": "string", "description": "The downloaded file's name." },
                        "file_name": { "type": "string", "description": "A path in your console; a bare name goes in its current directory. Defaults to the file's name." }
                    }),
                    &["name"],
                ),
            ),
            save_download_tool(session),
        ),
    ]
}

/// The most a download copied into the console may be.
const MAX_DOWNLOAD: u64 = 512 * 1024 * 1024;

/// How much of a file goes to the console in one write, well under virtx's 64 MiB message.
const CHUNK: usize = 16 * 1024 * 1024;

fn save_download_tool(session: &Arc<BrowserSession>) -> ToolFunc {
    let session = session.clone();
    crate::tool_func!(async |args: Value, id: String, console: &mut ConsoleClient| -> Message
        with [session = session.clone()]
    {
        let said = async {
            let name = str_arg(&args, "name").context("missing required parameter: name")?;
            let downloaded = session.run(|tabs| Box::pin(async move { Ok(tabs.downloaded()) })).await?;
            // Only what the agent's own tabs downloaded: never any other file on this host.
            let Some((_, host)) = downloaded.iter().rev().find(|(n, _)| n == name) else {
                let names: Vec<&str> = downloaded.iter().map(|(n, _)| n.as_str()).collect();
                if names.is_empty() {
                    bail!("nothing has been downloaded for you yet");
                }
                bail!("no download named {name:?}; the downloads are {}", names.join(", "));
            };
            let size = tokio::fs::metadata(host).await?.len();
            if size > MAX_DOWNLOAD {
                bail!(
                    "{name} is {} bytes, over the 512 MiB a download may be to copy",
                    thousands(size as usize)
                );
            }
            let path = console_path(console, str_arg(&args, "file_name").unwrap_or(name)).await?;
            copy_to_console(console, host, &path, CHUNK).await?;
            anyhow::Ok(format!(
                "Saved the download {name} ({} bytes) to {path}.",
                thousands(size as usize)
            ))
        }
        .await;
        tool_text(id, said)
    })
}

/// Copies the host's file `from` to `to` in the console, `chunk` bytes per write.
async fn copy_to_console(
    console: &mut ConsoleClient,
    from: &std::path::Path,
    to: &str,
    chunk: usize,
) -> anyhow::Result<()> {
    use tokio::io::AsyncReadExt as _;
    let mut file = tokio::fs::File::open(from).await?;
    let mut offset = 0u64;
    loop {
        let mut buf = Vec::with_capacity(chunk);
        (&mut file).take(chunk as u64).read_to_end(&mut buf).await?;
        let read = buf.len() as u64;
        if offset == 0 {
            write_console_file(console, to, buf).await?;
        } else if read > 0 {
            console.write(to, buf, Some(offset)).await?;
        }
        offset += read;
        if read < chunk as u64 {
            return Ok(());
        }
    }
}

/// The bytes a page gives to keep as a file.
type Take = for<'a> fn(&'a mut page::Tabs) -> BoxFuture<'a, anyhow::Result<Vec<u8>>>;

fn save_tool(session: &Arc<BrowserSession>, what: &'static str, take: Take) -> ToolFunc {
    let session = session.clone();
    crate::tool_func!(async |args: Value, id: String, console: &mut ConsoleClient| -> Message
        with [session = session.clone()]
    {
        let said = async {
            let name = str_arg(&args, "file_name").context("missing required parameter: file_name")?;
            let path = console_path(console, name).await?;
            let bytes = session.run(take).await?;
            write_console_file(console, &path, bytes).await?;
            anyhow::Ok(format!("Saved the {what} to {path}."))
        }
        .await;
        tool_text(id, said)
    })
}

fn pdf_tool(session: &Arc<BrowserSession>) -> ToolFunc {
    let session = session.clone();
    crate::tool_func!(async |args: Value, id: String, console: &mut ConsoleClient| -> Message
        with [session = session.clone()]
    {
        let said = async {
            let pdf = page::Pdf {
                print_background: bool_arg(&args, "print_background").unwrap_or(true),
                landscape: bool_arg(&args, "landscape").unwrap_or(false),
                scale: num_arg(&args, "scale").unwrap_or(1.0).clamp(0.1, 2.0),
                paper_format: str_arg(&args, "paper_format").unwrap_or("Letter").to_string(),
                display_header_footer: bool_arg(&args, "display_header_footer").unwrap_or(true),
                header_template: str_arg(&args, "header_template").map(str::to_string),
                footer_template: str_arg(&args, "footer_template").map(str::to_string),
            };
            let (bytes, title) = session.run(move |tabs| Box::pin(tabs.pdf(pdf))).await?;
            let name = match str_arg(&args, "file_name") {
                Some(name) => name.to_string(),
                None => {
                    let safe: String = title
                        .chars()
                        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || matches!(c, '_' | '-'))
                        .collect();
                    let safe: String = safe.trim().chars().take(50).collect();
                    if safe.is_empty() { "page".to_string() } else { safe }
                }
            };
            let name = if name.to_lowercase().ends_with(".pdf") { name } else { format!("{name}.pdf") };
            let mut path = console_path(console, &name).await?;
            // A file already there is kept, and this one numbered after it.
            if console_has(console, &path).await {
                let base = path[..path.len() - 4].to_string();
                let mut n = 1;
                while console_has(console, &format!("{base} ({n}).pdf")).await {
                    n += 1;
                }
                path = format!("{base} ({n}).pdf");
            }
            let size = bytes.len();
            write_console_file(console, &path, bytes).await?;
            anyhow::Ok(format!("Saved page as PDF: {path} ({} bytes)", thousands(size)))
        }
        .await;
        tool_text(id, said)
    })
}

/// Whether the console has a file at `path`.
async fn console_has(console: &mut ConsoleClient, path: &str) -> bool {
    console
        .exec(["test", "-e", path], Some(10_000))
        .await
        .is_ok_and(|out| out.code == 0)
}

fn upload_tool(session: &Arc<BrowserSession>) -> ToolFunc {
    let session = session.clone();
    crate::tool_func!(async |args: Value, id: String, console: &mut ConsoleClient| -> Message
        with [session = session.clone()]
    {
        let (Some(r), Some(path)) = (str_arg(&args, "ref").map(str::to_string), str_arg(&args, "path")) else {
            return tool_text(id, Err(anyhow::anyhow!("missing required parameters: ref and path")));
        };
        let bytes = match read_console_file(console, path).await {
            Ok(bytes) if bytes.is_empty() => {
                return tool_text(id, Err(anyhow::anyhow!("File {path} is empty (0 bytes). The file may not have been saved correctly.")));
            }
            other => other,
        };
        let held = match bytes.and_then(|bytes| session.hold_upload(path, &bytes)) {
            Ok(held) => held,
            Err(e) => return tool_text(id, Err(e)),
        };
        session
            .act(id, move |tabs: &mut page::Tabs| Box::pin(async move { tabs.upload(&r, &held).await }))
            .await
    })
}

fn tool_text(id: String, said: anyhow::Result<String>) -> Message {
    let text = match said {
        Ok(text) => text,
        Err(e) => format!("Error: {e:#}"),
    };
    Message::new(Role::Tool)
        .with_contents([Part::text(text)])
        .with_id(id)
}

/// `name` as a path in the console: as given when absolute, else in its current directory.
async fn console_path(console: &mut ConsoleClient, name: &str) -> anyhow::Result<String> {
    if name.starts_with('/') {
        return Ok(name.to_string());
    }
    let out = console.exec(["pwd"], Some(10_000)).await?;
    let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
    Ok(format!("{}/{name}", dir.trim_end_matches('/')))
}

/// Writes `bytes` to `path` in the console, making its directory if it is not there.
async fn write_console_file(
    console: &mut ConsoleClient,
    path: &str,
    bytes: Vec<u8>,
) -> anyhow::Result<()> {
    if console.write(path, bytes.clone(), None).await.is_ok() {
        return Ok(());
    }
    if let Some((parent, _)) = path.rsplit_once('/').filter(|(p, _)| !p.is_empty()) {
        console.exec(["mkdir", "-p", parent], Some(10_000)).await?;
    }
    console.write(path, bytes, None).await?;
    Ok(())
}

/// The whole of the console's file at `path`, read as far as its size says.
async fn read_console_file(console: &mut ConsoleClient, path: &str) -> anyhow::Result<Vec<u8>> {
    let mut data = Vec::new();
    loop {
        let read = console.read(path, Some(data.len() as u64), None).await?;
        let size = read.size;
        if read.data.is_empty() {
            break;
        }
        data.extend(read.data);
        if data.len() as u64 >= size {
            break;
        }
    }
    Ok(data)
}

/// `n` with commas between thousands.
fn thousands(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Appended to `browser_navigate` unless [`Browser::careful`] is off.
const CAREFUL: &str = " Sites may turn away traffic that looks automated: start from a site's \
     home page or search and go through its links and forms rather than typed-in query URLs, act \
     at a person's pace, and never retry in a loop. On a robot check, stop and tell the user.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_are_for_their_kind_of_browser() {
        assert!(Browser::user().check().is_ok());
        assert!(Browser::user().headless(true).check().is_err());
        assert!(
            Browser::isolated()
                .user_browser(UserBrowser::Chrome)
                .check()
                .is_err()
        );
        assert!(Browser::isolated().profile("/tmp/p").check().is_ok());
    }

    fn description(browser: Browser, name: &str) -> String {
        let session = BrowserSession::new(browser);
        tools(&session, false)
            .into_iter()
            .find(|(d, _)| d.name == name)
            .and_then(|(d, _)| d.description)
            .unwrap()
    }

    #[test]
    fn a_decline_holds_until_the_next_message() {
        let session = BrowserSession::new(Browser::user());
        session.declined.store(true, Ordering::Relaxed);
        assert!(session.declined.load(Ordering::Relaxed));
        session.catch_up();
        assert!(!session.declined.load(Ordering::Relaxed));
    }

    #[test]
    fn careful_is_on_unless_turned_off() {
        assert!(description(Browser::user(), "browser_navigate").contains(CAREFUL));
        let careless = description(Browser::user().careful(false), "browser_navigate");
        assert!(careless.contains("user's own") && !careless.contains(CAREFUL));
    }

    #[test]
    fn every_tool_is_named_once() {
        let session = BrowserSession::new(Browser::user());
        let names: Vec<String> = tools(&session, false)
            .into_iter()
            .map(|(d, _)| d.name)
            .collect();
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
            // Downloads land in the profile, not the downloads folder of whoever runs this.
            let downloads = dir.path().join("downloads");
            std::fs::create_dir_all(dir.path().join("Default")).unwrap();
            std::fs::write(
                dir.path().join("Default/Preferences"),
                serde_json::json!({ "download": { "default_directory": downloads, "prompt_for_download": false } })
                    .to_string(),
            )
            .unwrap();
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
        let end = line[at..].find(']').unwrap();
        line[at..at + end].to_string()
    }

    /// Every tool, on a real Chrome lent as the user's browser.
    #[tokio::test]
    #[ignore = "needs Chrome"]
    async fn the_tools_drive_a_chrome() {
        let chrome = TestChrome::start();
        let html = "<title>Form</title>\
            <form onsubmit=\"document.title='sent:'+q.value;return false\">\
            <input id=q aria-label=Query><button>Search</button></form>\
            <input type=date aria-label=When oninput=\"this.dataset.heard=this.value\">\
            <textarea aria-label=Note>old</textarea>\
            <div style=position:relative><button onclick=\"this.textContent='under hit'\">Under</button>\
            <div style=position:absolute;inset:0></div></div>\
            <div id=box style=height:40px;overflow:auto><p style=height:400px>Inside</p></div>\
            <select aria-label=Size><option value=s>Small</option><option value=l>Large</option></select>\
            <ul role=listbox aria-label=Color><li role=option aria-selected=true>Red</li>\
            <li role=option data-value=b onclick=\"this.textContent='Blue picked'\">Blue</li></ul>\
            <button onclick=\"this.textContent=confirm('Sure?')?'yes':'no'\">Ask</button>\
            <button onclick=\"window.open('about:blank#popup')\">Pop</button>\
            <a download=notes.txt href=\"data:text/plain,hello\">Get notes</a>\
            <button onclick=\"window.print()\">Print</button>\
            <a href=\"https://example.com/far\">Far link</a>\
            <p>Price $29.99 today</p>\
            <div style=height:3000px></div><button onclick=\"this.textContent='pressed'\">Far</button>";
        let page = format!("data:text/html,{}", urlencoding::encode(html));
        let session = BrowserSession::new(
            Browser::user()
                .user_browser(UserBrowser::Custom(chrome.dir.path().into()))
                .start_url(&page),
        );
        session.start().await.unwrap();
        let tools = tools(&session, false);
        let call = |name: &'static str, args: serde_json::Value| {
            let tools = &tools;
            async move { call(tools, name, args).await }
        };

        let (snapshot, _) = call("browser_snapshot", serde_json::json!({ "urls": true })).await;
        assert!(
            snapshot.contains("\"Form\"") && snapshot.contains("refs"),
            "{snapshot}"
        );
        assert!(
            snapshot.contains("-> https://example.com/far"),
            "{snapshot}"
        );
        assert!(
            snapshot.contains("pages above") && snapshot.contains("scroll down for more"),
            "{snapshot}"
        );
        let query = ref_of(&snapshot, "textbox \"Query\"");
        let (typed, _) = call(
            "browser_input",
            serde_json::json!({ "ref": query, "text": "hello 한글" }),
        )
        .await;
        assert!(typed.starts_with("Typed 'hello 한글'."), "{typed}");
        let (sent, _) = call("browser_send_keys", serde_json::json!({ "keys": "Enter" })).await;
        assert!(sent.contains("sent:hello 한글"), "{sent}");

        let when = ref_of(&snapshot, "\"When\"");
        let (typed, _) = call(
            "browser_input",
            serde_json::json!({ "ref": when, "text": "2026-10-20" }),
        )
        .await;
        assert!(typed.starts_with("Typed '2026-10-20'."), "{typed}");
        let note = ref_of(&snapshot, "textbox \"Note\"");
        call(
            "browser_input",
            serde_json::json!({ "ref": note, "text": "A+b" }),
        )
        .await;
        call("browser_send_keys", serde_json::json!({ "keys": "End" })).await;
        let (sent, _) = call(
            "browser_send_keys",
            serde_json::json!({ "keys": " typed!" }),
        )
        .await;
        assert!(sent.starts_with("Sent keys:  typed!."), "{sent}");
        let (read, _) = call(
            "browser_evaluate",
            serde_json::json!({ "code": "[document.querySelector('textarea').value, document.querySelector('[type=date]').dataset.heard]" }),
        )
        .await;
        assert!(
            read.contains("A+b typed!") && read.contains("2026-10-20"),
            "{read}"
        );

        let size = ref_of(&snapshot, "combobox \"Size\"");
        let (options, _) = call(
            "browser_dropdown_options",
            serde_json::json!({ "ref": size }),
        )
        .await;
        assert!(
            options.contains("1: text=\"Large\", value=\"l\"")
                && options.contains("text=\"Small\", value=\"s\" (selected)"),
            "{options}"
        );
        let (picked, _) = call(
            "browser_select_dropdown",
            serde_json::json!({ "ref": size, "text": "large" }),
        )
        .await;
        assert!(picked.contains("Selected Large (value l)"), "{picked}");

        let color = ref_of(&snapshot, "listbox \"Color\"");
        let (options, _) = call(
            "browser_dropdown_options",
            serde_json::json!({ "ref": color }),
        )
        .await;
        assert!(
            options.contains("0: text=\"Red\", value=\"Red\" (selected)")
                && options.contains("1: text=\"Blue\", value=\"b\""),
            "{options}"
        );
        let (picked, _) = call(
            "browser_select_dropdown",
            serde_json::json!({ "ref": color, "text": "b" }),
        )
        .await;
        assert!(picked.starts_with("Selected Blue"), "{picked}");
        let (got, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "link \"Get notes\"") }),
        )
        .await;
        assert!(
            got.contains("Downloaded file: notes.txt (5 bytes) saved to"),
            "{got}"
        );
        let (printed, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Print\"") }),
        )
        .await;
        assert!(printed.contains("browser_save_as_pdf"), "{printed}");
        let (shortcut, _) = call("browser_click", serde_json::json!({ "ref": size })).await;
        assert!(
            shortcut.starts_with("Found a <select> dropdown"),
            "{shortcut}"
        );
        let (under, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Under\"") }),
        )
        .await;
        assert!(under.starts_with("Clicked button \"Under\""), "{under}");

        let boxed = snapshot
            .lines()
            .find(|l| l.contains("[scroll:"))
            .unwrap_or_else(|| panic!("no box that scrolls in\n{snapshot}"));
        let boxed = boxed[boxed.find("[ref=").unwrap() + 5..]
            .trim_end_matches(']')
            .to_string();
        let (scrolled, _) = call(
            "browser_scroll",
            serde_json::json!({ "ref": boxed, "pages": 0.5 }),
        )
        .await;
        assert!(scrolled.starts_with("Scrolled down element"), "{scrolled}");
        tokio::time::sleep(Duration::from_millis(300)).await;
        let (top, _) = call(
            "browser_evaluate",
            serde_json::json!({ "code": "document.getElementById('box').scrollTop" }),
        )
        .await;
        let top: f64 = top.split_whitespace().next().unwrap().parse().unwrap();
        assert!(top > 0.0, "the box scrolled");

        let (found, _) = call(
            "browser_search_page",
            serde_json::json!({ "pattern": "Blue picked|under hit", "regex": true }),
        )
        .await;
        assert!(
            found.starts_with("Found 2 matches for \"Blue picked|under hit\"")
                && found.contains("(in div > button)"),
            "{found}"
        );

        let (found, _) = call(
            "browser_search_page",
            serde_json::json!({ "pattern": "$29.99" }),
        )
        .await;
        assert!(
            found.starts_with("Found 1 match for") && found.contains("Price $29.99 today"),
            "{found}"
        );
        let (elements, _) = call(
            "browser_find_elements",
            serde_json::json!({ "selector": "a", "attributes": ["href"] }),
        )
        .await;
        assert!(
            elements.starts_with("Found 2 elements matching \"a\"")
                && elements.contains(
                    "[1] <a> \"Far link\" {href=\"https://example.com/far\"} (0 children)"
                ),
            "{elements}"
        );

        let (far, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Far\"") }),
        )
        .await;
        assert!(far.starts_with("Clicked button \"Far\"."), "{far}");
        let (text, _) = call("browser_evaluate", serde_json::json!({ "code": "return [...document.querySelectorAll('button')].pop().textContent" })).await;
        assert!(text.starts_with("pressed"), "{text}");
        let (to_text, _) = call(
            "browser_find_text",
            serde_json::json!({ "text": "Price $29" }),
        )
        .await;
        assert!(
            to_text.starts_with("Scrolled to text: Price $29."),
            "{to_text}"
        );

        // A confirm holds the page until the agent answers it.
        let (asked, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Ask\"") }),
        )
        .await;
        assert!(
            asked.contains("confirm dialog") && asked.contains("Sure?"),
            "{asked}"
        );
        let (refused, _) = call("browser_snapshot", serde_json::json!({})).await;
        assert!(refused.contains("browser_dialog"), "{refused}");
        let (answered, _) = call("browser_dialog", serde_json::json!({ "accept": true })).await;
        assert!(answered.starts_with("Accepted the confirm"), "{answered}");
        let (after, _) = call(
            "browser_snapshot",
            serde_json::json!({ "interactive": true }),
        )
        .await;
        assert!(after.contains("button \"yes\""), "{after}");

        // Clicked by coordinates in the screenshot, the result says what was there.
        let (shot, image) = call("browser_screenshot", serde_json::json!({})).await;
        assert!(
            image && shot.contains("Screenshot of the viewport"),
            "{shot}"
        );
        let yes = call("browser_evaluate", serde_json::json!({ "code": "const r = [...document.querySelectorAll('button')].find(b => b.textContent === 'yes').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2, innerWidth]" })).await.0;
        let yes: Vec<f64> = serde_json::from_str(&yes[..yes.find(']').unwrap() + 1]).unwrap();
        let scale = shot
            .split("viewport, ")
            .nth(1)
            .and_then(|s| s.split('x').next())
            .and_then(|w| w.parse::<f64>().ok())
            .unwrap()
            / yes[2];
        let (clicked, _) = call(
            "browser_click",
            serde_json::json!({ "x": yes[0] * scale, "y": yes[1] * scale }),
        )
        .await;
        assert!(
            clicked.starts_with("Clicked on coordinate") && clicked.contains("confirm dialog"),
            "{clicked}"
        );
        // That button asks again.
        let (dismissed, _) = call("browser_dialog", serde_json::json!({ "accept": false })).await;
        assert!(
            dismissed.starts_with("Dismissed the confirm"),
            "{dismissed}"
        );

        // A tab the agent's page opens becomes its own, and current.
        let (popped, _) = call(
            "browser_click",
            serde_json::json!({ "ref": ref_of(&after, "button \"Pop\"") }),
        )
        .await;
        assert!(popped.contains("new tab"), "{popped}");
        let (listed, _) = call("browser_tabs", serde_json::json!({})).await;
        assert_eq!(listed.lines().count(), 2, "{listed}");
        let popup = listed.lines().find(|l| l.contains("#popup")).unwrap();
        assert!(popup.contains("(current)"), "{listed}");
        let popup_id = popup.split_whitespace().next().unwrap();
        let (closed, _) = call(
            "browser_close_tab",
            serde_json::json!({ "tab_id": popup_id }),
        )
        .await;
        assert!(closed.starts_with("Closed."), "{closed}");
        let (opened, _) = call(
            "browser_navigate",
            serde_json::json!({ "url": "about:blank#second", "new_tab": true }),
        )
        .await;
        assert!(opened.starts_with("Opened new tab with URL"), "{opened}");
        let (listed, _) = call("browser_tabs", serde_json::json!({})).await;
        let first = listed.lines().find(|l| !l.contains("(current)")).unwrap();
        let (switched, _) = call(
            "browser_switch_tab",
            serde_json::json!({ "tab_id": first.split_whitespace().next().unwrap() }),
        )
        .await;
        assert!(
            switched.contains("\"Form\"") || switched.contains("sent:"),
            "{switched}"
        );

        let (stale, _) = call("browser_click", serde_json::json!({ "ref": "e999" })).await;
        assert!(stale.contains("take a new `browser_snapshot`"), "{stale}");
        let (scrolled, _) = call(
            "browser_scroll",
            serde_json::json!({ "down": false, "pages": 0.5 }),
        )
        .await;
        assert!(scrolled.starts_with("Scrolled up"), "{scrolled}");
        let (waited, _) = call(
            "browser_wait",
            serde_json::json!({ "text": "pressed", "seconds": 2 }),
        )
        .await;
        assert!(waited.contains("is on the page"), "{waited}");

        let (drawn, image) = call(
            "browser_evaluate",
            serde_json::json!({ "code": "(() => { const c = document.createElement('canvas'); c.width = c.height = 2; return c.toDataURL(); })()" }),
        )
        .await;
        assert!(drawn.starts_with("[Image]") && image, "{drawn}");
        call(
            "browser_navigate",
            serde_json::json!({ "url": "about:blank#next" }),
        )
        .await;
        let (back, _) = call("browser_back", serde_json::json!({})).await;
        assert!(
            back.starts_with("Navigated back.") && back.contains("Form"),
            "{back}"
        );
        drop(chrome);
    }

    /// An isolated browser (one installed here) starts on first use, on its own profile, and
    /// stops with the agent.
    #[tokio::test]
    #[ignore = "needs Chrome"]
    async fn an_isolated_browser_starts_and_stops_with_the_agent() {
        let profile = tempfile::tempdir().unwrap();
        let session =
            BrowserSession::new(Browser::isolated().headless(true).profile(profile.path()));
        let tools = tools(&session, false);
        let (opened, _) = call(
            &tools,
            "browser_navigate",
            serde_json::json!({ "url": "data:text/html,<title>Own</title><button>Go</button>" }),
        )
        .await;
        assert!(opened.contains("\"Own\""), "{opened}");
        let (tabs, _) = call(&tools, "browser_tabs", serde_json::json!({})).await;
        assert_eq!(
            tabs.lines().filter(|l| l.contains("(current)")).count(),
            1,
            "{tabs}"
        );
        let port = std::fs::read_to_string(profile.path().join("DevToolsActivePort")).unwrap();
        let port: u16 = port.lines().next().unwrap().parse().unwrap();
        drop(tools);
        drop(session);
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            std::net::TcpStream::connect(("127.0.0.1", port)).is_err(),
            "the browser closed with the agent"
        );
    }

    /// A page with a frame of its own origin and one of another site, served here.
    async fn serve_frames() -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                tokio::spawn(async move {
                    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
                    let mut buf = vec![0; 4096];
                    let n = stream.read(&mut buf).await.unwrap_or(0);
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let body = if request.starts_with("GET /frame") {
                        "<button onclick=\"this.textContent='hit'\">Inner</button>".to_string()
                    } else {
                        // `localhost` is another site than `127.0.0.1`, so its frame is a process of its own.
                        format!(
                            "<title>Frames</title><h1>Outer</h1>\
                             <iframe srcdoc=\"<button>Same</button>\"></iframe>\
                             <iframe src=\"http://localhost:{port}/frame\" style=\"width:300px;height:100px\"></iframe>"
                        )
                    };
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: text/html\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        port
    }

    /// Frames appear in a snapshot, and a ref inside another site's frame is clicked where it is.
    #[tokio::test]
    #[ignore = "needs Chrome"]
    async fn a_snapshot_reads_into_frames() {
        let chrome = TestChrome::start();
        let port = serve_frames().await;
        let session = BrowserSession::new(
            Browser::user()
                .user_browser(UserBrowser::Custom(chrome.dir.path().into()))
                .start_url(format!("http://127.0.0.1:{port}/")),
        );
        session.start().await.unwrap();
        let tools = tools(&session, false);
        tokio::time::sleep(Duration::from_millis(500)).await;
        let (snapshot, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        assert!(snapshot.contains("button \"Same\""), "{snapshot}");
        assert!(snapshot.contains("button \"Inner\""), "{snapshot}");
        let (clicked, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "button \"Inner\"") }),
        )
        .await;
        assert!(clicked.starts_with("Clicked button \"Inner\""), "{clicked}");
        let (after, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        assert!(after.contains("button \"hit\""), "{after}");

        let (scoped, _) = call(
            &tools,
            "browser_snapshot",
            serde_json::json!({ "selector": "h1" }),
        )
        .await;
        assert!(
            scoped.contains("heading \"Outer\"") && !scoped.contains("Same"),
            "{scoped}"
        );
        let (shallow, _) = call(
            &tools,
            "browser_snapshot",
            serde_json::json!({ "depth": 1 }),
        )
        .await;
        assert!(!shallow.contains("Inner"), "{shallow}");
        drop(chrome);
    }

    /// The file tools keep what they take in the console, and upload from it.
    #[tokio::test]
    #[ignore = "needs Chrome and a console server"]
    async fn the_file_tools_go_through_the_console() {
        let chrome = TestChrome::start();
        let html = "<title>Files</title><div><button>Choose a file</button>\
            <input type=file hidden onchange=\"document.title='got:'+this.files[0].name+':'+this.files[0].size\"></div>\
            <a download=notes.txt href=\"data:text/plain,hello%20world\">Get notes</a>";
        let session = BrowserSession::new(
            Browser::user()
                .user_browser(UserBrowser::Custom(chrome.dir.path().into()))
                .start_url(format!("data:text/html,{}", urlencoding::encode(html))),
        );
        session.start().await.unwrap();
        let tools = tools(&session, true);
        let mut console = crate::test_console().await;
        let by_console = |name: &str, args: serde_json::Value| {
            let (_, func) = tools.iter().find(|(d, _)| d.name == name).unwrap();
            let args: Value = serde_json::from_value(args).unwrap();
            (func.clone(), args)
        };
        async fn run(func: ToolFunc, args: Value, console: &mut ConsoleClient) -> String {
            use futures::StreamExt as _;
            let mut stream = func.call(args, "1", console);
            let mut last = None;
            while let Some(output) = stream.next().await {
                last = Some(output.message);
            }
            last.unwrap()
                .contents
                .iter()
                .filter_map(|p| p.as_text())
                .collect::<Vec<_>>()
                .join("\n")
        }

        let (f, a) = by_console(
            "browser_save_screenshot",
            serde_json::json!({ "file_name": "/tmp/shots/page.png" }),
        );
        let saved = run(f, a, &mut console).await;
        assert!(saved.contains("/tmp/shots/page.png"), "{saved}");
        let png = console
            .read("/tmp/shots/page.png", None, Some(8))
            .await
            .unwrap();
        assert_eq!(&png.data[..4], b"\x89PNG");
        let (f, a) = by_console(
            "browser_save_as_pdf",
            serde_json::json!({ "file_name": "page.pdf" }),
        );
        let saved = run(f, a, &mut console).await;
        assert!(
            saved.starts_with("Saved page as PDF:") && saved.contains("page.pdf"),
            "{saved}"
        );
        for named in ["Files.pdf", "Files (1).pdf"] {
            let (f, a) = by_console("browser_save_as_pdf", serde_json::json!({}));
            let saved = run(f, a, &mut console).await;
            assert!(
                saved.contains(named),
                "named after the title, numbered when taken: {saved}"
            );
        }

        let (snapshot, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        let (got, _) = call(
            &tools,
            "browser_click",
            serde_json::json!({ "ref": ref_of(&snapshot, "link \"Get notes\"") }),
        )
        .await;
        assert!(got.contains("`browser_save_download`"), "{got}");
        let (f, a) = by_console(
            "browser_save_download",
            serde_json::json!({ "name": "notes.txt", "file_name": "/tmp/dl/notes.txt" }),
        );
        let copied = run(f, a, &mut console).await;
        assert!(
            copied.contains("(11 bytes) to /tmp/dl/notes.txt"),
            "{copied}"
        );
        let read = console.read("/tmp/dl/notes.txt", None, None).await.unwrap();
        assert_eq!(read.data, b"hello world");
        let (f, a) = by_console(
            "browser_save_download",
            serde_json::json!({ "name": "/etc/hosts" }),
        );
        let refused = run(f, a, &mut console).await;
        assert!(refused.contains("no download named"), "{refused}");
        // In pieces, as a large file goes.
        let host = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(host.path(), b"0123456789").unwrap();
        copy_to_console(&mut console, host.path(), "/tmp/dl/pieces.txt", 3)
            .await
            .unwrap();
        let read = console
            .read("/tmp/dl/pieces.txt", None, None)
            .await
            .unwrap();
        assert_eq!(read.data, b"0123456789");

        console
            .write("/tmp/note.txt", b"hello".to_vec(), None)
            .await
            .unwrap();
        let (snapshot, _) = call(&tools, "browser_snapshot", serde_json::json!({})).await;
        let input = snapshot
            .lines()
            .find(|l| l.contains("[ref="))
            .and_then(|l| l.split("[ref=").nth(1))
            .map(|r| r.trim_end_matches(']').to_string())
            .unwrap_or_else(|| panic!("no ref in\n{snapshot}"));
        let (f, a) = by_console(
            "browser_upload_file",
            serde_json::json!({ "ref": input, "path": "/tmp/note.txt" }),
        );
        let uploaded = run(f, a, &mut console).await;
        assert!(uploaded.contains("got:note.txt:5"), "{uploaded}");
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
                Browser::user()
                    .user_browser(UserBrowser::Custom(chrome.dir.path().into()))
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
        assert!(called.iter().any(|n| n == "browser_input"), "{called:?}");
        assert!(
            answer.contains("Glide") && answer.contains("Lite") && !answer.contains("Pro"),
            "{answer}"
        );
        // Applied: the button wrote the coupon into the title.
        let session = agent.state.browser.clone().unwrap();
        let (title, _) = call(
            &tools(&session, false),
            "browser_evaluate",
            serde_json::json!({ "code": "document.title" }),
        )
        .await;
        assert!(title.starts_with("code:SAVE10"), "{title}");
        drop(chrome);
    }
}
