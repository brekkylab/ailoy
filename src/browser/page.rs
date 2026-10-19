//! The agent's tabs over one [`Cdp`] connection, and what the browser tools do in them.
//!
//! Only tabs the agent opened, or that opened from one of its own, are reachable: whatever
//! else the browser has (the person's tabs) is not in [`Tabs`] and no tool can name it.

use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context as _, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

use super::{
    cdp::{CallError, Cdp},
    snapshot,
};

/// How long an action waits for the page it started to load.
const LOAD_WAIT: Duration = Duration::from_secs(15);

struct Tab {
    target: String,
    session: String,
}

pub(super) struct Tabs {
    cdp: Arc<Cdp>,
    tabs: Vec<Tab>,
    current: usize,
    /// From the last snapshot of the current tab, while it shows the document they were taken
    /// from.
    refs: HashMap<String, i64>,
    /// That document, as its loader.
    refs_loader: Option<String>,
}

/// What a tool says back: text, and a screenshot when one was taken.
pub(super) struct Said {
    pub text: String,
    pub image: Option<Vec<u8>>,
}

impl From<String> for Said {
    fn from(text: String) -> Self {
        Self { text, image: None }
    }
}

impl Tabs {
    /// The agent's tabs on `cdp`, of which `targets` are those it has; the first is current.
    pub(super) async fn new(cdp: Arc<Cdp>, targets: &[String]) -> anyhow::Result<Self> {
        let mut tabs = Self {
            cdp,
            tabs: Vec::new(),
            current: 0,
            refs: HashMap::new(),
            refs_loader: None,
        };
        for target in targets {
            tabs.adopt(target).await?;
        }
        tabs.cdp
            .call(
                "Target.setDiscoverTargets",
                json!({ "discover": true }),
                None,
            )
            .await?;
        Ok(tabs)
    }

    pub(super) fn is_alive(&self) -> bool {
        self.cdp.is_alive()
    }

    pub(super) fn targets(&self) -> Vec<String> {
        self.tabs.iter().map(|t| t.target.clone()).collect()
    }

    /// Attaches to `target` and makes it the agent's current tab.
    async fn adopt(&mut self, target: &str) -> anyhow::Result<()> {
        let attached = self
            .cdp
            .call(
                "Target.attachToTarget",
                json!({ "targetId": target, "flatten": true }),
                None,
            )
            .await?;
        let session = attached["sessionId"]
            .as_str()
            .context("attaching gave no session")?
            .to_string();
        for domain in ["Page.enable", "DOM.enable", "Accessibility.enable"] {
            self.cdp.call(domain, json!({}), Some(&session)).await?;
        }
        self.tabs.push(Tab {
            target: target.to_string(),
            session,
        });
        self.current = self.tabs.len() - 1;
        self.refs.clear();
        Ok(())
    }

    fn session(&self) -> &str {
        &self.tabs[self.current].session
    }

    async fn page(&self, method: &str, params: Value) -> Result<Value, CallError> {
        self.cdp.call(method, params, Some(self.session())).await
    }

    /// Page tabs whose opener is one of the agent's, adopted; the last becomes current.
    async fn adopt_opened(&mut self) -> anyhow::Result<bool> {
        let all = self.cdp.call("Target.getTargets", json!({}), None).await?;
        let mut adopted = false;
        for info in all["targetInfos"].as_array().into_iter().flatten() {
            let (Some(target), Some(opener)) =
                (info["targetId"].as_str(), info["openerId"].as_str())
            else {
                continue;
            };
            if info["type"] == "page"
                && self.tabs.iter().any(|t| t.target == opener)
                && !self.tabs.iter().any(|t| t.target == target)
            {
                self.adopt(target).await?;
                adopted = true;
            }
        }
        if adopted {
            self.activate().await;
        }
        Ok(adopted)
    }

    /// Brings the current tab to the front, where the person sees it.
    async fn activate(&self) {
        let target = self.tabs[self.current].target.clone();
        let _ = self
            .cdp
            .call("Target.activateTarget", json!({ "targetId": target }), None)
            .await;
    }

    async fn eval(&self, expression: &str) -> Result<Value, CallError> {
        let answer = self
            .page(
                "Runtime.evaluate",
                json!({ "expression": expression, "returnByValue": true, "awaitPromise": true }),
            )
            .await?;
        if let Some(exception) = answer.get("exceptionDetails") {
            let text = exception["exception"]["description"]
                .as_str()
                .or(exception["text"].as_str())
                .unwrap_or("exception");
            return Err(CallError::Browser(text.to_string()));
        }
        Ok(answer["result"]["value"].clone())
    }

    /// Waits for the current tab to finish loading, a short settle first so a navigation an
    /// action started has begun.
    async fn settle(&self) {
        tokio::time::sleep(Duration::from_millis(400)).await;
        let deadline = tokio::time::Instant::now() + LOAD_WAIT;
        while tokio::time::Instant::now() < deadline {
            match self.eval("document.readyState").await {
                Ok(state) if state == "complete" => return,
                Err(CallError::Dialog | CallError::Gone(_)) => return,
                _ => tokio::time::sleep(Duration::from_millis(250)).await,
            }
        }
    }

    /// Where the current tab is, as one line.
    async fn whereabouts(&self) -> String {
        match self.eval("[location.href, document.title]").await {
            Ok(Value::Array(v)) => {
                // A data: or tracking-laden URL can run to kilobytes, which say nothing more.
                let url = v[0].as_str().unwrap_or_default();
                let url = match url.char_indices().nth(120) {
                    Some((at, _)) => format!("{}…", &url[..at]),
                    None => url.to_string(),
                };
                format!("Page {:?} at {url}.", v[1].as_str().unwrap_or_default())
            }
            _ => "The page is still loading.".to_string(),
        }
    }

    /// After an action: new tabs adopted, dialogs told, and where the agent is now.
    async fn after(&mut self, done: &str, result: Result<(), CallError>) -> anyhow::Result<Said> {
        let mut text = done.to_string();
        match result {
            Ok(()) => {}
            Err(CallError::Dialog) => {}
            Err(e) => bail!("{e}"),
        }
        // A dialog holds the page where it is, and the page answers nothing until it closes.
        if self.cdp.dialog(self.session()).is_none() {
            self.settle().await;
            if self.adopt_opened().await? {
                text.push_str(" It opened a new tab, which you are on now.");
                self.settle().await;
            }
            if self.loader().await != self.refs_loader {
                self.refs.clear();
            }
        }
        Ok(self.with_page_news(text).await.into())
    }

    /// The loader of the current tab's document, which a new document changes.
    async fn loader(&self) -> Option<String> {
        let tree = self.page("Page.getFrameTree", json!({})).await.ok()?;
        tree["frameTree"]["frame"]["loaderId"]
            .as_str()
            .map(str::to_string)
    }

    /// `text`, with what the page did besides: dialogs accepted, one waiting, or where it is.
    async fn with_page_news(&self, mut text: String) -> String {
        for alert in self.cdp.take_accepted(self.session()) {
            text.push_str(&format!(
                " The page showed {} {:?}, which was accepted.",
                article(&alert.kind),
                alert.message
            ));
        }
        match self.cdp.dialog(self.session()) {
            Some(dialog) => text.push_str(&format!(
                " The page is waiting on {} dialog: {:?}. Answer it with `browser_dialog` before anything else.",
                article(&dialog.kind),
                dialog.message
            )),
            None => {
                text.push(' ');
                text.push_str(&self.whereabouts().await);
            }
        }
        text
    }

    /// Refuses an action while a dialog holds the page.
    fn no_dialog(&self) -> anyhow::Result<()> {
        if let Some(dialog) = self.cdp.dialog(self.session()) {
            bail!(
                "the page is waiting on {} dialog ({:?}): answer it with `browser_dialog` first",
                article(&dialog.kind),
                dialog.message
            );
        }
        Ok(())
    }

    fn node(&self, r#ref: &str) -> anyhow::Result<i64> {
        let r#ref = r#ref.trim_start_matches('@');
        self.refs.get(r#ref).copied().with_context(|| {
            format!(
                "no ref {ref:?} on this page now: take a new `browser_snapshot` and use its refs"
            )
        })
    }

    // ---- the tools --------------------------------------------------------------------

    pub(super) async fn navigate(&mut self, url: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let scheme = ["about:", "data:", "file:", "blob:"];
        let url = if url.contains("://") || scheme.iter().any(|s| url.starts_with(s)) {
            url.to_string()
        } else {
            format!("https://{url}")
        };
        let mut tries = 0;
        let result = loop {
            let result = self
                .page("Page.navigate", json!({ "url": url }))
                .await
                .and_then(|answer| match answer["errorText"].as_str() {
                    Some(e) if !e.is_empty() => Err(CallError::Browser(format!("{url}: {e}"))),
                    _ => Ok(()),
                });
            // A browser just booted may still be bringing its network up.
            match result {
                Err(CallError::Browser(e)) if e.ends_with("ERR_NETWORK_CHANGED") && tries < 3 => {
                    tries += 1;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                result => break result,
            }
        };
        self.after("Opened.", result).await
    }

    pub(super) async fn back(&mut self) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let result = self.eval("history.back()").await.map(|_| ());
        self.after("Went back.", result).await
    }

    pub(super) async fn snapshot(
        &mut self,
        interactive: bool,
        max_chars: usize,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let tree = self.page("Accessibility.getFullAXTree", json!({})).await?;
        let nodes = tree["nodes"].as_array().cloned().unwrap_or_default();
        let snapshot = snapshot::render(&nodes, interactive);
        self.refs = snapshot.refs;
        self.refs_loader = self.loader().await;
        let head = self.whereabouts().await;
        let body = if snapshot.text.is_empty() {
            "(nothing on the page yet)".to_string()
        } else {
            snapshot::cut(&snapshot.text, max_chars)
        };
        Ok(format!("{head}\n\n{body}").into())
    }

    /// The middle of `node` on screen, scrolled into view.
    async fn center(&self, node: i64) -> Option<(f64, f64)> {
        let _ = self
            .page(
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": node }),
            )
            .await;
        let quads = self
            .page("DOM.getContentQuads", json!({ "backendNodeId": node }))
            .await
            .ok()?;
        let quad = quads["quads"].as_array()?.first()?.as_array()?;
        let at = |i: usize| quad.get(i).and_then(Value::as_f64);
        let xs = [at(0)?, at(2)?, at(4)?, at(6)?];
        let ys = [at(1)?, at(3)?, at(5)?, at(7)?];
        Some((xs.iter().sum::<f64>() / 4.0, ys.iter().sum::<f64>() / 4.0))
    }

    /// Calls `function` (JavaScript source) with `this` as `node`.
    async fn call_on(&self, node: i64, function: &str, args: Value) -> Result<Value, CallError> {
        let resolved = self
            .page("DOM.resolveNode", json!({ "backendNodeId": node }))
            .await?;
        let object = resolved["object"]["objectId"].clone();
        let answer = self
            .page(
                "Runtime.callFunctionOn",
                json!({
                    "objectId": object,
                    "functionDeclaration": function,
                    "arguments": args,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
            )
            .await?;
        if let Some(exception) = answer.get("exceptionDetails") {
            let text = exception["exception"]["description"]
                .as_str()
                .unwrap_or("exception");
            return Err(CallError::Browser(text.to_string()));
        }
        Ok(answer["result"]["value"].clone())
    }

    pub(super) async fn click(&mut self, r#ref: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let result = match self.center(node).await {
            Some((x, y)) => self.mouse_click(x, y).await,
            // Not laid out as a box (hidden, zero-sized): clicked as script would.
            None => self
                .call_on(node, "function() { this.click(); }", json!([]))
                .await
                .map(|_| ()),
        };
        self.after("Clicked.", result).await
    }

    async fn mouse_click(&self, x: f64, y: f64) -> Result<(), CallError> {
        self.page(
            "Input.dispatchMouseEvent",
            json!({ "type": "mouseMoved", "x": x, "y": y }),
        )
        .await?;
        for kind in ["mousePressed", "mouseReleased"] {
            self.page(
                "Input.dispatchMouseEvent",
                json!({ "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1 }),
            )
            .await?;
        }
        Ok(())
    }

    pub(super) async fn type_text(
        &mut self,
        r#ref: &str,
        text: &str,
        clear: bool,
        submit: bool,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let result = async {
            if let Some((x, y)) = self.center(node).await {
                self.mouse_click(x, y).await?;
            }
            self.page("DOM.focus", json!({ "backendNodeId": node }))
                .await?;
            if clear {
                self.call_on(
                    node,
                    "function() { if (typeof this.select === 'function') this.select(); \
                     else getSelection().selectAllChildren(this); }",
                    json!([]),
                )
                .await?;
            }
            self.page("Input.insertText", json!({ "text": text }))
                .await?;
            if submit {
                self.key("Enter").await?;
            }
            Ok(())
        }
        .await;
        let done = if submit {
            "Typed and submitted."
        } else {
            "Typed."
        };
        self.after(done, result).await
    }

    pub(super) async fn select(&mut self, r#ref: &str, option: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let picked = self
            .call_on(
                node,
                "function(want) { \
                   if (!this.options) return null; \
                   const o = [...this.options].find(o => o.value === want || o.text.trim() === want) \
                     || [...this.options].find(o => o.text.trim().toLowerCase().includes(want.toLowerCase())); \
                   if (!o) return [...this.options].map(o => o.text.trim()); \
                   this.value = o.value; \
                   this.dispatchEvent(new Event('input', { bubbles: true })); \
                   this.dispatchEvent(new Event('change', { bubbles: true })); \
                   return o.text.trim(); }",
                json!([{ "value": option }]),
            )
            .await?;
        match picked {
            Value::String(text) => self.after(&format!("Selected {text:?}."), Ok(())).await,
            Value::Array(options) => bail!(
                "no option {option:?}; the options are {}",
                serde_json::to_string(&options).unwrap_or_default()
            ),
            _ => bail!("{ref} is not a <select>: click it and pick from what opens instead"),
        }
    }

    async fn key(&self, combo: &str) -> Result<(), CallError> {
        let (modifiers, key) = parse_key(combo);
        let (key, code, key_code, text) = key_info(&key);
        let mut down = json!({
            "type": if text.is_some() && modifiers & !8 == 0 { "keyDown" } else { "rawKeyDown" },
            "key": key, "code": code, "windowsVirtualKeyCode": key_code, "modifiers": modifiers,
        });
        if let Some(text) = &text
            && modifiers & !8 == 0
        {
            down["text"] = json!(text);
        }
        self.page("Input.dispatchKeyEvent", down).await?;
        self.page(
            "Input.dispatchKeyEvent",
            json!({ "type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": key_code, "modifiers": modifiers }),
        )
        .await?;
        Ok(())
    }

    pub(super) async fn press(&mut self, combo: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let result = self.key(combo).await;
        self.after(&format!("Pressed {combo}."), result).await
    }

    pub(super) async fn scroll(&mut self, down: bool, r#ref: Option<&str>) -> anyhow::Result<Said> {
        self.no_dialog()?;
        if let Some(r#ref) = r#ref {
            let node = self.node(r#ref)?;
            self.page(
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": node }),
            )
            .await?;
            return Ok(format!("Scrolled {ref} into view.").into());
        }
        let metrics = self.page("Page.getLayoutMetrics", json!({})).await?;
        let viewport = &metrics["cssVisualViewport"];
        let (w, h) = (
            viewport["clientWidth"].as_f64().unwrap_or(1280.0),
            viewport["clientHeight"].as_f64().unwrap_or(800.0),
        );
        let delta = if down { h * 0.8 } else { -h * 0.8 };
        self.page(
            "Input.dispatchMouseEvent",
            json!({ "type": "mouseWheel", "x": w / 2.0, "y": h / 2.0, "deltaX": 0, "deltaY": delta }),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(300)).await;
        let at = self
            .eval("[Math.round(scrollY), Math.max(0, document.documentElement.scrollHeight - innerHeight)]")
            .await
            .ok();
        let refs_note = "Refs from before still hold; new content needs a new snapshot.";
        Ok(match at {
            Some(Value::Array(v)) => format!(
                "Scrolled {}: at {} of {} px. {refs_note}",
                if down { "down" } else { "up" },
                v[0],
                v[1]
            ),
            _ => format!("Scrolled. {refs_note}"),
        }
        .into())
    }

    pub(super) async fn screenshot(&mut self) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let shot = self
            .page(
                "Page.captureScreenshot",
                json!({ "format": "jpeg", "quality": 75 }),
            )
            .await?;
        let data = BASE64
            .decode(shot["data"].as_str().unwrap_or_default())
            .context("the screenshot was not base64")?;
        Ok(Said {
            text: self.whereabouts().await,
            image: Some(data),
        })
    }

    pub(super) async fn evaluate(&mut self, expression: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        // A statement list or a function body runs as one: wrapped, its last value returned.
        let value = match self.eval(expression).await {
            Err(CallError::Browser(e)) if e.contains("SyntaxError") => {
                self.eval(&format!("(async () => {{ {expression} }})()"))
                    .await?
            }
            other => other?,
        };
        let text = match value {
            Value::String(s) => s,
            Value::Null => "undefined".to_string(),
            other => serde_json::to_string_pretty(&other).unwrap_or_default(),
        };
        Ok(self.with_page_news(text).await.into())
    }

    pub(super) async fn wait(&mut self, seconds: f64, text: Option<&str>) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let limit = Duration::from_secs_f64(seconds.clamp(0.0, 30.0));
        let Some(text) = text else {
            tokio::time::sleep(limit).await;
            return Ok(self.with_page_news("Waited.".into()).await.into());
        };
        let probe = format!(
            "document.body ? document.body.innerText.includes({}) : false",
            serde_json::to_string(text).unwrap_or_default()
        );
        let deadline = tokio::time::Instant::now() + limit;
        loop {
            if self.eval(&probe).await.ok() == Some(Value::Bool(true)) {
                return Ok(self
                    .with_page_news(format!("{text:?} is on the page."))
                    .await
                    .into());
            }
            if tokio::time::Instant::now() >= deadline {
                return Ok(self
                    .with_page_news(format!(
                        "{text:?} did not appear in {}s.",
                        limit.as_secs_f64()
                    ))
                    .await
                    .into());
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
    }

    pub(super) async fn dialog(
        &mut self,
        accept: bool,
        text: Option<&str>,
    ) -> anyhow::Result<Said> {
        let Some(dialog) = self.cdp.dialog(self.session()) else {
            bail!("no dialog is open");
        };
        let mut params = json!({ "accept": accept });
        if let Some(text) = text {
            params["promptText"] = json!(text);
        }
        self.page("Page.handleJavaScriptDialog", params).await?;
        let done = format!(
            "{} the {} {:?}.",
            if accept { "Accepted" } else { "Dismissed" },
            dialog.kind,
            dialog.message
        );
        self.after(&done, Ok(())).await
    }

    pub(super) async fn tabs(
        &mut self,
        action: &str,
        index: Option<usize>,
    ) -> anyhow::Result<Said> {
        self.adopt_opened().await?;
        match (action, index) {
            ("list", _) => {}
            ("new", _) => {
                // Opened from the current tab, so it lands in the agent's window and is its own.
                self.page(
                    "Runtime.evaluate",
                    json!({ "expression": "window.open('about:blank', '_blank')", "userGesture": true }),
                )
                .await?;
                tokio::time::sleep(Duration::from_millis(300)).await;
                if !self.adopt_opened().await? {
                    bail!("the browser opened no tab");
                }
            }
            ("switch", Some(i)) if i < self.tabs.len() => {
                self.current = i;
                self.refs.clear();
                self.activate().await;
            }
            ("close", Some(i)) if i < self.tabs.len() => {
                if self.tabs.len() == 1 {
                    bail!("that is your only tab: navigate it instead");
                }
                let tab = self.tabs.remove(i);
                let _ = self
                    .cdp
                    .call(
                        "Target.closeTarget",
                        json!({ "targetId": tab.target }),
                        None,
                    )
                    .await;
                if self.current >= i && self.current > 0 {
                    self.current -= 1;
                }
                self.refs.clear();
                self.activate().await;
            }
            ("switch" | "close", _) => bail!(
                "`index` is one of the tabs listed: 0 to {}",
                self.tabs.len() - 1
            ),
            (other, _) => bail!("action is list, new, switch or close, not {other:?}"),
        }
        let mut lines = Vec::new();
        for (i, tab) in self.tabs.iter().enumerate() {
            let info = self
                .cdp
                .call(
                    "Target.getTargetInfo",
                    json!({ "targetId": tab.target }),
                    None,
                )
                .await
                .unwrap_or_default();
            let info = &info["targetInfo"];
            lines.push(format!(
                "{i}{} {:?} {}",
                if i == self.current { " (current)" } else { "" },
                info["title"].as_str().unwrap_or_default(),
                info["url"].as_str().unwrap_or_default()
            ));
        }
        Ok(lines.join("\n").into())
    }

    /// Opens `url` in the current tab, for an app starting the browser somewhere.
    pub(super) async fn open(&mut self, url: &str) -> anyhow::Result<()> {
        self.navigate(url).await.map(|_| ())
    }
}

fn article(kind: &str) -> String {
    match kind {
        "alert" => "an alert".into(),
        "beforeunload" => "a leave-page check".into(),
        other => format!("a {other}"),
    }
}

/// `Control+Shift+a` as CDP modifier bits (Alt 1, Control 2, Meta 4, Shift 8) and the key.
fn parse_key(combo: &str) -> (i64, String) {
    let parts: Vec<&str> = combo.split('+').collect();
    let (last, mods) = match parts.split_last() {
        // `Control++` names the plus key.
        Some((&"", rest)) if !rest.is_empty() => ("+", &rest[..rest.len() - 1]),
        Some((last, rest)) => (*last, rest),
        None => (combo, &[][..]),
    };
    let mut bits = 0;
    for m in mods {
        bits |= match m.to_ascii_lowercase().as_str() {
            "alt" | "option" => 1,
            "control" | "ctrl" => 2,
            "meta" | "cmd" | "command" => 4,
            "shift" => 8,
            _ => 0,
        };
    }
    (bits, last.to_string())
}

/// A key's `key`, `code`, Windows key code, and the text it types, if any.
fn key_info(key: &str) -> (String, String, i64, Option<String>) {
    let named = |key: &str, code: &str, n: i64, text: Option<&str>| {
        (
            key.to_string(),
            code.to_string(),
            n,
            text.map(str::to_string),
        )
    };
    match key.to_ascii_lowercase().as_str() {
        "enter" | "return" => named("Enter", "Enter", 13, Some("\r")),
        "tab" => named("Tab", "Tab", 9, None),
        "escape" | "esc" => named("Escape", "Escape", 27, None),
        "backspace" => named("Backspace", "Backspace", 8, None),
        "delete" => named("Delete", "Delete", 46, None),
        "space" | " " => named(" ", "Space", 32, Some(" ")),
        "arrowup" | "up" => named("ArrowUp", "ArrowUp", 38, None),
        "arrowdown" | "down" => named("ArrowDown", "ArrowDown", 40, None),
        "arrowleft" | "left" => named("ArrowLeft", "ArrowLeft", 37, None),
        "arrowright" | "right" => named("ArrowRight", "ArrowRight", 39, None),
        "home" => named("Home", "Home", 36, None),
        "end" => named("End", "End", 35, None),
        "pageup" => named("PageUp", "PageUp", 33, None),
        "pagedown" => named("PageDown", "PageDown", 34, None),
        _ => {
            let mut chars = key.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => {
                    let upper = c.to_ascii_uppercase();
                    let (code, n) = if upper.is_ascii_alphabetic() {
                        (format!("Key{upper}"), upper as i64)
                    } else if c.is_ascii_digit() {
                        (format!("Digit{c}"), c as i64)
                    } else {
                        (String::new(), 0)
                    };
                    (c.to_string(), code, n, Some(c.to_string()))
                }
                _ => (key.to_string(), key.to_string(), 0, None),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_combo_is_modifier_bits_and_a_key() {
        assert_eq!(parse_key("Enter"), (0, "Enter".into()));
        assert_eq!(parse_key("Control+a"), (2, "a".into()));
        assert_eq!(parse_key("Control+Shift+Tab"), (10, "Tab".into()));
        assert_eq!(parse_key("Meta++"), (4, "+".into()));
        assert_eq!(key_info("enter").3.as_deref(), Some("\r"));
        assert_eq!(
            key_info("a"),
            ("a".into(), "KeyA".into(), 65, Some("a".into()))
        );
        assert_eq!(key_info("ArrowDown").2, 40);
    }
}
