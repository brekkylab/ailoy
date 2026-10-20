//! The agent's tabs over one [`Cdp`] connection, and what the browser tools do in them.
//!
//! Only tabs the agent opened, or that opened from one of its own, are reachable: whatever
//! else the browser has (the person's tabs) is not in [`Tabs`] and no tool can name it.

use std::{
    collections::{HashMap, HashSet},
    io::Cursor,
    path::Path,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, bail};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

use super::{
    cdp::{self, CallError, Cdp},
    snapshot,
};

/// How long an action waits for the page it started to load.
const LOAD_WAIT: Duration = Duration::from_secs(15);

/// Widest a screenshot is sent, in pixels: models place coordinates best around this size.
const SHOT_WIDTH: u32 = 1280;

struct Tab {
    target: String,
    session: String,
}

impl Tab {
    /// What the agent calls it: the last four characters of its target id.
    fn id(&self) -> &str {
        &self.target[self.target.len().saturating_sub(4)..]
    }
}

/// Where a ref's node is: the session of its document, and for a frame in a process of its
/// own, the iframe holding it in the page, whose place on screen its coordinates start from.
#[derive(Clone)]
struct Node {
    session: String,
    backend: i64,
    owner: Option<(String, i64)>,
}

/// The last screenshot's size against the viewport it shows, to turn its pixels into the page's.
#[derive(Clone, Copy)]
struct Shot {
    css_width: f64,
    width: f64,
}

pub(super) struct Tabs {
    cdp: Arc<Cdp>,
    tabs: Vec<Tab>,
    current: usize,
    /// From the last snapshot of the current tab, while it shows the document they were taken
    /// from.
    refs: HashMap<String, Node>,
    /// That document, as its loader.
    refs_loader: Option<String>,
    shot: Option<Shot>,
}

/// What a tool says back: text, and a screenshot when one was taken.
pub(super) struct Said {
    pub text: String,
    /// Its media type, and its bytes.
    pub image: Option<(&'static str, Vec<u8>)>,
}

impl From<String> for Said {
    fn from(text: String) -> Self {
        Self { text, image: None }
    }
}

/// What `browser_snapshot` is asked for.
pub(super) struct Look<'a> {
    pub interactive: bool,
    pub depth: Option<usize>,
    pub selector: Option<&'a str>,
    pub urls: bool,
    pub max_chars: usize,
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
            shot: None,
        };
        for target in targets {
            tabs.adopt(target).await?;
        }
        tabs.current = 0;
        tabs.cdp
            .call(
                "Target.setDiscoverTargets",
                json!({ "discover": true }),
                None,
            )
            .await?;
        // Downloads go where the person's settings put them; the browser only reports them.
        let _ = tabs
            .cdp
            .call(
                "Browser.setDownloadBehavior",
                json!({ "behavior": "default", "eventsEnabled": true }),
                None,
            )
            .await;
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
        // Frames in a process of their own get sessions of their own, for snapshots to read.
        self.cdp
            .call(
                "Target.setAutoAttach",
                json!({
                    "autoAttach": true,
                    "waitForDebuggerOnStart": false,
                    "flatten": true,
                    "filter": [{ "type": "iframe" }, { "exclude": true }],
                }),
                Some(&session),
            )
            .await?;
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
        self.eval_in(self.session(), expression).await
    }

    async fn eval_in(&self, session: &str, expression: &str) -> Result<Value, CallError> {
        let answer = self
            .cdp
            .call(
                "Runtime.evaluate",
                json!({ "expression": expression, "returnByValue": true, "awaitPromise": true }),
                Some(session),
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

    fn node(&self, r#ref: &str) -> anyhow::Result<Node> {
        let r#ref = r#ref.trim_start_matches('@');
        self.refs.get(r#ref).cloned().with_context(|| {
            format!(
                "no ref {ref:?} on this page now: take a new `browser_snapshot` and use its refs"
            )
        })
    }

    /// The top-left of a frame's content on screen, from the iframe holding it.
    async fn frame_origin(&self, owner: &(String, i64)) -> Option<(f64, f64)> {
        let (session, backend) = owner;
        let _ = self
            .cdp
            .call(
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": backend }),
                Some(session),
            )
            .await;
        let quads = self
            .cdp
            .call(
                "DOM.getContentQuads",
                json!({ "backendNodeId": backend }),
                Some(session),
            )
            .await
            .ok()?;
        let quad = quads["quads"].as_array()?.first()?.as_array()?;
        Some((quad.first()?.as_f64()?, quad.get(1)?.as_f64()?))
    }

    /// The middle of `node` on screen, scrolled into view.
    async fn center(&self, node: &Node) -> Option<(f64, f64)> {
        self.centers(node).await.map(|(at, _)| at)
    }

    /// The middle of `node`, scrolled into view: on screen, and in its own frame.
    async fn centers(&self, node: &Node) -> Option<((f64, f64), (f64, f64))> {
        let origin = match &node.owner {
            Some(owner) => self.frame_origin(owner).await?,
            None => (0.0, 0.0),
        };
        let _ = self
            .cdp
            .call(
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": node.backend }),
                Some(&node.session),
            )
            .await;
        let (x, y, w, h) = self.rect(node).await?;
        let local = (x + w / 2.0, y + h / 2.0);
        Some(((origin.0 + local.0, origin.1 + local.1), local))
    }

    /// `node`'s box in its frame as x, y, width, height: from its first content quad, else its
    /// box model, else `getBoundingClientRect`; `None` when it has no area.
    async fn rect(&self, node: &Node) -> Option<(f64, f64, f64, f64)> {
        let of_quad = |quad: &Value| {
            let quad = quad.as_array()?;
            let at = |i: usize| quad.get(i).and_then(Value::as_f64);
            let xs = [at(0)?, at(2)?, at(4)?, at(6)?];
            let ys = [at(1)?, at(3)?, at(5)?, at(7)?];
            let (x0, x1) = (
                xs.iter().copied().fold(f64::MAX, f64::min),
                xs.iter().copied().fold(f64::MIN, f64::max),
            );
            let (y0, y1) = (
                ys.iter().copied().fold(f64::MAX, f64::min),
                ys.iter().copied().fold(f64::MIN, f64::max),
            );
            Some((x0, y0, x1 - x0, y1 - y0))
        };
        let by = json!({ "backendNodeId": node.backend });
        let quad = match self
            .cdp
            .call("DOM.getContentQuads", by.clone(), Some(&node.session))
            .await
        {
            Ok(quads) => quads["quads"].get(0).cloned(),
            Err(_) => None,
        };
        let quad = match quad {
            Some(quad) => Some(quad),
            None => self
                .cdp
                .call("DOM.getBoxModel", by, Some(&node.session))
                .await
                .ok()
                .map(|m| m["model"]["content"].clone()),
        };
        if let Some(rect) = quad.as_ref().and_then(of_quad) {
            return (rect.2 > 0.0 && rect.3 > 0.0).then_some(rect);
        }
        let r = self
            .call_on(
                node,
                "function() { const r = this.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }",
                json!([]),
            )
            .await
            .ok()?;
        let n = |i: usize| r[i].as_f64();
        let rect = (n(0)?, n(1)?, n(2)?, n(3)?);
        (rect.2 > 0.0 && rect.3 > 0.0).then_some(rect)
    }

    /// Whether something else is on top of `node` at `at`, in its frame's coordinates.
    async fn occluded(&self, node: &Node, at: (f64, f64)) -> bool {
        match self
            .call_on(
                node,
                OCCLUSION,
                json!([{ "value": at.0 }, { "value": at.1 }]),
            )
            .await
        {
            Ok(clickable) => clickable != true,
            Err(_) => false,
        }
    }

    /// Calls `function` (JavaScript source) with `this` as `node`.
    async fn call_on(&self, node: &Node, function: &str, args: Value) -> Result<Value, CallError> {
        let resolved = self
            .cdp
            .call(
                "DOM.resolveNode",
                json!({ "backendNodeId": node.backend }),
                Some(&node.session),
            )
            .await?;
        let object = resolved["object"]["objectId"].clone();
        let answer = self
            .cdp
            .call(
                "Runtime.callFunctionOn",
                json!({
                    "objectId": object,
                    "functionDeclaration": function,
                    "arguments": args,
                    "returnByValue": true,
                    "awaitPromise": true,
                }),
                Some(&node.session),
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

    /// A click as a person makes one: move, 50 ms, press, `hold`, release. A press or
    /// release the page sits on is given up on, not failed.
    async fn mouse_click(&self, (x, y): (f64, f64), hold: Duration) -> Result<(), CallError> {
        self.page(
            "Input.dispatchMouseEvent",
            json!({ "type": "mouseMoved", "x": x, "y": y }),
        )
        .await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let button =
            |kind: &str| json!({ "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1 });
        if let Ok(pressed) = tokio::time::timeout(
            Duration::from_secs(3),
            self.page("Input.dispatchMouseEvent", button("mousePressed")),
        )
        .await
        {
            pressed?;
            tokio::time::sleep(hold).await;
        }
        if let Ok(released) = tokio::time::timeout(
            Duration::from_secs(5),
            self.page("Input.dispatchMouseEvent", button("mouseReleased")),
        )
        .await
        {
            released?;
        }
        Ok(())
    }

    async fn viewport_size(&self) -> (f64, f64) {
        let metrics = self
            .page("Page.getLayoutMetrics", json!({}))
            .await
            .unwrap_or_default();
        let viewport = &metrics["layoutViewport"];
        (
            viewport["clientWidth"].as_f64().unwrap_or(1280.0),
            viewport["clientHeight"].as_f64().unwrap_or(800.0),
        )
    }

    async fn mouse_press(&self, (x, y): (f64, f64), count: i64) -> Result<(), CallError> {
        for kind in ["mousePressed", "mouseReleased"] {
            self.page(
                "Input.dispatchMouseEvent",
                json!({ "type": kind, "x": x, "y": y, "button": "left", "clickCount": count }),
            )
            .await?;
        }
        Ok(())
    }

    async fn key_event(&self, kind: &str, key: &str, modifiers: i64) -> Result<(), CallError> {
        let (code, key_code) = key_code(key);
        let mut params = json!({ "type": kind, "key": key, "code": code });
        if modifiers != 0 {
            params["modifiers"] = json!(modifiers);
        }
        if let Some(key_code) = key_code {
            params["windowsVirtualKeyCode"] = json!(key_code);
        }
        self.page("Input.dispatchKeyEvent", params).await?;
        Ok(())
    }

    /// One character as a person types it: keyDown, char, keyUp, with Shift where the US layout
    /// needs it; `pause` sits between keyDown and char.
    async fn type_char(&self, c: char, pause: Duration) -> Result<(), CallError> {
        let (modifiers, key_code, key) = char_key(c);
        let code = char_code(&key);
        let stroke = |kind: &str| json!({ "type": kind, "key": key, "code": code, "modifiers": modifiers, "windowsVirtualKeyCode": key_code });
        self.page("Input.dispatchKeyEvent", stroke("keyDown"))
            .await?;
        if !pause.is_zero() {
            tokio::time::sleep(pause).await;
        }
        self.page(
            "Input.dispatchKeyEvent",
            json!({ "type": "char", "text": c.to_string(), "key": c.to_string() }),
        )
        .await?;
        self.page("Input.dispatchKeyEvent", stroke("keyUp")).await?;
        Ok(())
    }

    /// Enter as typed in a field: keyDown, a carriage return, keyUp.
    async fn type_enter(&self) -> Result<(), CallError> {
        let stroke = |kind: &str| json!({ "type": kind, "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13 });
        self.page("Input.dispatchKeyEvent", stroke("keyDown"))
            .await?;
        tokio::time::sleep(Duration::from_millis(1)).await;
        self.page(
            "Input.dispatchKeyEvent",
            json!({ "type": "char", "text": "\r", "key": "Enter" }),
        )
        .await?;
        self.page("Input.dispatchKeyEvent", stroke("keyUp")).await?;
        Ok(())
    }

    /// `text` typed into whatever has focus, a key at a time.
    async fn type_to_page(&self, text: &str) -> Result<(), CallError> {
        for c in text.chars() {
            if c == '\n' {
                self.type_enter().await?;
            } else {
                let c = c.to_string();
                self.page(
                    "Input.dispatchKeyEvent",
                    json!({ "type": "keyDown", "key": c }),
                )
                .await?;
                self.page(
                    "Input.dispatchKeyEvent",
                    json!({ "type": "char", "text": c }),
                )
                .await?;
                self.page(
                    "Input.dispatchKeyEvent",
                    json!({ "type": "keyUp", "key": c }),
                )
                .await?;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok(())
    }

    /// Empties a field: by script, else a triple click and Delete, else select-all and Backspace.
    async fn clear_field(&self, node: &Node) -> bool {
        match self.call_on(node, CLEAR_FIELD, json!([])).await {
            Ok(cleared)
                if cleared["cleared"] == true
                    && cleared["finalText"]
                        .as_str()
                        .is_none_or(|t| t.trim().is_empty()) =>
            {
                return true;
            }
            Ok(_) => {}
            Err(_) => return false,
        }
        if let Some(at) = self.center(node).await {
            let tripled = async {
                self.mouse_press(at, 3).await?;
                self.key_event("keyDown", "Delete", 0).await?;
                self.key_event("keyUp", "Delete", 0).await
            };
            if tripled.await.is_ok() {
                return true;
            }
        }
        let select_all = if cfg!(target_os = "macos") { 4 } else { 2 };
        let shortcut = async {
            self.page(
                "Input.dispatchKeyEvent",
                json!({ "type": "keyDown", "key": "a", "code": "KeyA", "modifiers": select_all }),
            )
            .await?;
            self.page(
                "Input.dispatchKeyEvent",
                json!({ "type": "keyUp", "key": "a", "code": "KeyA", "modifiers": select_all }),
            )
            .await?;
            self.key_event("keyDown", "Backspace", 0).await?;
            self.key_event("keyUp", "Backspace", 0).await
        };
        shortcut.await.is_ok()
    }

    /// Types `text` into `node`, and reads back what the field holds then.
    async fn type_into(
        &self,
        node: &Node,
        text: &str,
        clear: bool,
        traits: &Value,
    ) -> Result<Option<String>, CallError> {
        let centers = self.centers(node).await;
        self.call_on(node, "function() {}", json!([])).await?;
        let mut click_at = None;
        if let Some((at, local)) = centers
            && !self.occluded(node, local).await
        {
            click_at = Some(at);
        }
        let focused = self
            .cdp
            .call(
                "DOM.focus",
                json!({ "backendNodeId": node.backend }),
                Some(&node.session),
            )
            .await;
        if focused.is_err()
            && let Some(at) = click_at
        {
            let _ = self.mouse_press(at, 1).await;
        }
        if traits["direct"] == true {
            self.call_on(node, SET_VALUE, json!([{ "value": text }]))
                .await?;
            return Ok(None);
        }
        if clear {
            self.clear_field(node).await;
        }
        // A contenteditable can drop the first character typed at its start.
        let first = (traits["editable"] == true && clear)
            .then(|| text.chars().next())
            .flatten();
        for (i, c) in text.chars().enumerate() {
            if c == '\n' {
                self.type_enter().await?;
            } else {
                self.type_char(c, Duration::from_millis(5)).await?;
            }
            if i == 0
                && let Some(first) = first
            {
                let content = self
                    .cdp
                    .call(
                        "Runtime.evaluate",
                        json!({ "expression": "document.activeElement.textContent", "returnByValue": true }),
                        Some(&node.session),
                    )
                    .await?;
                let content = content["result"]["value"].as_str().unwrap_or_default();
                if !content.contains(first) {
                    self.type_char(first, Duration::from_millis(5)).await?;
                }
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let _ = self.call_on(node, FRAMEWORK_EVENTS, json!([])).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut actual = self
            .call_on(
                node,
                "function() { return this.value !== undefined ? this.value : this.textContent; }",
                json!([]),
            )
            .await
            .ok()
            .and_then(|v| v.as_str().map(str::to_string));
        // Text the clear missed, around what was typed: set it outright.
        if clear
            && let Some(value) = &actual
            && value != text
            && value.len() > text.len()
            && (value.starts_with(text) || value.ends_with(text))
            && let Ok(retried) = self
                .call_on(node, REPLACE_VALUE, json!([{ "value": text }]))
                .await
            && let Some(retried) = retried.as_str()
        {
            actual = Some(retried.to_string());
        }
        Ok(actual)
    }

    async fn viewport(&self) -> (f64, f64) {
        let metrics = self
            .page("Page.getLayoutMetrics", json!({}))
            .await
            .unwrap_or_default();
        let viewport = &metrics["cssVisualViewport"];
        (
            viewport["clientWidth"].as_f64().unwrap_or(1280.0),
            viewport["clientHeight"].as_f64().unwrap_or(800.0),
        )
    }

    // ---- the tools --------------------------------------------------------------------

    pub(super) async fn navigate(&mut self, url: &str, new_tab: bool) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let scheme = ["about:", "data:", "file:", "blob:"];
        let url = if url.contains("://") || scheme.iter().any(|s| url.starts_with(s)) {
            url.to_string()
        } else {
            format!("https://{url}")
        };
        let here = self.eval("location.href").await.unwrap_or_default();
        let here = here.as_str().unwrap_or_default().to_string();
        // A tab still on a new-tab page is navigated in place.
        let new_tab = new_tab && !is_new_tab_page(&here);
        let done = if new_tab {
            format!("Opened new tab with URL {url}.")
        } else {
            format!("Navigated to {url}.")
        };
        if new_tab {
            let blank = self.blank_tab().await;
            match blank {
                Some(i) => {
                    self.current = i;
                    self.refs.clear();
                    self.activate().await;
                }
                None => {
                    // Opened from the current tab, so it lands in the agent's window and is its own.
                    let open = format!(
                        "window.open({}, '_blank')",
                        serde_json::to_string(&url).unwrap_or_default()
                    );
                    self.page(
                        "Runtime.evaluate",
                        json!({ "expression": open, "userGesture": true }),
                    )
                    .await?;
                    tokio::time::sleep(Duration::from_millis(300)).await;
                    if !self.adopt_opened().await? {
                        bail!("the browser opened no tab");
                    }
                    self.refs.clear();
                    self.settle().await;
                    return Ok(self.with_page_news(done).await.into());
                }
            }
        }
        let host = |u: &str| u.split('/').nth(2).map(str::to_string);
        let same_site =
            url.starts_with("http") && here.starts_with("http") && host(&url) == host(&here);
        let wait = Duration::from_secs(if same_site { 3 } else { 8 });
        let mut result = self.go(&url, wait).await;
        if result.is_ok() && url.starts_with("http") {
            // An empty page may still be rendering, or need loading again.
            if self.looks_empty(false).await {
                tokio::time::sleep(Duration::from_secs(3)).await;
                if self.looks_empty(false).await {
                    let _ = self.go(&url, wait).await;
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    if self.looks_empty(true).await {
                        bail!(
                            "Page loaded but returned empty content for {url}. The page may require \
                             JavaScript that failed to render, use anti-bot measures, or have a \
                             connection issue (e.g. tunnel/proxy error). Try a different URL or approach."
                        );
                    }
                }
            }
        }
        if let Err(CallError::Browser(e)) = &result
            && e.contains("net::")
        {
            result = Err(CallError::Browser(format!(
                "Navigation failed - site unavailable: {url}"
            )));
        }
        self.after(&done, result).await
    }

    /// Navigates the current tab to `url` and waits up to `wait` for the new document to load;
    /// a load that takes longer is not a failure.
    async fn go(&self, url: &str, wait: Duration) -> Result<(), CallError> {
        let mut tries = 0;
        let answer = loop {
            let navigated = tokio::time::timeout(
                Duration::from_secs(20),
                self.page(
                    "Page.navigate",
                    json!({ "url": url, "transitionType": "address_bar" }),
                ),
            )
            .await
            .map_err(|_| {
                CallError::Browser(format!("Page.navigate() timed out after 20s for {url}"))
            })?;
            let answer = navigated?;
            match answer["errorText"].as_str().filter(|e| !e.is_empty()) {
                // A browser just booted may still be bringing its network up.
                Some(e) if e.ends_with("ERR_NETWORK_CHANGED") && tries < 3 => {
                    tries += 1;
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
                Some(e) => return Err(CallError::Browser(format!("Navigation failed: {e}"))),
                None => break answer,
            }
        };
        let Some(loader) = answer["loaderId"].as_str() else {
            return Ok(());
        };
        let deadline = tokio::time::Instant::now() + wait;
        while tokio::time::Instant::now() < deadline {
            if self.loader().await.as_deref() == Some(loader)
                && matches!(self.eval("document.readyState").await, Ok(s) if s == "complete")
            {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        Ok(())
    }

    /// Whether the page shows nothing: no text and nothing to act on, or with `bare`, no body
    /// at all.
    async fn looks_empty(&self, bare: bool) -> bool {
        let test = if bare {
            "!document.body || document.body.children.length === 0"
        } else {
            "!document.body || (document.body.innerText.trim() === '' \
             && !document.querySelector('a, button, input, select, textarea, img, iframe'))"
        };
        matches!(self.eval(test).await, Ok(Value::Bool(true)))
    }

    /// An agent tab besides the current one that is still blank.
    async fn blank_tab(&self) -> Option<usize> {
        let all = self
            .cdp
            .call("Target.getTargets", json!({}), None)
            .await
            .ok()?;
        let infos = all["targetInfos"].as_array()?;
        self.tabs.iter().enumerate().position(|(i, tab)| {
            i != self.current
                && infos.iter().any(|info| {
                    info["targetId"] == tab.target.as_str() && info["url"] == "about:blank"
                })
        })
    }

    pub(super) async fn back(&mut self) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let history = self.page("Page.getNavigationHistory", json!({})).await?;
        let current = history["currentIndex"].as_i64().unwrap_or(0);
        if current <= 0 {
            bail!("there is no page before this one to go back to");
        }
        let entry = history["entries"][(current - 1) as usize]["id"].clone();
        let result = self
            .page("Page.navigateToHistoryEntry", json!({ "entryId": entry }))
            .await
            .map(|_| ());
        tokio::time::sleep(Duration::from_millis(500)).await;
        self.after("Navigated back.", result).await
    }

    pub(super) async fn snapshot(&mut self, look: &Look<'_>) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let main = self.session().to_string();
        let tree = self.page("Accessibility.getFullAXTree", json!({})).await?;
        let mut docs = vec![snapshot::Doc::new(
            tree["nodes"].as_array().cloned().unwrap_or_default(),
        )];
        let mut places = vec![(main.clone(), None::<(String, i64)>)];
        let mut frames = HashMap::new();
        self.read_frames(&main, &mut docs, &mut places, &mut frames)
            .await;
        // One DOM per process: the page's, and each frame's in a process of its own.
        let mut doms: HashMap<String, Arc<snapshot::Dom>> = HashMap::new();
        for (doc, (session, owner)) in docs.iter_mut().zip(&places) {
            if !doms.contains_key(session) {
                doms.insert(session.clone(), Arc::new(self.dom(session).await));
            }
            doc.dom = doms[session].clone();
            if let Some((_, owner)) = owner {
                doc.prefix = format!("{owner}.");
            }
        }
        let loader = self.loader().await;
        if loader != self.refs_loader {
            self.refs.clear();
        }

        let scope = match look.selector {
            Some(selector) => Some(self.subtree(selector).await?),
            None => None,
        };
        let urls = if look.urls {
            self.link_urls().await
        } else {
            HashMap::new()
        };
        let options = snapshot::Options {
            interactive: look.interactive,
            depth: look.depth,
            scope,
            urls,
            seen: self.refs.keys().cloned().collect(),
        };
        let snapshot = snapshot::render(&docs, &frames, &options);
        let shown = snapshot.refs.len();
        // A ref names one element for as long as its document lasts, so earlier ones still hold.
        self.refs
            .extend(snapshot.refs.into_iter().map(|(r#ref, at)| {
                let (session, owner) = places[at.doc].clone();
                let node = Node {
                    session,
                    backend: at.backend,
                    owner,
                };
                (r#ref, node)
            }));
        self.refs_loader = loader;

        let mut head = self.whereabouts().await;
        if let Some(stats) = self.stats(shown).await {
            head.push('\n');
            head.push_str(&stats);
        }
        for alert in self.cdp.take_accepted(self.session()) {
            head.push_str(&format!(
                "\nClosed {}: {:?}.",
                article(&alert.kind),
                alert.message
            ));
        }
        let body = if snapshot.text.is_empty() {
            "(nothing on the page yet)".to_string()
        } else {
            snapshot::cut(&snapshot.text, look.max_chars)
        };
        Ok(format!("{head}\n\n{body}").into())
    }

    /// The page's frames, read into `docs`: one in the page's own process by its frame id, one in
    /// a process of its own through the session Chrome attached for it. `frames` says which node
    /// holds which.
    async fn read_frames(
        &self,
        main: &str,
        docs: &mut Vec<snapshot::Doc>,
        places: &mut Vec<(String, Option<(String, i64)>)>,
        frames: &mut HashMap<snapshot::At, usize>,
    ) {
        let Ok(tree) = self.page("Page.getFrameTree", json!({})).await else {
            return;
        };
        // (frame, its parent's document), breadth first.
        let mut doc_of: HashMap<String, usize> = HashMap::new();
        if let Some(id) = tree["frameTree"]["frame"]["id"].as_str() {
            doc_of.insert(id.to_string(), 0);
        }
        let mut queue: Vec<(&Value, String)> = Vec::new();
        let root_id = tree["frameTree"]["frame"]["id"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        for child in tree["frameTree"]["childFrames"]
            .as_array()
            .into_iter()
            .flatten()
        {
            queue.push((child, root_id.clone()));
        }
        let detached = self.cdp.frames(main);
        let mut seen: HashSet<String> = HashSet::new();
        let mut i = 0;
        while i < queue.len() {
            let (frame, parent) = (queue[i].0, queue[i].1.clone());
            i += 1;
            let Some(id) = frame["frame"]["id"].as_str() else {
                continue;
            };
            seen.insert(id.to_string());
            let Some(&parent_doc) = doc_of.get(&parent) else {
                continue;
            };
            let Some(owner) = self.frame_owner(main, id).await else {
                continue;
            };
            let read = self
                .page("Accessibility.getFullAXTree", json!({ "frameId": id }))
                .await
                .ok()
                .and_then(|t| t["nodes"].as_array().cloned())
                .filter(|nodes| !nodes.is_empty());
            let (nodes, place) = match read {
                Some(nodes) => (nodes, (main.to_string(), None)),
                None => match detached.iter().find(|(frame, _)| frame == id) {
                    Some((_, session)) => match self.oopif_tree(session).await {
                        Some(nodes) => (nodes, (session.clone(), Some((main.to_string(), owner)))),
                        None => continue,
                    },
                    None => continue,
                },
            };
            docs.push(snapshot::Doc::new(nodes));
            places.push(place);
            let doc = docs.len() - 1;
            frames.insert(
                snapshot::At {
                    doc: parent_doc,
                    backend: owner,
                },
                doc,
            );
            doc_of.insert(id.to_string(), doc);
            for child in frame["childFrames"].as_array().into_iter().flatten() {
                queue.push((child, id.to_string()));
            }
        }
        // Frames in a process of their own that the page's frame tree left out.
        for (id, session) in detached.iter().filter(|(id, _)| !seen.contains(id)) {
            let (Some(owner), Some(nodes)) = (
                self.frame_owner(main, id).await,
                self.oopif_tree(session).await,
            ) else {
                continue;
            };
            docs.push(snapshot::Doc::new(nodes));
            places.push((session.clone(), Some((main.to_string(), owner))));
            frames.insert(
                snapshot::At {
                    doc: 0,
                    backend: owner,
                },
                docs.len() - 1,
            );
        }
    }

    /// The DOM of the process `session` is in, with the elements a click, mouse or pointer
    /// listener is on; empty when it cannot be read.
    async fn dom(&self, session: &str) -> snapshot::Dom {
        let Ok(captured) = self
            .cdp
            .call(
                "DOMSnapshot.captureSnapshot",
                json!({
                    "computedStyles": snapshot::STYLES,
                    "includePaintOrder": true,
                    "includeDOMRects": true,
                }),
                Some(session),
            )
            .await
        else {
            return snapshot::Dom::new();
        };
        let mut dom = snapshot::read_dom(&captured);
        for backend in self.listened(session).await {
            if let Some(facts) = dom.get_mut(&backend) {
                facts.listener = true;
            }
        }
        dom
    }

    /// The elements with a click, mouse or pointer listener: none on a page of more than 10,000 elements, or with more than 100 such.
    async fn listened(&self, session: &str) -> Vec<i64> {
        let call =
            |method: &'static str, params: Value| self.cdp.call(method, params, Some(session));
        let Ok(found) = call(
            "Runtime.evaluate",
            json!({
                "expression": LISTENED,
                "includeCommandLineAPI": true,
                "objectGroup": "ailoy-listeners",
            }),
        )
        .await
        else {
            return Vec::new();
        };
        let mut backends = Vec::new();
        if let Some(array) = found["result"]["objectId"].as_str()
            && let Ok(items) = call(
                "Runtime.getProperties",
                json!({ "objectId": array, "ownProperties": true }),
            )
            .await
        {
            for item in items["result"].as_array().into_iter().flatten() {
                let is_index = item["name"]
                    .as_str()
                    .is_some_and(|n| n.parse::<usize>().is_ok());
                let Some(object) = item["value"]["objectId"].as_str().filter(|_| is_index) else {
                    continue;
                };
                if let Ok(node) = call("DOM.describeNode", json!({ "objectId": object })).await
                    && let Some(backend) = node["node"]["backendNodeId"].as_i64()
                {
                    backends.push(backend);
                }
            }
        }
        let _ = call(
            "Runtime.releaseObjectGroup",
            json!({ "objectGroup": "ailoy-listeners" }),
        )
        .await;
        backends
    }

    async fn frame_owner(&self, session: &str, frame: &str) -> Option<i64> {
        self.cdp
            .call(
                "DOM.getFrameOwner",
                json!({ "frameId": frame }),
                Some(session),
            )
            .await
            .ok()?["backendNodeId"]
            .as_i64()
    }

    async fn oopif_tree(&self, session: &str) -> Option<Vec<Value>> {
        for domain in ["DOM.enable", "Accessibility.enable"] {
            let _ = self.cdp.call(domain, json!({}), Some(session)).await;
        }
        let tree = self
            .cdp
            .call("Accessibility.getFullAXTree", json!({}), Some(session))
            .await
            .ok()?;
        tree["nodes"].as_array().cloned()
    }

    /// The backend ids under the first element `selector` matches, it included.
    async fn subtree(&self, selector: &str) -> anyhow::Result<HashSet<i64>> {
        let found = self
            .page(
                "Runtime.evaluate",
                json!({ "expression": format!("document.querySelector({})", serde_json::to_string(selector)?) }),
            )
            .await?;
        let Some(object) = found["result"]["objectId"].as_str() else {
            bail!("no element matches {selector:?}");
        };
        let described = self
            .page(
                "DOM.describeNode",
                json!({ "objectId": object, "depth": -1, "pierce": true }),
            )
            .await?;
        let mut ids = HashSet::new();
        collect_backends(&described["node"], &mut ids);
        Ok(ids)
    }

    /// Every link's address in the page, by its node.
    async fn link_urls(&self) -> HashMap<snapshot::At, String> {
        let mut urls = HashMap::new();
        let Ok(document) = self
            .page("DOM.getDocument", json!({ "depth": -1, "pierce": true }))
            .await
        else {
            return urls;
        };
        let base = document["root"]["documentURL"]
            .as_str()
            .and_then(|u| url::Url::parse(u).ok());
        let mut stack = vec![&document["root"]];
        while let Some(node) = stack.pop() {
            if node["nodeName"] == "A"
                && let (Some(backend), Some(attributes)) = (
                    node["backendNodeId"].as_i64(),
                    node["attributes"].as_array(),
                )
                && let Some(href) = attributes
                    .chunks(2)
                    .find(|pair| pair[0] == "href")
                    .and_then(|pair| pair.get(1)?.as_str())
            {
                let href = base
                    .as_ref()
                    .and_then(|base| base.join(href).ok())
                    .map_or_else(|| href.to_string(), |u| u.to_string());
                urls.insert(snapshot::At { doc: 0, backend }, href);
            }
            for key in ["children", "shadowRoots"] {
                stack.extend(node[key].as_array().into_iter().flatten());
            }
            if node.get("contentDocument").is_some() {
                stack.push(&node["contentDocument"]);
            }
        }
        urls
    }

    /// One line on the page as a whole: what it has, whether it still loads, and how much lies
    /// above and below.
    async fn stats(&self, refs: usize) -> Option<String> {
        let stats = self
            .eval(
                "(() => { const q = s => document.querySelectorAll(s).length; \
                 const H = Math.max(document.documentElement.scrollHeight, document.body ? document.body.scrollHeight : 0); \
                 return [q('a[href]'), q('iframe'), q('img'), q('*'), document.readyState, scrollY, innerHeight, H]; })()",
            )
            .await
            .ok()?;
        let n = |i: usize| stats[i].as_f64().unwrap_or(0.0);
        let (y, h, total) = (n(5), n(6).max(1.0), n(7));
        let above = y / h;
        let below = ((total - y - h) / h).max(0.0);
        let mut line = format!(
            "{} links, {refs} refs, {} iframes, {} images, {} elements; {above:.1} pages above, {below:.1} below",
            n(0),
            n(1),
            n(2),
            n(3)
        );
        if below > 0.2 {
            line.push_str(" (scroll down for more)");
        }
        line.push('.');
        if stats[4] != "complete" {
            line.push_str(" The page is still loading: consider `browser_wait`.");
        } else if n(3) < 10.0 {
            line.push_str(" The page looks empty: consider `browser_wait`.");
        }
        Some(line)
    }

    /// Clicks `node` at the middle of its box, kept on screen, by script
    /// when something covers it or it has no box or the mouse fails; a checkbox or radio the
    /// mouse did not toggle is clicked by script too.
    /// Waits half a second for a download one of the agent's tabs began
    /// after the first `before`, then up to 30 s for it to finish; says how it went.
    async fn download_after(&self, before: usize) -> Option<String> {
        let ours = |d: &cdp::Download| self.tabs.iter().any(|t| t.target == d.frame);
        let started = tokio::time::Instant::now();
        let guid = loop {
            let found = self
                .cdp
                .downloads()
                .into_iter()
                .skip(before)
                .find(|(_, d)| ours(d));
            if let Some((guid, _)) = found {
                break guid;
            }
            if started.elapsed() > Duration::from_millis(500) {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        };
        let wait = Duration::from_secs(30);
        let started = tokio::time::Instant::now();
        let mut last = (0, tokio::time::Instant::now());
        loop {
            let d = self
                .cdp
                .downloads()
                .into_iter()
                .find(|(g, _)| *g == guid)?
                .1;
            if d.received != last.0 {
                last = (d.received, tokio::time::Instant::now());
            }
            match d.state.as_str() {
                "completed" => {
                    let place = d
                        .path
                        .unwrap_or_else(|| "the user's downloads folder".into());
                    return Some(format!(
                        "Downloaded file: {} ({} bytes) saved to {place}.",
                        d.name, d.received
                    ));
                }
                "canceled" => return Some(format!("The download of {} was canceled.", d.name)),
                _ => {}
            }
            if started.elapsed() > wait {
                let (received, total) = (d.received, d.total);
                let name = d.name;
                return Some(if last.1.elapsed() < Duration::from_secs(5) {
                    let progress = if total > 0 {
                        format!(
                            "{:.1}% ({}/{} bytes)",
                            received as f64 / total as f64 * 100.0,
                            super::thousands(received as usize),
                            super::thousands(total as usize)
                        )
                    } else {
                        format!(
                            "{} bytes downloaded (total size unknown)",
                            super::thousands(received as usize)
                        )
                    };
                    format!(
                        "Download timed out after 30s but is still in progress: {name} - {progress}. \
                         The download appears to be progressing normally. Consider using the wait \
                         action to allow more time for the download to complete."
                    )
                } else if received > 0 {
                    format!(
                        "Download timed out after 30s: {name}. Last progress: {} bytes received. \
                         The download may have stalled or completed - check the downloads folder.",
                        super::thousands(received as usize)
                    )
                } else {
                    format!(
                        "Download timed out after 30s: {name}. No progress data received - the \
                         download may have failed to start properly."
                    )
                });
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn click_node(&self, node: &Node) -> Result<(), CallError> {
        let js_click = |pause: u64| async move {
            self.call_on(node, "function() { this.click(); }", json!([]))
                .await?;
            tokio::time::sleep(Duration::from_millis(pause)).await;
            Ok::<_, CallError>(())
        };
        let checked = "function() { return this.checked; }";
        let toggle = self
            .call_on(
                node,
                "function() { return this.tagName === 'INPUT' && ['checkbox', 'radio'].includes((this.getAttribute('type') || '').toLowerCase()); }",
                json!([]),
            )
            .await?
            == true;
        let before = if toggle {
            self.call_on(node, checked, json!([])).await.ok()
        } else {
            None
        };
        let (width, height) = self.viewport_size().await;
        let origin = match &node.owner {
            Some(owner) => self.frame_origin(owner).await.unwrap_or((0.0, 0.0)),
            None => (0.0, 0.0),
        };
        let _ = self
            .cdp
            .call(
                "DOM.scrollIntoViewIfNeeded",
                json!({ "backendNodeId": node.backend }),
                Some(&node.session),
            )
            .await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        let Some((x, y, w, h)) = self.rect(node).await else {
            return js_click(50).await;
        };
        let at = (
            (origin.0 + x + w / 2.0).clamp(0.0, (width - 1.0).max(0.0)),
            (origin.1 + y + h / 2.0).clamp(0.0, (height - 1.0).max(0.0)),
        );
        if self
            .occluded(node, (at.0 - origin.0, at.1 - origin.1))
            .await
        {
            return js_click(50).await;
        }
        match self.mouse_click(at, Duration::from_millis(80)).await {
            Err(CallError::Browser(_)) => return js_click(100).await,
            other => other?,
        }
        if let Some(before) = before {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if self.call_on(node, checked, json!([])).await.ok() == Some(before) {
                js_click(50).await?;
            }
        }
        Ok(())
    }

    pub(super) async fn click(
        &mut self,
        r#ref: Option<&str>,
        at: Option<(f64, f64)>,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        if let Some(r#ref) = r#ref {
            let node = self.node(r#ref)?;
            let element = self.call_on(&node, CLICK_TARGET, json!([])).await?;
            match element["kind"].as_str() {
                Some("select") => {
                    return match self.dropdown_options(r#ref).await {
                        Ok(options) => Ok(options),
                        Err(_) => bail!(
                            "cannot click on <select> elements: use browser_dropdown_options(ref={ref}) instead"
                        ),
                    };
                }
                Some("file") => bail!(
                    "cannot click on file input element ({ref}): file uploads must be handled using browser_upload_file"
                ),
                _ => {}
            }
            if element["kind"] == "print" {
                bail!(
                    "{ref} prints the page, which would open the print dialog, so it was not \
                     clicked: browser_save_as_pdf saves the page as a PDF"
                );
            }
            let mut done = format!(
                "Clicked {}.",
                element["description"].as_str().unwrap_or("it")
            );
            let before = self.cdp.downloads().len();
            let result = self.click_node(&node).await;
            if result.is_ok()
                && let Some(download) = self.download_after(before).await
            {
                done.push(' ');
                done.push_str(&download);
            }
            return self.after(&done, result).await;
        }
        let Some((x, y)) = at else {
            bail!("give a ref, or x and y");
        };
        // In the last screenshot's pixels when there is one, else the page's.
        let scale = self.shot.map_or(1.0, |shot| shot.css_width / shot.width);
        let before = self.cdp.downloads().len();
        let result = self
            .mouse_click((x * scale, y * scale), Duration::from_millis(50))
            .await;
        let mut done = format!("Clicked on coordinate {x}, {y}.");
        if result.is_ok()
            && let Some(download) = self.download_after(before).await
        {
            done.push(' ');
            done.push_str(&download);
        }
        self.after(&done, result).await
    }

    pub(super) async fn input(
        &mut self,
        r#ref: &str,
        text: &str,
        clear: bool,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let clear = clear || text.is_empty();
        let traits = self
            .call_on(&node, INPUT_TRAITS, json!([]))
            .await
            .unwrap_or_default();
        let mut done = format!("Typed '{text}'.");
        let result = match self.type_into(&node, text, clear, &traits).await {
            Ok(actual) => {
                if let Some(actual) = actual.filter(|a| a != text) {
                    done.push_str(&format!(
                        "\nNote: the field's actual value '{actual}' differs from typed text '{text}'. \
                         The page may have reformatted or autocompleted your input."
                    ));
                }
                Ok(())
            }
            Err(CallError::Browser(_)) => {
                // Typed to the page instead, after a click to give the field focus.
                let _ = tokio::time::timeout(Duration::from_secs(10), self.click_node(&node)).await;
                self.type_to_page(text).await
            }
            Err(e) => Err(e),
        };
        if result.is_ok() && traits["autocomplete"] == true {
            done.push_str(
                "\nThis is an autocomplete field. Wait for suggestions to appear, then click the \
                 correct suggestion instead of pressing Enter.",
            );
            if traits["waits"] == true {
                tokio::time::sleep(Duration::from_millis(400)).await;
            }
        }
        self.after(&done, result).await
    }

    pub(super) async fn send_keys(&mut self, keys: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let result = self.press(keys).await;
        self.after(&format!("Sent keys: {keys}."), result).await
    }

    /// Sends `keys`: a shortcut, a named key, or text typed a key at a time.
    async fn press(&self, keys: &str) -> Result<(), CallError> {
        if let Some((modifiers, main)) = combination(keys) {
            let bits = modifiers
                .iter()
                .filter_map(|m| modifier_bit(m))
                .fold(0, |a, b| a | b);
            for m in &modifiers {
                self.key_event("keyDown", m, 0).await?;
            }
            self.key_event("keyDown", &main, bits).await?;
            self.key_event("keyUp", &main, bits).await?;
            for m in modifiers.iter().rev() {
                self.key_event("keyUp", m, 0).await?;
            }
        } else {
            let keys = alias(keys.trim()).unwrap_or(keys);
            if SPECIAL_KEYS.contains(&keys) {
                self.key_event("keyDown", keys, 0).await?;
                // keypress listeners hear Enter by its char.
                if keys == "Enter" {
                    self.page(
                        "Input.dispatchKeyEvent",
                        json!({ "type": "char", "text": "\r", "key": "Enter" }),
                    )
                    .await?;
                }
                self.key_event("keyUp", keys, 0).await?;
            } else {
                for c in keys.chars() {
                    if c == '\n' || c == '\r' {
                        for kind in ["rawKeyDown", "char", "keyUp"] {
                            self.page(
                                "Input.dispatchKeyEvent",
                                json!({ "type": kind, "windowsVirtualKeyCode": 13, "unmodifiedText": "\r", "text": "\r" }),
                            )
                            .await?;
                        }
                        continue;
                    }
                    self.type_char(c, Duration::ZERO).await?;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
        let lower = keys.to_lowercase();
        if lower.contains("enter") || lower.contains("return") {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Ok(())
    }

    pub(super) async fn scroll(
        &mut self,
        down: bool,
        pages: f64,
        r#ref: Option<&str>,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = match r#ref {
            Some(r#ref) => Some(self.node(r#ref)?),
            None => None,
        };
        let direction = if down { "down" } else { "up" };
        let target = r#ref.map(|r| format!(" element {r}")).unwrap_or_default();
        let metrics = self
            .page("Page.getLayoutMetrics", json!({}))
            .await
            .unwrap_or_default();
        let height = metrics["cssVisualViewport"]["clientHeight"]
            .as_f64()
            .or(metrics["cssLayoutViewport"]["clientHeight"].as_f64())
            .unwrap_or(1000.0)
            .trunc();
        let sign = if down { 1.0 } else { -1.0 };
        let done = if pages >= 1.0 {
            let full = pages.trunc() as usize;
            let fraction = pages - pages.trunc();
            let mut scrolled = 0.0;
            for _ in 0..full {
                if self.scroll_by(sign * height, node.as_ref()).await.is_ok() {
                    scrolled += 1.0;
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            if fraction > 0.0
                && self
                    .scroll_by(sign * (fraction * height).trunc(), node.as_ref())
                    .await
                    .is_ok()
            {
                scrolled += fraction;
            }
            if pages == 1.0 {
                format!("Scrolled {direction}{target} {height}px.")
            } else {
                format!("Scrolled {direction}{target} {scrolled:.1} pages.")
            }
        } else {
            self.scroll_by(sign * (pages * height).trunc(), node.as_ref())
                .await?;
            format!("Scrolled {direction}{target} {pages} pages.")
        };
        Ok(self.with_page_news(done).await.into())
    }

    /// Scrolls `pixels` down (up when negative): inside `node` when it can be, by the wheel at its
    /// middle or an iframe's document, else the page by a scroll gesture.
    async fn scroll_by(&self, pixels: f64, node: Option<&Node>) -> Result<(), CallError> {
        if let Some(node) = node {
            let iframe = self
                .call_on(
                    node,
                    "function(d) { if (this.tagName !== 'IFRAME') return null; \
                     try { const doc = this.contentDocument || this.contentWindow.document; \
                       const el = doc && (doc.documentElement || doc.body); \
                       if (!el) return false; el.scrollTop += d; return true; } catch (e) { return false; } }",
                    json!([{ "value": pixels }]),
                )
                .await;
            match iframe {
                Ok(Value::Bool(true)) => {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    return Ok(());
                }
                Err(CallError::Dialog) => return Err(CallError::Dialog),
                _ => {}
            }
            let origin = match &node.owner {
                Some(owner) => self.frame_origin(owner).await,
                None => Some((0.0, 0.0)),
            };
            let model = self
                .cdp
                .call(
                    "DOM.getBoxModel",
                    json!({ "backendNodeId": node.backend }),
                    Some(&node.session),
                )
                .await;
            if let (Some(origin), Ok(model)) = (origin, model) {
                let q = &model["model"]["content"];
                let n = |i: usize| q[i].as_f64().unwrap_or_default();
                let x = origin.0 + (n(0) + n(2) + n(4) + n(6)) / 4.0;
                let y = origin.1 + (n(1) + n(3) + n(5) + n(7)) / 4.0;
                let wheel = self
                    .page(
                        "Input.dispatchMouseEvent",
                        json!({ "type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": pixels }),
                    )
                    .await;
                match wheel {
                    Ok(_) => return Ok(()),
                    Err(CallError::Browser(_)) => {}
                    Err(e) => return Err(e),
                }
            }
        }
        let (width, height) = self.viewport_size().await;
        self.page(
            "Input.synthesizeScrollGesture",
            json!({ "x": width / 2.0, "y": height / 2.0, "xDistance": 0, "yDistance": -pixels, "speed": 50000 }),
        )
        .await?;
        Ok(())
    }

    /// Scrolls to the first element holding `text`, found by XPath on its own text, its text with its descendants', then its attributes, else by walking the text.
    pub(super) async fn find_text(&mut self, text: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let _ = self.page("DOM.enable", json!({})).await;
        let _ = self.page("DOM.getDocument", json!({ "depth": -1 })).await;
        let mut found = false;
        // A double quote would end the XPath string; such text is left to the walk.
        if !text.contains('"') {
            for query in [
                format!("//*[contains(text(), \"{text}\")]"),
                format!("//*[contains(., \"{text}\")]"),
                format!("//*[@*[contains(., \"{text}\")]]"),
            ] {
                let Ok(search) = self
                    .page("DOM.performSearch", json!({ "query": query }))
                    .await
                else {
                    continue;
                };
                let id = search["searchId"].clone();
                if search["resultCount"].as_i64().unwrap_or(0) > 0
                    && let Ok(nodes) = self
                        .page(
                            "DOM.getSearchResults",
                            json!({ "searchId": id, "fromIndex": 0, "toIndex": 1 }),
                        )
                        .await
                    && let Some(node) = nodes["nodeIds"].get(0)
                    && self
                        .page("DOM.scrollIntoViewIfNeeded", json!({ "nodeId": node }))
                        .await
                        .is_ok()
                {
                    found = true;
                }
                let _ = self
                    .page("DOM.discardSearchResults", json!({ "searchId": id }))
                    .await;
                if found {
                    break;
                }
            }
        }
        if !found {
            let walked = self
                .eval(&format!(
                    "(() => {{ const want = {}; \
                     const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); \
                     let node; while (node = walker.nextNode()) {{ \
                       if (node.textContent.includes(want)) {{ \
                         node.parentElement.scrollIntoView({{ behavior: 'smooth', block: 'center' }}); return true; }} }} \
                     return false; }})()",
                    serde_json::to_string(text)?
                ))
                .await;
            found = matches!(walked, Ok(Value::Bool(true)));
        }
        let done = if found {
            format!("Scrolled to text: {text}.")
        } else {
            format!("Text '{text}' not found or not visible on page.")
        };
        Ok(self.with_page_news(done).await.into())
    }

    pub(super) async fn dropdown_options(&mut self, r#ref: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let combobox = self
            .call_on(
                &node,
                "function() { const c = this.getAttribute('aria-controls'); \
                 return this.getAttribute('role') === 'combobox' && c \
                   ? [c, this.getAttribute('aria-expanded') === 'true'] : null; }",
                json!([]),
            )
            .await?;
        let found = match &combobox {
            Value::Array(c) => {
                let expanded = c[1] == true;
                if !expanded {
                    self.call_on(&node, OPEN_COMBOBOX, json!([])).await?;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                let found = self
                    .call_on(&node, COMBOBOX_OPTIONS, json!([{ "value": c[0] }]))
                    .await?;
                if !expanded {
                    self.call_on(&node, CLOSE_COMBOBOX, json!([])).await?;
                }
                found
            }
            _ => self.call_on(&node, DROPDOWN_OPTIONS, json!([])).await?,
        };
        if let Some(error) = found["error"].as_str() {
            bail!("{error}");
        }
        let options = found["options"].as_array().cloned().unwrap_or_default();
        if options.is_empty() {
            bail!("no options found in the dropdown at {ref}");
        }
        let mut lines = vec![format!(
            "Found {} dropdown{}:",
            found["type"].as_str().unwrap_or("a"),
            found["source"]
                .as_str()
                .filter(|s| *s != "target")
                .map(|s| format!(" in {s}"))
                .unwrap_or_default()
        )];
        for (i, o) in options.iter().enumerate() {
            lines.push(format!(
                "{i}: text={}, value={}{}",
                serde_json::to_string(&o["text"]).unwrap_or_default(),
                serde_json::to_string(&o["value"]).unwrap_or_default(),
                if o["selected"] == true {
                    " (selected)"
                } else {
                    ""
                }
            ));
        }
        lines.push(format!(
            "\nUse the exact text or value in browser_select_dropdown(ref={ref}, text=...)."
        ));
        Ok(lines.join("\n").into())
    }

    /// Picks the option of the dropdown at `ref` whose text or value is `text`, ignoring case: a
    /// `<select>` is set and told so; a menu, a listbox or a Semantic UI one
    /// has its item clicked.
    pub(super) async fn select_dropdown(
        &mut self,
        r#ref: &str,
        text: &str,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let picked = self
            .call_on(&node, SELECT_OPTION, json!([{ "value": text }]))
            .await?;
        if picked["success"] == true {
            let done = picked["message"]
                .as_str()
                .unwrap_or("Selected.")
                .to_string();
            return self.after(&format!("{done}."), Ok(())).await;
        }
        let mut error = picked["error"]
            .as_str()
            .unwrap_or("the option could not be selected")
            .to_string();
        if let Some(options) = picked["availableOptions"].as_array() {
            let options: Vec<String> = options
                .iter()
                .map(|o| {
                    format!(
                        "text={}, value={}",
                        serde_json::to_string(&o["text"]).unwrap_or_default(),
                        serde_json::to_string(&o["value"]).unwrap_or_default()
                    )
                })
                .collect();
            error.push_str(&format!(". Available: {}", options.join("; ")));
        }
        bail!("{error}")
    }

    pub(super) async fn search_page(
        &mut self,
        pattern: &str,
        search: &Search<'_>,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let params = format!(
            "var PATTERN = {}; var IS_REGEX = {}; var CASE_SENSITIVE = {}; var CONTEXT_CHARS = {}; \
             var CSS_SCOPE = {}; var MAX_RESULTS = {};\n",
            serde_json::to_string(pattern)?,
            search.regex,
            search.case_sensitive,
            search.context_chars,
            serde_json::to_string(&search.css_scope)?,
            search.max_results,
        );
        let data = self
            .eval(&format!("(function() {{\n{params}{SEARCH_PAGE}\n}})()"))
            .await
            .map_err(|e| anyhow::anyhow!("search_page failed: {e}"))?;
        if let Some(error) = data["error"].as_str() {
            bail!("search_page: {error}");
        }
        let total = data["total"].as_u64().unwrap_or(0);
        if total == 0 {
            return Ok(format!("No matches found for \"{pattern}\" on page.").into());
        }
        let matches = data["matches"].as_array().cloned().unwrap_or_default();
        let mut lines = vec![
            format!(
                "Found {total} match{} for \"{pattern}\" on page:",
                if total == 1 { "" } else { "es" }
            ),
            String::new(),
        ];
        for (i, m) in matches.iter().enumerate() {
            let path = m["element_path"].as_str().unwrap_or_default();
            let at = if path.is_empty() {
                String::new()
            } else {
                format!(" (in {path})")
            };
            lines.push(format!(
                "[{}] {}{at}",
                i + 1,
                m["context"].as_str().unwrap_or_default()
            ));
        }
        if data["has_more"] == true {
            lines.push(format!(
                "\n... showing {} of {total} total matches. Increase max_results to see more.",
                matches.len()
            ));
        }
        Ok(lines.join("\n").into())
    }

    pub(super) async fn find_elements(
        &mut self,
        selector: &str,
        attributes: Option<&[String]>,
        max_results: usize,
        include_text: bool,
    ) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let params = format!(
            "var SELECTOR = {}; var ATTRIBUTES = {}; var MAX_RESULTS = {max_results}; var INCLUDE_TEXT = {include_text};\n",
            serde_json::to_string(selector)?,
            serde_json::to_string(&attributes)?,
        );
        let data = self
            .eval(&format!("(function() {{\n{params}{FIND_ELEMENTS}\n}})()"))
            .await
            .map_err(|e| anyhow::anyhow!("find_elements failed: {e}"))?;
        if let Some(error) = data["error"].as_str() {
            bail!("find_elements: {error}");
        }
        let total = data["total"].as_u64().unwrap_or(0);
        if total == 0 {
            return Ok(format!("No elements found matching \"{selector}\".").into());
        }
        let mut lines = vec![
            format!(
                "Found {total} element{} matching \"{selector}\":",
                if total == 1 { "" } else { "s" }
            ),
            String::new(),
        ];
        for e in data["elements"].as_array().into_iter().flatten() {
            let mut parts = vec![format!(
                "[{}] <{}>",
                e["index"].as_u64().unwrap_or(0),
                e["tag"].as_str().unwrap_or("?")
            )];
            let text = e["text"]
                .as_str()
                .unwrap_or_default()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if !text.is_empty() {
                let shown: String = text.chars().take(120).collect();
                let more = if text.chars().count() > 120 {
                    "..."
                } else {
                    ""
                };
                parts.push(format!("\"{shown}{more}\""));
            }
            if let Some(attrs) = e["attrs"].as_object().filter(|a| !a.is_empty()) {
                let pairs: Vec<String> = attrs
                    .iter()
                    .map(|(k, v)| format!("{k}=\"{}\"", v.as_str().unwrap_or_default()))
                    .collect();
                parts.push(format!("{{{}}}", pairs.join(", ")));
            }
            parts.push(format!(
                "({} children)",
                e["children_count"].as_u64().unwrap_or(0)
            ));
            lines.push(parts.join(" "));
        }
        let showing = data["showing"].as_u64().unwrap_or(0);
        if showing < total {
            lines.push(format!(
                "\nShowing {showing} of {total} total elements. Increase max_results to see more."
            ));
        }
        Ok(lines.join("\n").into())
    }

    pub(super) async fn screenshot(&mut self) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let (css_width, _) = self.viewport().await;
        let shot = self
            .page(
                "Page.captureScreenshot",
                json!({ "format": "jpeg", "quality": 75 }),
            )
            .await?;
        let mut data = BASE64
            .decode(shot["data"].as_str().unwrap_or_default())
            .context("the screenshot was not base64")?;
        let mut picture = image::load_from_memory(&data).context("reading the screenshot")?;
        if picture.width() > SHOT_WIDTH {
            let height = picture.height() * SHOT_WIDTH / picture.width();
            picture =
                picture.resize_exact(SHOT_WIDTH, height, image::imageops::FilterType::Triangle);
            data.clear();
            picture
                .write_to(&mut Cursor::new(&mut data), image::ImageFormat::Jpeg)
                .context("writing the screenshot")?;
        }
        let (width, height) = (picture.width(), picture.height());
        self.shot = Some(Shot {
            css_width,
            width: width as f64,
        });
        let text = format!(
            "{} Screenshot of the viewport, {width}x{height}: `browser_click`'s x and y are pixels in it.",
            self.whereabouts().await
        );
        Ok(Said {
            text,
            image: Some(("image/jpeg", data)),
        })
    }

    /// The viewport as a PNG, at full size, for a file.
    pub(super) async fn screenshot_png(&mut self) -> anyhow::Result<Vec<u8>> {
        self.no_dialog()?;
        let shot = self
            .page("Page.captureScreenshot", json!({ "format": "png" }))
            .await?;
        BASE64
            .decode(shot["data"].as_str().unwrap_or_default())
            .context("the screenshot was not base64")
    }

    /// The page as a PDF.
    pub(super) async fn pdf(&mut self) -> anyhow::Result<Vec<u8>> {
        self.no_dialog()?;
        let printed = self
            .page("Page.printToPDF", json!({ "printBackground": true }))
            .await?;
        BASE64
            .decode(printed["data"].as_str().unwrap_or_default())
            .context("the PDF was not base64")
    }

    /// Puts `file`, on this host, in the file input at `ref`.
    pub(super) async fn upload(&mut self, r#ref: &str, file: &Path) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let node = self.node(r#ref)?;
        let is_file = self
            .call_on(
                &node,
                "function() { return this.tagName === 'INPUT' && this.type === 'file'; }",
                json!([]),
            )
            .await?;
        if is_file != Value::Bool(true) {
            bail!("{ref} is not a file input: take a snapshot and find the one to upload into");
        }
        let result = self
            .cdp
            .call(
                "DOM.setFileInputFiles",
                json!({ "files": [file.display().to_string()], "backendNodeId": node.backend }),
                Some(&node.session),
            )
            .await
            .map(|_| ());
        self.after("Uploaded.", result).await
    }

    pub(super) async fn evaluate(&mut self, code: &str) -> anyhow::Result<Said> {
        self.no_dialog()?;
        let code = fix_javascript(code);
        // A statement list or a function body runs as one: wrapped, its last value returned.
        let value = match self.eval(&code).await {
            Err(CallError::Browser(e)) if e.contains("SyntaxError") => {
                self.eval(&format!("(async () => {{ {code} }})()")).await
            }
            other => other,
        };
        let value = match value {
            Err(CallError::Browser(e)) => {
                let shown: String = code.chars().take(500).collect();
                let more = if code.chars().count() > 500 {
                    "..."
                } else {
                    ""
                };
                bail!(
                    "JavaScript Execution Failed:\nJavaScript execution error: {e}\n\
                     Validated Code (after quote fixing):\n{shown}{more}"
                );
            }
            other => other?,
        };
        let mut text = match value {
            Value::String(s) => s,
            Value::Null => "undefined".to_string(),
            other => serde_json::to_string(&other).unwrap_or_default(),
        };
        // Images it returns as data URLs are shown as images, not as base64.
        let mut image = None;
        let pattern = fancy_regex::Regex::new(r"data:(image/[^;]+);base64,([A-Za-z0-9+/=]+)")?;
        if let Ok(Some(found)) = pattern.captures(&text)
            && let (Some(kind), Some(data)) = (found.get(1), found.get(2))
            && let Ok(bytes) = BASE64.decode(data.as_str())
        {
            let kind: &'static str = match kind.as_str() {
                "image/png" => "image/png",
                "image/gif" => "image/gif",
                "image/webp" => "image/webp",
                _ => "image/jpeg",
            };
            image = Some((kind, bytes));
        }
        text = pattern.replace_all(&text, "[Image]").into_owned();
        if text.chars().count() > 20_000 {
            text = text.chars().take(19_950).collect::<String>()
                + "\n... [Truncated after 20000 characters]";
        }
        Ok(Said {
            text: self.with_page_news(text).await,
            image,
        })
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

    fn tab_index(&self, id: &str) -> anyhow::Result<usize> {
        let id = id.trim();
        self.tabs
            .iter()
            .position(|t| t.id() == id || t.target == id)
            .with_context(|| format!("no tab {id:?}: `browser_tabs` lists yours"))
    }

    pub(super) async fn list_tabs(&mut self) -> anyhow::Result<Said> {
        self.adopt_opened().await?;
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
                "{}{} {:?} {}",
                tab.id(),
                if i == self.current { " (current)" } else { "" },
                info["title"].as_str().unwrap_or_default(),
                info["url"].as_str().unwrap_or_default()
            ));
        }
        Ok(lines.join("\n").into())
    }

    pub(super) async fn switch_tab(&mut self, id: &str) -> anyhow::Result<Said> {
        self.adopt_opened().await?;
        self.current = self.tab_index(id)?;
        self.refs.clear();
        self.activate().await;
        Ok(self.with_page_news("Switched.".into()).await.into())
    }

    pub(super) async fn close_tab(&mut self, id: &str) -> anyhow::Result<Said> {
        self.adopt_opened().await?;
        let i = self.tab_index(id)?;
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
        Ok(self.with_page_news("Closed.".into()).await.into())
    }

    /// Opens `url` in the current tab, for an app starting the browser somewhere.
    pub(super) async fn open(&mut self, url: &str) -> anyhow::Result<()> {
        self.navigate(url, false).await.map(|_| ())
    }
}

/// Focuses and clicks a closed ARIA combobox, so its options render.
const OPEN_COMBOBOX: &str = "function() { \
    this.dispatchEvent(new FocusEvent('focus', { bubbles: true, cancelable: true })); this.focus(); \
    this.dispatchEvent(new FocusEvent('focusin', { bubbles: true, cancelable: true })); \
    this.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window })); \
    this.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true, view: window })); }";

/// Closes a combobox it opened.
const CLOSE_COMBOBOX: &str = "function() { this.blur(); \
    this.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); }";

/// The options in the listbox an ARIA combobox controls: its `[role=option]`s, else its `<li>`s.
const COMBOBOX_OPTIONS: &str = "function(id) { const box = document.getElementById(id); \
    if (!box) return { error: `the combobox controls #${id}, which is not on the page` }; \
    const read = item => ({ text: (item.textContent || '').trim(), \
      value: item.getAttribute('data-value') || item.getAttribute('value') || (item.textContent || '').trim(), \
      selected: item.getAttribute('aria-selected') === 'true' || item.classList.contains('selected') }); \
    let options = [...box.querySelectorAll('[role=option]')].map(read).filter(o => o.text); \
    if (!options.length) options = [...box.querySelectorAll('li')].map(read).filter(o => o.text); \
    return { type: 'an ARIA combobox', options, source: 'aria-controls' }; }";

/// The options of a `<select>`, a menu or listbox, or a Semantic UI dropdown: this element, or
/// the first such within it, four deep.
const DROPDOWN_OPTIONS: &str = "function() { \
    const item = (e, selected) => ({ text: (e.textContent || '').trim(), value: e.getAttribute('data-value') || (e.textContent || '').trim(), selected }); \
    const check = e => { \
      if (e.tagName.toLowerCase() === 'select') return { type: 'a <select>', \
        options: [...e.options].map(o => ({ text: o.text.trim(), value: o.value, selected: o.selected })) }; \
      const role = e.getAttribute('role'); \
      if (role === 'menu' || role === 'listbox') return { type: 'an ARIA', \
        options: [...e.querySelectorAll('[role=menuitem], [role=option]')] \
          .map(i => item(i, i.getAttribute('aria-selected') === 'true' || i.classList.contains('selected'))).filter(o => o.text) }; \
      if (e.classList.contains('dropdown') || e.classList.contains('ui')) { \
        const options = [...e.querySelectorAll('.item, .option, [data-value]')] \
          .map(i => item(i, i.classList.contains('selected') || i.classList.contains('active'))).filter(o => o.text); \
        if (options.length) return { type: 'a custom', options }; } \
      return null; }; \
    const search = (e, depth) => { if (depth >= 4) return null; \
      for (const child of e.children) { const found = check(child); \
        if (found) return { ...found, source: `a child ${depth + 1} deep` }; \
        const deeper = search(child, depth + 1); if (deeper) return deeper; } \
      return null; }; \
    const found = check(this); if (found) return { ...found, source: 'target' }; \
    return search(this, 0) || { error: `this element and its children, 4 deep, are no dropdown \
      (tag ${this.tagName.toLowerCase()}, role ${this.getAttribute('role')}, classes ${this.className})` }; }";

/// Selects the option whose text or value is `want`, ignoring case, in a `<select>`, a menu or
/// listbox, or a Semantic UI dropdown, this element or the first within it, four deep.
const SELECT_OPTION: &str = "function(want) { const lower = want.toLowerCase(); \
    const listed = items => items.map(i => ({ text: (i.textContent || '').trim(), value: i.getAttribute('data-value') || '' })) \
      .filter(o => o.text || o.value); \
    const attempt = e => { \
      if (e.tagName.toLowerCase() === 'select') { \
        const o = [...e.options].find(o => o.text.trim().toLowerCase() === lower || o.value.toLowerCase() === lower); \
        if (!o) return { success: false, error: `no option with text or value '${want}' in the <select>`, \
          availableOptions: [...e.options].map(o => ({ text: o.text.trim(), value: o.value })) }; \
        e.focus(); e.value = o.value; o.selected = true; e.selectedIndex = o.index; \
        e.dispatchEvent(new Event('input', { bubbles: true, cancelable: true })); \
        e.dispatchEvent(new Event('change', { bubbles: true, cancelable: true })); e.blur(); \
        if (e.value !== o.value) return { success: false, \
          error: 'the selection was set, but the page reverted it: the dropdown may need clicking' }; \
        return { success: true, message: `Selected ${o.text.trim()} (value ${o.value})` }; } \
      const role = e.getAttribute('role'); \
      const custom = e.classList.contains('dropdown') || e.classList.contains('ui'); \
      if (role === 'menu' || role === 'listbox' || role === 'combobox' || custom) { \
        const items = [...e.querySelectorAll(custom ? '.item, .option, [data-value]' : '[role=menuitem], [role=option]')]; \
        const hit = items.find(i => (i.textContent || '').trim().toLowerCase() === lower \
          || (i.getAttribute('data-value') || '').toLowerCase() === lower); \
        if (!hit) return { success: false, error: `no item with text or value '${want}' in the dropdown`, \
          availableOptions: listed(items) }; \
        hit.click(); \
        if (custom) { const t = e.querySelector('.text'); if (t) t.textContent = hit.textContent.trim(); \
          e.dispatchEvent(new Event('change', { bubbles: true })); } \
        return { success: true, message: `Selected ${hit.textContent.trim()}` }; } \
      return null; }; \
    const search = (e, depth) => { if (depth >= 4) return null; \
      for (const child of e.children) { const r = attempt(child) || search(child, depth + 1); if (r) return r; } \
      return null; }; \
    return attempt(this) || search(this, 0) || { success: false, \
      error: `this element and its children, 4 deep, are no dropdown (tag ${this.tagName.toLowerCase()})` }; }";

/// Every backend id in a `DOM.describeNode` tree.
fn collect_backends(node: &Value, into: &mut HashSet<i64>) {
    if let Some(id) = node["backendNodeId"].as_i64() {
        into.insert(id);
    }
    for key in ["children", "shadowRoots"] {
        for child in node[key].as_array().into_iter().flatten() {
            collect_backends(child, into);
        }
    }
    if node.get("contentDocument").is_some() {
        collect_backends(&node["contentDocument"], into);
    }
}

/// What `browser_search_page` is asked for besides its pattern.
pub(super) struct Search<'a> {
    pub regex: bool,
    pub case_sensitive: bool,
    pub context_chars: usize,
    pub css_scope: Option<&'a str>,
    pub max_results: usize,
}

/// Fixes what models get wrong in JavaScript: escaped quotes, doubled
/// backslashes in regular expressions, and double quotes inside a selector's double quotes.
fn fix_javascript(code: &str) -> String {
    let replace = |code: String, pattern: &str, with: &str| match fancy_regex::Regex::new(pattern) {
        Ok(re) => re.replace_all(&code, with).into_owned(),
        Err(_) => code,
    };
    let code = replace(code.to_string(), r#"\\""#, "\"");
    let code = replace(code, r"\\\\([dDsSwWbBnrtfv])", r"\$1");
    let code = replace(code, r"\\\\([.*+?^${}()|\[\]])", r"\$1");
    let code = replace(
        code,
        r#"document\.evaluate\s*\(\s*"([^"]*)"\s*,"#,
        "document.evaluate(`$1`,",
    );
    let code = replace(
        code,
        r#"(querySelector(?:All)?)\s*\(\s*"([^"]*)"\s*\)"#,
        "$1(`$2`)",
    );
    let code = replace(code, r#"\.closest\s*\(\s*"([^"]*)"\s*\)"#, ".closest(`$1`)");
    replace(code, r#"\.matches\s*\(\s*"([^"]*)"\s*\)"#, ".matches(`$1`)")
}

/// Searches the text of `CSS_SCOPE` (or the body) joined across its text nodes,
/// each match with `CONTEXT_CHARS` around it and the path of the element it starts in.
const SEARCH_PAGE: &str = r#"try {
    var scope = CSS_SCOPE ? document.querySelector(CSS_SCOPE) : document.body;
    if (!scope) return {error: 'CSS scope selector not found: ' + CSS_SCOPE, matches: [], total: 0};
    var walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    var fullText = '';
    var nodeOffsets = [];
    while (walker.nextNode()) {
        var node = walker.currentNode;
        var text = node.textContent;
        if (text && text.trim()) {
            nodeOffsets.push({offset: fullText.length, length: text.length, node: node});
            fullText += text;
        }
    }
    var re;
    try {
        var flags = CASE_SENSITIVE ? 'g' : 'gi';
        re = IS_REGEX ? new RegExp(PATTERN, flags) : new RegExp(PATTERN.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), flags);
    } catch (e) {
        return {error: 'Invalid regex pattern: ' + e.message, matches: [], total: 0};
    }
    var matches = [];
    var match;
    var totalFound = 0;
    while ((match = re.exec(fullText)) !== null) {
        totalFound++;
        if (matches.length < MAX_RESULTS) {
            var start = Math.max(0, match.index - CONTEXT_CHARS);
            var end = Math.min(fullText.length, match.index + match[0].length + CONTEXT_CHARS);
            var context = fullText.slice(start, end);
            var elementPath = '';
            for (var i = 0; i < nodeOffsets.length; i++) {
                var no = nodeOffsets[i];
                if (no.offset <= match.index && no.offset + no.length > match.index) {
                    elementPath = _getPath(no.node.parentElement);
                    break;
                }
            }
            matches.push({
                match_text: match[0],
                context: (start > 0 ? '...' : '') + context + (end < fullText.length ? '...' : ''),
                element_path: elementPath,
                char_position: match.index
            });
        }
        if (match[0].length === 0) re.lastIndex++;
    }
    return {matches: matches, total: totalFound, has_more: totalFound > MAX_RESULTS};
} catch (e) {
    return {error: 'search_page error: ' + e.message, matches: [], total: 0};
}
function _getPath(el) {
    var parts = [];
    var current = el;
    while (current && current !== document.body && current !== document) {
        var desc = current.tagName ? current.tagName.toLowerCase() : '';
        if (!desc) break;
        if (current.id) desc += '#' + current.id;
        else if (current.className && typeof current.className === 'string') {
            var classes = current.className.trim().split(/\s+/).slice(0, 2).join('.');
            if (classes) desc += '.' + classes;
        }
        parts.unshift(desc);
        current = current.parentElement;
    }
    return parts.join(' > ');
}"#;

/// Queries elements: each match's tag, text, the attributes asked for (`src` and
/// `href` as absolute URLs), and how many children it has.
const FIND_ELEMENTS: &str = r#"try {
    var elements;
    try {
        elements = document.querySelectorAll(SELECTOR);
    } catch (e) {
        return {error: 'Invalid CSS selector: ' + e.message, elements: [], total: 0};
    }
    var total = elements.length;
    var limit = Math.min(total, MAX_RESULTS);
    var results = [];
    for (var i = 0; i < limit; i++) {
        var el = elements[i];
        var item = {index: i, tag: el.tagName.toLowerCase()};
        if (INCLUDE_TEXT) {
            var text = (el.textContent || '').trim();
            item.text = text.length > 300 ? text.slice(0, 300) + '...' : text;
        }
        if (ATTRIBUTES && ATTRIBUTES.length > 0) {
            item.attrs = {};
            for (var j = 0; j < ATTRIBUTES.length; j++) {
                var attrName = ATTRIBUTES[j];
                var val;
                if ((attrName === 'src' || attrName === 'href') && typeof el[attrName] === 'string' && el[attrName] !== '') {
                    val = el[attrName];
                } else {
                    val = el.getAttribute(attrName);
                }
                if (val !== null) item.attrs[attrName] = val.length > 500 ? val.slice(0, 500) + '...' : val;
            }
        }
        item.children_count = el.children.length;
        results.push(item);
    }
    return {elements: results, total: total, showing: limit};
} catch (e) {
    return {error: 'find_elements error: ' + e.message, elements: [], total: 0};
}"#;

/// Whether `url` is a browser's new-tab page.
fn is_new_tab_page(url: &str) -> bool {
    matches!(
        url,
        "about:blank" | "chrome://newtab/" | "chrome://new-tab-page/"
    ) || url.starts_with("chrome://newtab")
        || url.starts_with("chrome://new-tab-page")
}

fn article(kind: &str) -> String {
    match kind {
        "alert" => "an alert".into(),
        "beforeunload" => "a leave-page check".into(),
        other => format!("a {other}"),
    }
}

/// Key names taken for others: `ctrl` for `Control`, `esc` for `Escape`, ….
fn alias(name: &str) -> Option<&'static str> {
    Some(match name.to_lowercase().as_str() {
        "ctrl" | "control" => "Control",
        "alt" | "option" => "Alt",
        "meta" | "cmd" | "command" => "Meta",
        "shift" => "Shift",
        "enter" | "return" => "Enter",
        "tab" => "Tab",
        "delete" => "Delete",
        "backspace" => "Backspace",
        "escape" | "esc" => "Escape",
        "space" => " ",
        "up" => "ArrowUp",
        "down" => "ArrowDown",
        "left" => "ArrowLeft",
        "right" => "ArrowRight",
        "pageup" => "PageUp",
        "pagedown" => "PageDown",
        "home" => "Home",
        "end" => "End",
        _ => return None,
    })
}

/// CDP's modifier bits.
fn modifier_bit(key: &str) -> Option<i64> {
    Some(match key {
        "Alt" => 1,
        "Control" => 2,
        "Meta" => 4,
        "Shift" => 8,
        _ => return None,
    })
}

/// Keys pressed as a key, not typed as text.
const SPECIAL_KEYS: [&str; 29] = [
    "Enter",
    "Tab",
    "Delete",
    "Backspace",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "PageUp",
    "PageDown",
    "Home",
    "End",
    "Control",
    "Alt",
    "Meta",
    "Shift",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
];

/// `Control+Shift+a` as its modifiers and main key; `None` unless every part before the last `+`
/// is a modifier. `Control++` is the plus key.
fn combination(keys: &str) -> Option<(Vec<String>, String)> {
    if !keys.contains('+') || keys == "+" {
        return None;
    }
    let (prefix, main) = match keys.strip_suffix("++") {
        Some(prefix) => (prefix, "+".to_string()),
        None => {
            let (prefix, suffix) = keys.rsplit_once('+')?;
            if suffix.trim().is_empty() {
                return None;
            }
            let main = alias(suffix.trim()).map_or_else(|| suffix.to_string(), str::to_string);
            (prefix, main)
        }
    };
    let parts: Vec<&str> = prefix.split('+').collect();
    if parts.iter().any(|p| p.trim().is_empty()) {
        return None;
    }
    let modifiers: Vec<String> = parts
        .iter()
        .map(|p| alias(p.trim()).map_or_else(|| p.to_string(), str::to_string))
        .collect();
    modifiers
        .iter()
        .all(|m| modifier_bit(m).is_some())
        .then_some((modifiers, main))
}

/// A key's `code` and Windows virtual key code.
fn key_code(key: &str) -> (String, Option<i64>) {
    let named = |code: &str, n: i64| (code.to_string(), Some(n));
    match key {
        "Backspace" => named("Backspace", 8),
        "Tab" => named("Tab", 9),
        "Enter" => named("Enter", 13),
        "Escape" => named("Escape", 27),
        "Space" | " " => named("Space", 32),
        "PageUp" => named("PageUp", 33),
        "PageDown" => named("PageDown", 34),
        "End" => named("End", 35),
        "Home" => named("Home", 36),
        "ArrowLeft" => named("ArrowLeft", 37),
        "ArrowUp" => named("ArrowUp", 38),
        "ArrowRight" => named("ArrowRight", 39),
        "ArrowDown" => named("ArrowDown", 40),
        "Insert" => named("Insert", 45),
        "Delete" => named("Delete", 46),
        "Shift" | "ShiftLeft" => named("ShiftLeft", 16),
        "ShiftRight" => named("ShiftRight", 16),
        "Control" | "ControlLeft" => named("ControlLeft", 17),
        "ControlRight" => named("ControlRight", 17),
        "Alt" | "AltLeft" => named("AltLeft", 18),
        "AltRight" => named("AltRight", 18),
        "Meta" | "MetaLeft" => named("MetaLeft", 91),
        "MetaRight" => named("MetaRight", 92),
        "NumLock" => named("NumLock", 144),
        "NumpadMultiply" => named("NumpadMultiply", 106),
        "NumpadAdd" => named("NumpadAdd", 107),
        "NumpadSubtract" => named("NumpadSubtract", 109),
        "NumpadDecimal" => named("NumpadDecimal", 110),
        "NumpadDivide" => named("NumpadDivide", 111),
        "CapsLock" => named("CapsLock", 20),
        "ScrollLock" => named("ScrollLock", 145),
        "Semicolon" | ";" => named("Semicolon", 186),
        "Equal" | "=" => named("Equal", 187),
        "Comma" | "," => named("Comma", 188),
        "Minus" | "-" => named("Minus", 189),
        "Period" | "." => named("Period", 190),
        "Slash" | "/" => named("Slash", 191),
        "Backquote" | "`" => named("Backquote", 192),
        "BracketLeft" | "[" => named("BracketLeft", 219),
        "Backslash" | "\\" => named("Backslash", 220),
        "BracketRight" | "]" => named("BracketRight", 221),
        "Quote" | "'" => named("Quote", 222),
        "AudioVolumeMute" => named("AudioVolumeMute", 173),
        "AudioVolumeDown" => named("AudioVolumeDown", 174),
        "AudioVolumeUp" => named("AudioVolumeUp", 175),
        "MediaTrackNext" => named("MediaTrackNext", 176),
        "MediaTrackPrevious" => named("MediaTrackPrevious", 177),
        "MediaStop" => named("MediaStop", 178),
        "MediaPlayPause" => named("MediaPlayPause", 179),
        "BrowserBack" => named("BrowserBack", 166),
        "BrowserForward" => named("BrowserForward", 167),
        "BrowserRefresh" => named("BrowserRefresh", 168),
        "BrowserStop" => named("BrowserStop", 169),
        "BrowserSearch" => named("BrowserSearch", 170),
        "BrowserFavorites" => named("BrowserFavorites", 171),
        "BrowserHome" => named("BrowserHome", 172),
        "Clear" => named("Clear", 12),
        "Pause" => named("Pause", 19),
        "Select" => named("Select", 41),
        "Print" => named("Print", 42),
        "Execute" => named("Execute", 43),
        "PrintScreen" => named("PrintScreen", 44),
        "Help" => named("Help", 47),
        "ContextMenu" => named("ContextMenu", 93),
        _ => {
            if let Some(n) = key.strip_prefix('F').and_then(|n| n.parse::<i64>().ok())
                && (1..=24).contains(&n)
            {
                return (key.to_string(), Some(111 + n));
            }
            if let Some(n) = key
                .strip_prefix("Numpad")
                .and_then(|n| n.parse::<i64>().ok())
                && (0..=9).contains(&n)
            {
                return (key.to_string(), Some(96 + n));
            }
            let mut chars = key.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_alphabetic() => {
                    let upper = upper_one(c);
                    (format!("Key{upper}"), Some(upper as i64))
                }
                (Some(c), None) if c.is_ascii_digit() => (format!("Digit{c}"), Some(c as i64)),
                _ => (key.to_string(), None),
            }
        }
    }
}

/// `c` in upper case when that is one character, else `c`.
fn upper_one(c: char) -> char {
    let mut upper = c.to_uppercase();
    match (upper.next(), upper.next()) {
        (Some(u), None) => u,
        _ => c,
    }
}

/// The modifiers, Windows key code, and unshifted key a character is typed with on a US layout.
fn char_key(c: char) -> (i64, i64, String) {
    const SHIFTED: [(char, char, i64); 21] = [
        ('!', '1', 49),
        ('@', '2', 50),
        ('#', '3', 51),
        ('$', '4', 52),
        ('%', '5', 53),
        ('^', '6', 54),
        ('&', '7', 55),
        ('*', '8', 56),
        ('(', '9', 57),
        (')', '0', 48),
        ('_', '-', 189),
        ('+', '=', 187),
        ('{', '[', 219),
        ('}', ']', 221),
        ('|', '\\', 220),
        (':', ';', 186),
        ('"', '\'', 222),
        ('<', ',', 188),
        ('>', '.', 190),
        ('?', '/', 191),
        ('~', '`', 192),
    ];
    const PLAIN: [(char, i64); 12] = [
        (' ', 32),
        ('-', 189),
        ('=', 187),
        ('[', 219),
        (']', 221),
        ('\\', 220),
        (';', 186),
        ('\'', 222),
        (',', 188),
        ('.', 190),
        ('/', 191),
        ('`', 192),
    ];
    if let Some(&(_, base, n)) = SHIFTED.iter().find(|(s, ..)| *s == c) {
        return (8, n, base.to_string());
    }
    if c.is_uppercase() {
        let lower = c.to_lowercase().next().unwrap_or(c);
        return (8, c as i64, lower.to_string());
    }
    if c.is_lowercase() {
        return (0, upper_one(c) as i64, c.to_string());
    }
    if let Some(&(_, n)) = PLAIN.iter().find(|(p, _)| *p == c) {
        return (0, n, c.to_string());
    }
    let n = if c.is_alphabetic() { upper_one(c) } else { c };
    (0, n as i64, c.to_string())
}

/// The `code` of the key that types `key`.
fn char_code(key: &str) -> String {
    let mut chars = key.chars();
    let Some(c) = chars.next() else {
        return String::new();
    };
    if chars.next().is_none() && c.is_ascii_digit() {
        return format!("Digit{c}");
    }
    if c.is_alphabetic() {
        return format!("Key{}", key.to_uppercase());
    }
    let code = match c {
        ' ' => "Space",
        '.' => "Period",
        ',' => "Comma",
        '-' | '_' => "Minus",
        '@' => "Digit2",
        '!' => "Digit1",
        '?' | '/' => "Slash",
        ':' | ';' => "Semicolon",
        '(' => "Digit9",
        ')' => "Digit0",
        '[' | '{' => "BracketLeft",
        ']' | '}' => "BracketRight",
        '\\' => "Backslash",
        '=' | '+' => "Equal",
        '*' => "Digit8",
        '&' => "Digit7",
        '%' => "Digit5",
        '$' => "Digit4",
        '#' => "Digit3",
        '^' => "Digit6",
        '~' | '`' => "Backquote",
        '\'' | '"' => "Quote",
        _ => return format!("Key{}", key.to_uppercase()),
    };
    code.to_string()
}

/// The elements a click, mouse or pointer listener is on; `getEventListeners` is there only
/// with the command line API. Null on a heavy page, or one with too many to look up.
const LISTENED: &str = r#"(() => {
    if (typeof getEventListeners !== 'function') return null;
    const all = document.querySelectorAll('*');
    if (all.length > 10000) return null;
    const found = [];
    for (const el of all) {
        try {
            const l = getEventListeners(el);
            if (l.click || l.mousedown || l.mouseup || l.pointerdown || l.pointerup) {
                found.push(el);
                if (found.length > 100) return null;
            }
        } catch (e) {}
    }
    return found;
})()"#;

/// Whether `this` is a <select>, a file input or a print button, which a click cannot work, and a short
/// description of a clicked element: tag, type, role, checkbox state, text, id,
/// name and aria-label.
const CLICK_TARGET: &str = r#"function() {
    const tag = this.tagName.toLowerCase();
    const type = this.getAttribute('type') || '';
    if (tag === 'select') return { kind: 'select' };
    if (tag === 'input' && type.toLowerCase() === 'file') return { kind: 'file' };
    if ((this.getAttribute('onclick') || '').toLowerCase().includes('print')) return { kind: 'print' };
    const state = on => `checkbox-state=${on ? 'checked' : 'unchecked'}`;
    const parts = [tag];
    if (tag === 'input' && type) {
        parts.push(`type=${type}`);
        if (type === 'checkbox') parts.push(state(this.checked));
    }
    const role = this.getAttribute('role');
    if (role) {
        parts.push(`role=${role}`);
        if (role === 'checkbox') parts.push(state(['true', 'checked'].includes((this.getAttribute('aria-checked') || '').toLowerCase())));
    }
    if (['label', 'span', 'div'].includes(tag) && !parts.some(p => p.startsWith('type='))) {
        for (const child of this.children) {
            if (child.tagName === 'INPUT' && child.getAttribute('type') === 'checkbox') {
                const hidden = ['0', '0.0'].includes(getComputedStyle(child).opacity) || !child.checkVisibility();
                if (hidden) {
                    parts.push(state(child.checked));
                    break;
                }
            }
        }
    }
    const text = (this.textContent || '').trim();
    if (text) parts.push(`"${text.slice(0, 30)}${text.length > 30 ? '...' : ''}"`);
    for (const attr of ['id', 'name', 'aria-label']) {
        const value = this.getAttribute(attr);
        if (value) parts.push(`${attr}=${value.slice(0, 20)}`);
    }
    return { kind: 'element', description: parts.join(' ') };
}"#;

/// Whether `this` is clickable at (x, y): it, or what holds it or is in it, is on top there, or
/// a label tied to it.
const OCCLUSION: &str = r#"function(x, y) {
    const at = document.elementFromPoint(x, y);
    if (!at) return false;
    if (this === at || this.contains(at) || at.contains(this)) return true;
    if (this.tagName === 'INPUT' && this.id) {
        const label = document.querySelector('label[for="' + CSS.escape(this.id) + '"]');
        if (label && (label === at || label.contains(at))) return true;
    }
    if (this.tagName === 'INPUT') {
        let up = at;
        for (let i = 0; i < 3 && up; i++) {
            if (up.tagName === 'LABEL' && up.contains(this)) return true;
            up = up.parentElement;
        }
    }
    if (this.tagName === 'LABEL' && at.tagName === 'INPUT'
        && ((this.htmlFor && at.id === this.htmlFor) || this.contains(at))) return true;
    return false;
}"#;

/// What typing into `this` turns on: whether its value is set outright (date, color, range,
/// datepickers), whether it is a contenteditable, and whether it suggests as typed.
const INPUT_TRAITS: &str = r#"function() {
    const tag = this.tagName.toLowerCase();
    const type = (this.getAttribute('type') || '').toLowerCase();
    const cls = (this.getAttribute('class') || '').toLowerCase();
    const direct = tag === 'input' && (
        ['date', 'time', 'datetime-local', 'month', 'week', 'color', 'range'].includes(type)
        || (['text', ''].includes(type) && (
            ['datepicker', 'daterangepicker', 'datetimepicker', 'bootstrap-datepicker'].some(i => cls.includes(i))
            || ['data-datepicker', 'data-date-format', 'data-provide'].some(a => this.hasAttribute(a)))));
    const ce = this.getAttribute('contenteditable');
    const role = this.getAttribute('role');
    const editable = ce === 'true' || ce === '' || (role === 'textbox' && tag !== 'input' && tag !== 'textarea');
    const ac = this.getAttribute('aria-autocomplete') || '';
    const popup = this.getAttribute('aria-haspopup') || '';
    const waits = role === 'combobox' || (ac !== '' && ac !== 'none');
    const autocomplete = waits || !!this.getAttribute('list')
        || (popup !== '' && popup !== 'false' && !!(this.getAttribute('aria-controls') || this.getAttribute('aria-owns')));
    return { direct, editable, autocomplete, waits };
}"#;

/// Empties `this` by script. React tracks the instance's value setter, so an input or textarea
/// is set through its prototype's.
const CLEAR_FIELD: &str = r#"function() {
    if (this.getAttribute('contenteditable') === 'true' || this.getAttribute('contenteditable') === ''
        || this.isContentEditable === true) {
        while (this.firstChild) this.removeChild(this.firstChild);
        this.textContent = '';
        this.innerHTML = '';
        this.focus();
        const selection = window.getSelection();
        const range = document.createRange();
        range.setStart(this, 0);
        range.setEnd(this, 0);
        selection.removeAllRanges();
        selection.addRange(range);
        this.dispatchEvent(new Event('input', { bubbles: true }));
        this.dispatchEvent(new Event('change', { bubbles: true }));
        return { cleared: true, finalText: this.textContent };
    }
    if (this.value !== undefined) {
        try { this.select(); } catch (e) {}
        if (this instanceof HTMLInputElement || this instanceof HTMLTextAreaElement) {
            const proto = this instanceof HTMLTextAreaElement
                ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
            try { Object.getOwnPropertyDescriptor(proto, 'value').set.call(this, ''); }
            catch (e) { this.value = ''; }
        } else {
            this.value = '';
        }
        this.dispatchEvent(new Event('input', { bubbles: true }));
        this.dispatchEvent(new Event('change', { bubbles: true }));
        return { cleared: true, finalText: this.value };
    }
    return { cleared: false };
}"#;

/// Sets an input's value outright, then sends what a person's edit would: focus, input,
/// change, blur, and jQuery's change.
const SET_VALUE: &str = r#"function(text) {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(this, text);
    this.dispatchEvent(new FocusEvent('focus', { bubbles: true }));
    this.dispatchEvent(new Event('input', { bubbles: true, cancelable: true }));
    this.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
    this.dispatchEvent(new FocusEvent('blur', { bubbles: true }));
    if (typeof jQuery !== 'undefined' && jQuery.fn) {
        try {
            jQuery(this).trigger('change');
            if (jQuery(this).data('datepicker')) jQuery(this).datepicker('update');
        } catch (e) {}
    }
    return this.value;
}"#;

/// The events frameworks update their state on after typing: input, change, blur, and React's
/// and Vue's own.
const FRAMEWORK_EVENTS: &str = r#"function() {
    this.focus();
    this.dispatchEvent(new InputEvent('input', { bubbles: true, cancelable: true, data: this.value, inputType: 'insertText' }));
    this.dispatchEvent(new Event('change', { bubbles: true, cancelable: true }));
    this.dispatchEvent(new Event('blur', { bubbles: true, cancelable: true }));
    if (this._reactInternalFiber || this._reactInternalInstance || this.__reactInternalInstance) {
        const event = new InputEvent('input', { bubbles: true, cancelable: true, data: this.value });
        Object.defineProperty(event, 'isTrusted', { value: true });
        this.dispatchEvent(event);
    }
    if (this.__vue__ || this._vnode || this.__vueParentComponent) {
        setTimeout(() => this.dispatchEvent(new Event('input', { bubbles: true })), 0);
    }
    return true;
}"#;

/// Replaces what `this` holds with `text`, through the setter its own element type has.
const REPLACE_VALUE: &str = r#"function(text) {
    if (this.value !== undefined) {
        let desc = null;
        if (this instanceof HTMLInputElement) desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
        else if (this instanceof HTMLTextAreaElement) desc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
        if (desc && desc.set) {
            try { desc.set.call(this, text); } catch (e) { this.value = text; }
        } else {
            this.value = text;
        }
    } else if (this.isContentEditable) {
        this.textContent = text;
    }
    this.dispatchEvent(new Event('input', { bubbles: true }));
    this.dispatchEvent(new Event('change', { bubbles: true }));
    return this.value !== undefined ? this.value : this.textContent;
}"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shortcut_is_its_modifiers_and_a_key() {
        let mods = |m: &[&str]| m.iter().map(|m| m.to_string()).collect::<Vec<_>>();
        assert_eq!(
            combination("Control+a"),
            Some((mods(&["Control"]), "a".into()))
        );
        assert_eq!(
            combination("ctrl+shift+tab"),
            Some((mods(&["Control", "Shift"]), "Tab".into()))
        );
        assert_eq!(combination("Meta++"), Some((mods(&["Meta"]), "+".into())));
        assert_eq!(combination("a+b"), None, "text with a plus in it");
        assert_eq!(combination("Enter"), None);
        assert_eq!(key_code("Enter"), ("Enter".into(), Some(13)));
        assert_eq!(key_code("F5"), ("F5".into(), Some(116)));
        assert_eq!(key_code("a"), ("KeyA".into(), Some(65)));
    }

    #[test]
    fn a_character_is_typed_with_the_key_a_us_layout_has_for_it() {
        assert_eq!(char_key('a'), (0, 65, "a".into()));
        assert_eq!(char_key('A'), (8, 65, "a".into()));
        assert_eq!(char_key('@'), (8, 50, "2".into()));
        assert_eq!(char_key('.'), (0, 190, ".".into()));
        assert_eq!(char_code("2"), "Digit2");
        assert_eq!(char_code("a"), "KeyA");
        assert_eq!(char_code(" "), "Space");
    }

    #[test]
    fn javascript_over_escaped_by_a_model_is_fixed() {
        assert_eq!(
            fix_javascript(r#"document.querySelector(\"a[href*=\\"x\\"]\")"#),
            r#"document.querySelector("a[href*=\"x\"]")"#
        );
        assert_eq!(fix_javascript(r"/\\d+\\./"), r"/\d+\./");
        assert_eq!(
            fix_javascript(r#"el.closest("div[data-id='1']")"#),
            "el.closest(`div[data-id='1']`)"
        );
    }
}
