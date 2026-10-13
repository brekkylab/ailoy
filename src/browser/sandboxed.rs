//! Chromium in a console of its own, which a person watches and uses through `/abin/viewer` on a
//! port published to this host: [`Browser::sandboxed`](super::Browser::sandboxed).

use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use anyhow::Context as _;
use tokio::sync::{Mutex, OnceCell};
use virtx::{console::ConsoleClient, image::Recipe, protocol::Port};

use super::{free_port, require, run, token, wait_for};

/// Where Chromium's DevTools and the viewer listen inside the browser console.
const CDP_PORT: u16 = 9222;
const VIEWER_PORT: u16 = 4849;
/// HeadlessChrome in the user agent is what sites turn away.
const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36";

#[derive(Default)]
pub(super) struct Sandboxed {
    running: OnceCell<Running>,
    /// The last viewer event already told to the agent.
    seen: AtomicU64,
}

struct Running {
    /// Held so the browser lives as long as the agent; never used after boot.
    _console: Mutex<ConsoleClient>,
    #[cfg_attr(not(test), allow(dead_code))]
    cdp_port: u16,
    viewer_port: u16,
    token: String,
    http: reqwest::Client,
}

impl Sandboxed {
    /// Where a person watches and uses the browser, token included, once it has booted.
    pub(super) fn url(&self) -> Option<String> {
        let running = self.running.get()?;
        Some(format!(
            "http://localhost:{}/#t={}",
            running.viewer_port, running.token
        ))
    }

    /// Boots the browser if it is not up, and points `agent`'s agent-browser at it.
    pub(super) async fn ensure(
        &self,
        agent: &mut ConsoleClient,
        locale: Option<&str>,
    ) -> anyhow::Result<()> {
        self.running
            .get_or_try_init(|| boot(agent, locale))
            .await
            .map(|_| ())
    }

    /// Shows `message` to the person; with `wait`, with a Done button, returning once they
    /// press it.
    pub(super) async fn request(&self, message: &str, wait: bool) -> anyhow::Result<()> {
        let body = serde_json::json!({ "message": message, "wait": wait });
        let mut call = self.control(reqwest::Method::POST, "request")?.json(&body);
        if !wait {
            call = call.timeout(Duration::from_secs(10));
        }
        call.send().await?.error_for_status()?;
        Ok(())
    }

    /// Takes the request down.
    pub(super) async fn clear(&self) -> anyhow::Result<()> {
        self.control(reqwest::Method::POST, "clear")?
            .timeout(Duration::from_secs(10))
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }

    /// What the person did in the viewer since the agent last heard, as one note for its next
    /// message; with `clear`, the request it was answering taken down.
    pub(super) async fn catch_up(&self, clear: bool) -> anyhow::Result<Option<String>> {
        if self.running.get().is_none() {
            return Ok(None);
        }
        let since = self.seen.load(Ordering::Relaxed);
        let answer: serde_json::Value = self
            .control(reqwest::Method::GET, &format!("events?since={since}"))?
            .timeout(Duration::from_secs(10))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if clear {
            self.clear().await?;
        }
        if let Some(next) = answer["next"].as_u64() {
            self.seen.store(next, Ordering::Relaxed);
        }
        let events = answer["events"].as_array().cloned().unwrap_or_default();
        Ok(describe_events(&events))
    }

    fn control(
        &self,
        method: reqwest::Method,
        path: &str,
    ) -> anyhow::Result<reqwest::RequestBuilder> {
        let running = self.running.get().context("the browser is not running")?;
        Ok(running
            .http
            .request(
                method,
                format!("http://127.0.0.1:{}/control/{path}", running.viewer_port),
            )
            .bearer_auth(&running.token))
    }
}

/// The browser console, Chromium and the viewer in it, and the agent's agent-browser pointed
/// at them.
async fn boot(agent: &mut ConsoleClient, locale: Option<&str>) -> anyhow::Result<Running> {
    let locale = match locale {
        Some(locale) => {
            language_tag(locale).with_context(|| format!("{locale:?} is not a language tag"))?
        }
        None => sys_locale::get_locale()
            .as_deref()
            .and_then(language_tag)
            .unwrap_or_else(|| "en-US".into()),
    };
    let accept = accept_languages(&locale);
    let (cdp_port, viewer_port) = (free_port()?, free_port()?);
    let mut console = ConsoleClient::builder()
        .image(
            Recipe::new("debian:trixie-slim").step(
                "apt-get update \
                 && apt-get install -y --no-install-recommends chromium fonts-noto-cjk ca-certificates \
                 && rm -rf /var/lib/apt/lists/* /var/log/journal",
            ),
        )
        .network(true)
        .ports([
            format!("{cdp_port}:{CDP_PORT}").parse::<Port>()?,
            format!("{viewer_port}:{VIEWER_PORT}").parse::<Port>()?,
        ])
        .vcpus(2)
        .memory_mib(2048)
        .build()
        .await
        .context("booting the browser console")?;

    require(&mut console, "/abin/viewer").await?;
    require(agent, "/abin/agent-browser").await?;

    let token = token();
    run(
        &mut console,
        &format!(
            "setsid nohup chromium --headless --no-sandbox --user-data-dir=/root/chromium \
             --window-size=1280,860 --lang={locale} --accept-lang={accept} \
             --user-agent='{USER_AGENT}' --remote-debugging-port={CDP_PORT} about:blank >/tmp/chromium.log 2>&1 &"
        ),
    )
    .await?;
    let http = reqwest::Client::builder().build()?;
    let first_tab = first_tab(&http, cdp_port).await?;
    run(
        &mut console,
        &format!(
            "setsid nohup /abin/viewer --cdp 127.0.0.1:{CDP_PORT} --listen 127.0.0.1:{VIEWER_PORT} \
             --token {token} >/tmp/viewer.log 2>&1 &"
        ),
    )
    .await?;

    // Pinned, so a tab the person opens does not pull the agent away; bound beforehand to
    // Chromium's first tab, which a pinned session would otherwise leave behind for one of its own.
    let config = serde_json::json!({ "cdp": cdp_port.to_string(), "pinTab": true });
    let binding =
        serde_json::json!({ "targetId": first_tab, "url": "about:blank", "pinned": true });
    run(
        agent,
        &format!(
            "mkdir -p \"$HOME/.agent-browser\" \
             && printf '%s' '{config}' > \"$HOME/.agent-browser/config.json\" \
             && printf '%s' '{binding}' > \"$HOME/.agent-browser/default.target\""
        ),
    )
    .await?;

    wait_for(&http, &format!("http://127.0.0.1:{viewer_port}/"))
        .await
        .context("the viewer did not come up")?;
    Ok(Running {
        _console: Mutex::new(console),
        cdp_port,
        viewer_port,
        token,
        http,
    })
}

/// Chromium's first page, once its DevTools answer.
async fn first_tab(http: &reqwest::Client, cdp_port: u16) -> anyhow::Result<String> {
    let url = format!("http://127.0.0.1:{cdp_port}/json/list");
    wait_for(http, &url)
        .await
        .context("Chromium did not come up")?;
    let targets: serde_json::Value = http.get(&url).send().await?.json().await?;
    targets
        .as_array()
        .and_then(|all| all.iter().find(|t| t["type"] == "page"))
        .and_then(|t| t["id"].as_str())
        .map(str::to_string)
        .context("Chromium has no page")
}

/// `locale` as a BCP 47 tag (`ko_KR.UTF-8` is `ko-KR`), or `None` for one that names no
/// language (`C`, `POSIX`). Only letters, digits and dashes come out, so it is safe in a shell.
fn language_tag(locale: &str) -> Option<String> {
    let tag = locale.split(['.', '@']).next()?.replace('_', "-");
    let language = tag.split('-').next()?;
    let valid = (2..=3).contains(&language.len())
        && language.bytes().all(|b| b.is_ascii_alphabetic())
        && tag.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        && !tag.ends_with('-');
    valid.then_some(tag)
}

/// The languages pages are asked for, in order: `ko-KR` asks for `ko-KR,ko,en`.
fn accept_languages(tag: &str) -> String {
    let language = tag.split('-').next().unwrap_or(tag);
    let mut accept = vec![tag];
    if language != tag {
        accept.push(language);
    }
    if language != "en" {
        accept.push("en");
    }
    accept.join(",")
}

/// The viewer's events as one sentence each, or `None` when the person did nothing.
fn describe_events(events: &[serde_json::Value]) -> Option<String> {
    let lines: Vec<String> = events
        .iter()
        .filter_map(|e| {
            let url = e["url"].as_str().unwrap_or_default();
            let whose = if e["agentTab"].as_bool() == Some(true) {
                "your tab"
            } else {
                "their own tab"
            };
            match e["kind"].as_str()? {
                "done" => Some(format!(
                    "pressed Done on your request \"{}\"",
                    e["request"].as_str().unwrap_or_default()
                )),
                "opened" => Some("opened a tab of their own".to_string()),
                "navigated" => Some(format!("moved {whose} to {url}")),
                "closed" => Some(format!("closed {whose} ({url})")),
                _ => None,
            }
        })
        .collect();
    (!lines.is_empty()).then(|| {
        format!(
            "[browser] Since your last turn, the user, in the browser viewer: {}.",
            lines.join("; ")
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_locale_becomes_a_language_tag() {
        assert_eq!(language_tag("ko_KR.UTF-8").as_deref(), Some("ko-KR"));
        assert_eq!(language_tag("ko-KR").as_deref(), Some("ko-KR"));
        assert_eq!(language_tag("de_DE@euro").as_deref(), Some("de-DE"));
        assert_eq!(language_tag("zh-Hant-TW").as_deref(), Some("zh-Hant-TW"));
        assert_eq!(language_tag("ja").as_deref(), Some("ja"));
        for no in ["C", "POSIX", "", "ko-KR'; rm -rf /", "ko-", "k1-KR"] {
            assert_eq!(language_tag(no), None, "{no:?}");
        }
        assert_eq!(accept_languages("ko-KR"), "ko-KR,ko,en");
        assert_eq!(accept_languages("en-US"), "en-US,en");
        assert_eq!(accept_languages("ja"), "ja,en");
    }

    #[test]
    fn events_become_one_note() {
        assert_eq!(describe_events(&[]), None);
        let note = describe_events(&[
            serde_json::json!({ "kind": "done", "request": "Sign in" }),
            serde_json::json!({ "kind": "navigated", "agentTab": true, "url": "https://x.test/" }),
            serde_json::json!({ "kind": "closed", "agentTab": false, "url": "https://y.test/" }),
            serde_json::json!({ "kind": "unknown" }),
        ])
        .unwrap();
        assert_eq!(
            note,
            "[browser] Since your last turn, the user, in the browser viewer: pressed Done on your \
             request \"Sign in\"; moved your tab to https://x.test/; closed their own tab \
             (https://y.test/)."
        );
    }

    /// Boots for real: needs a virtx whose `/abin` carries `viewer` and `agent-browser`
    /// (`VIRTX_HOME` naming one), and the network.
    #[tokio::test]
    #[ignore = "needs a virtx with viewer and agent-browser in /abin"]
    async fn boots_and_points_the_agent_at_the_browser() {
        let mut agent = ConsoleClient::builder()
            .image(Recipe::new("debian:trixie-slim"))
            .network(true)
            .build()
            .await
            .unwrap();
        let session = Sandboxed::default();
        assert_eq!(session.url(), None);
        session.ensure(&mut agent, Some("ja-JP")).await.unwrap();
        let url = session.url().unwrap();
        let viewer = url.split('#').next().unwrap();

        // The agent drives the tab Chromium started with, so there is still one.
        let opened = agent
            .exec(
                [
                    "sh",
                    "-c",
                    "agent-browser open https://example.com && agent-browser get title",
                ],
                Some(60_000),
            )
            .await
            .unwrap();
        assert_eq!(
            opened.code,
            0,
            "{}",
            String::from_utf8_lossy(&opened.stderr)
        );
        assert!(String::from_utf8_lossy(&opened.stdout).contains("Example Domain"));
        let languages = agent
            .exec(
                ["agent-browser", "eval", "navigator.languages.join()"],
                Some(60_000),
            )
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&languages.stdout).contains("ja-JP,ja,en"));
        let running = session.running.get().unwrap();
        let http = &running.http;
        let tabs: serde_json::Value = http
            .get(format!("http://127.0.0.1:{}/json/list", running.cdp_port))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let pages = tabs
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["type"] == "page")
            .count();
        assert_eq!(pages, 1);
        let page = http.get(viewer).send().await.unwrap();
        assert!(page.status().is_success());
        let unauthorized = http
            .get(format!("{viewer}control/events"))
            .send()
            .await
            .unwrap();
        assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);

        // EndTurn: shown at once, nothing to report, and taken down by the next catch-up.
        session.request("Sign in, please", false).await.unwrap();
        assert_eq!(session.catch_up(true).await.unwrap(), None);
    }

    /// A turn that boots the browser by driving it and ends on `ask_user`: needs the same virtx
    /// as above, and `ANTHROPIC_API_KEY`.
    #[tokio::test]
    #[ignore = "needs a virtx with viewer and agent-browser in /abin, and ANTHROPIC_API_KEY"]
    async fn a_turn_boots_the_browser_and_ends_on_ask_user() {
        use futures::StreamExt as _;

        dotenvy::dotenv().ok();
        let console = ConsoleClient::builder()
            .image(Recipe::new("debian:trixie-slim"))
            .network(true)
            .build()
            .await
            .unwrap();
        let mut agent = crate::agent::AgentBuilder::new("anthropic/claude-haiku-4-5")
            .system_tools()
            .console(console)
            .browser(super::super::Browser::sandboxed())
            .build()
            .await
            .unwrap();
        assert_eq!(agent.browser_url(), None);

        let query = crate::message::Message::new(crate::message::Role::User).with_contents([
            crate::message::Part::text(
                "Open https://example.com in the browser and read its title. Then ask the user                  to press Done, and stop.",
            ),
        ]);
        let mut called = Vec::new();
        let mut last_text = String::new();
        let mut run = agent.run(query);
        while let Some(output) = run.next().await {
            let message = output.unwrap().message;
            for call in message.tool_calls.iter().flatten() {
                if let Some((_, name, _)) = call.as_function() {
                    called.push(name.to_string());
                }
            }
            if let Some(text) = message.contents.iter().find_map(|p| p.as_text()) {
                last_text = text.to_string();
            }
        }
        drop(run);
        assert!(called.iter().any(|n| n == "shell"), "{called:?}");
        assert_eq!(
            called.last().map(String::as_str),
            Some("ask_user"),
            "{called:?}"
        );
        assert!(agent.browser_url().is_some());
        eprintln!("tools: {called:?}\nlast text: {last_text}");
    }
}
