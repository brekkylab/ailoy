//! The person's own Chromium-based browser, lent with their consent:
//! [`Browser::user`](super::Browser::user).
//!
//! It is found by the `DevToolsActivePort` file it writes in its user data directory while
//! remote debugging is on. Chrome 144 and later, with remote debugging turned on at
//! `chrome://inspect/#remote-debugging`, lets a client in only once the person allows it in a
//! prompt; this process holds that one connection for the agent's life, so the prompt comes once.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Context as _;
use serde_json::json;

use super::cdp::{Cdp, Refusal};

/// How long the person has to answer the browser's prompt.
const CONSENT_WAIT: Duration = Duration::from_secs(120);

pub(super) const DECLINED: &str = "The user declined to let you use their browser. Do not try the \
     browser again unless they ask you to in a new message; then it asks them again.";

/// A Chromium-based browser on this host, by name or by its user data directory, for
/// [`Browser::user_browser`](super::Browser::user_browser).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UserBrowser {
    Chrome,
    ChromeBeta,
    ChromeDev,
    ChromeCanary,
    Chromium,
    Edge,
    Brave,
    /// Any other Chromium-based browser, by the user data directory it writes
    /// `DevToolsActivePort` in.
    Custom(PathBuf),
}

impl UserBrowser {
    /// The named ones, in the order a search prefers them.
    pub const NAMED: [UserBrowser; 7] = [
        UserBrowser::Chrome,
        UserBrowser::ChromeBeta,
        UserBrowser::ChromeDev,
        UserBrowser::ChromeCanary,
        UserBrowser::Chromium,
        UserBrowser::Edge,
        UserBrowser::Brave,
    ];

    /// Its name as the bindings spell it: `chrome`, `chrome-beta`, …, `brave`, and `custom`
    /// for one named by its directory.
    pub fn name(&self) -> &'static str {
        match self {
            UserBrowser::Chrome => "chrome",
            UserBrowser::ChromeBeta => "chrome-beta",
            UserBrowser::ChromeDev => "chrome-dev",
            UserBrowser::ChromeCanary => "chrome-canary",
            UserBrowser::Chromium => "chromium",
            UserBrowser::Edge => "edge",
            UserBrowser::Brave => "brave",
            UserBrowser::Custom(_) => "custom",
        }
    }

    /// The named one called `name`.
    pub fn named(name: &str) -> Option<UserBrowser> {
        Self::NAMED.into_iter().find(|b| b.name() == name)
    }

    /// Where it keeps its user data on this OS (Chrome's as Chromium's `user_data_dir.md` lists
    /// them); `None` where this OS names no place for user data.
    pub fn user_data_dir(&self) -> Option<PathBuf> {
        // (macOS under Application Support, Windows under %LOCALAPPDATA%, Linux under ~/.config)
        let (mac, windows, linux) = match self {
            UserBrowser::Custom(dir) => return Some(dir.clone()),
            UserBrowser::Chrome => ("Google/Chrome", "Google/Chrome/User Data", "google-chrome"),
            UserBrowser::ChromeBeta => (
                "Google/Chrome Beta",
                "Google/Chrome Beta/User Data",
                "google-chrome-beta",
            ),
            UserBrowser::ChromeDev => (
                "Google/Chrome Dev",
                "Google/Chrome Dev/User Data",
                "google-chrome-unstable",
            ),
            UserBrowser::ChromeCanary => (
                "Google/Chrome Canary",
                "Google/Chrome SxS/User Data",
                "google-chrome-canary",
            ),
            UserBrowser::Chromium => ("Chromium", "Chromium/User Data", "chromium"),
            UserBrowser::Edge => (
                "Microsoft Edge",
                "Microsoft/Edge/User Data",
                "microsoft-edge",
            ),
            UserBrowser::Brave => (
                "BraveSoftware/Brave-Browser",
                "BraveSoftware/Brave-Browser/User Data",
                "BraveSoftware/Brave-Browser",
            ),
        };
        if cfg!(target_os = "macos") {
            Some(dirs::data_dir()?.join(mac))
        } else if cfg!(target_os = "windows") {
            Some(dirs::data_local_dir()?.join(windows))
        } else {
            // Linux, and any other Unix the same way.
            Some(dirs::config_dir()?.join(linux))
        }
    }
}

impl std::fmt::Display for UserBrowser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserBrowser::Custom(dir) => write!(f, "the browser at {}", dir.display()),
            named => f.write_str(named.name()),
        }
    }
}

pub(super) fn not_enabled(looked: &[PathBuf]) -> String {
    let looked: Vec<String> = looked
        .iter()
        .map(|dir| dir.join("DevToolsActivePort").display().to_string())
        .collect();
    format!(
        "The user's browser is not open to you: remote debugging is off, the browser is closed, \
         or it is too old (looked for {}). Ask the user to open it and turn on remote debugging; in \
         Chrome 144 or later that is at chrome://inspect/#remote-debugging.",
        looked.join(", ")
    )
}

/// Why the agent cannot have the browser now, as what it is told.
pub(super) enum Closed {
    Declined,
    NotEnabled(String),
}

/// A browser with remote debugging on: its endpoint, and when it said so.
struct Open {
    url: String,
    since: std::time::SystemTime,
}

/// The browser whose user data directory `dir` is, if it has remote debugging on: its
/// `DevToolsActivePort` names a port something listens on. A file left by a browser that has quit
/// names a port nothing does.
fn open(dir: &Path) -> Option<Open> {
    let url = devtools_url(dir)?;
    let port: u16 = url
        .strip_prefix("ws://127.0.0.1:")?
        .split('/')
        .next()?
        .parse()
        .ok()?;
    let at = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    std::net::TcpStream::connect_timeout(&at, Duration::from_millis(300)).ok()?;
    let since = std::fs::metadata(dir.join("DevToolsActivePort"))
        .and_then(|m| m.modified())
        .unwrap_or(std::time::UNIX_EPOCH);
    Some(Open { url, since })
}

/// Connects to `browser`, or, with none named, to the browser on this host that turned remote
/// debugging on last; connecting is when it asks the person.
pub(super) async fn connect(
    browser: Option<&UserBrowser>,
) -> anyhow::Result<Result<Arc<Cdp>, Closed>> {
    let candidates: Vec<UserBrowser> = match browser {
        Some(browser) => vec![browser.clone()],
        None => UserBrowser::NAMED.to_vec(),
    };
    let dirs: Vec<PathBuf> = candidates
        .iter()
        .filter_map(UserBrowser::user_data_dir)
        .collect();
    if dirs.is_empty() {
        anyhow::bail!("this OS names no place for a browser's user data");
    }
    let Some(chosen) = dirs
        .iter()
        .filter_map(|dir| open(dir))
        .max_by_key(|open| open.since)
    else {
        return Ok(Err(Closed::NotEnabled(not_enabled(&dirs))));
    };
    Ok(match Cdp::connect(&chosen.url, CONSENT_WAIT).await {
        Ok(cdp) => Ok(cdp),
        Err(Refusal::Unreachable) => Err(Closed::NotEnabled(not_enabled(&dirs))),
        Err(Refusal::Declined) => Err(Closed::Declined),
    })
}

/// The agent's tabs on `cdp`: `before` where they are still open, else a new window of its own.
pub(super) async fn agent_tabs(cdp: &Cdp, before: &[String]) -> anyhow::Result<Vec<String>> {
    let all = cdp.call("Target.getTargets", json!({}), None).await?;
    let open: Vec<String> = before
        .iter()
        .filter(|t| {
            all["targetInfos"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|i| i["targetId"].as_str() == Some(t.as_str()))
        })
        .cloned()
        .collect();
    if !open.is_empty() {
        return Ok(open);
    }
    let created = cdp
        .call(
            "Target.createTarget",
            json!({ "url": "about:blank", "newWindow": true }),
            None,
        )
        .await
        .map_err(|e| anyhow::anyhow!("opening the agent's window: {e}"))?;
    Ok(vec![
        created["targetId"]
            .as_str()
            .context("the browser opened no tab")?
            .to_string(),
    ])
}

/// The browser endpoint from `DevToolsActivePort`: its port, then its path.
fn devtools_url(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("DevToolsActivePort")).ok()?;
    let mut lines = text.lines();
    let port: u16 = lines.next()?.trim().parse().ok()?;
    let path = lines.next()?.trim();
    path.starts_with('/')
        .then(|| format!("ws://127.0.0.1:{port}{path}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_browsers_read_back_by_name_and_have_a_place_here() {
        for browser in UserBrowser::NAMED {
            let name = browser.name();
            assert_eq!(UserBrowser::named(name), Some(browser.clone()), "{name}");
        }
        assert_eq!(UserBrowser::named("netscape"), None);
        assert!(UserBrowser::Chrome.user_data_dir().is_some());
        let dir = PathBuf::from("/somewhere");
        assert_eq!(UserBrowser::Custom(dir.clone()).name(), "custom");
        assert_eq!(UserBrowser::named("custom"), None);
        assert_eq!(UserBrowser::Custom(dir.clone()).user_data_dir(), Some(dir));
    }
}
