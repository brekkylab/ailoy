//! The person's own Chrome, lent with their consent: [`Browser::host`](super::Browser::host).
//!
//! Chrome 144 and later lets a client in only once the person allows it in a prompt, and only
//! while remote debugging is on at `chrome://inspect/#remote-debugging`. This process holds that
//! one connection for the agent's life, so the prompt comes once.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Context as _;
use serde_json::json;

use super::cdp::{Cdp, Refusal};

/// How long the person has to answer Chrome's prompt.
const CONSENT_WAIT: Duration = Duration::from_secs(120);

pub(super) const DECLINED: &str = "The user declined to let you use their Chrome. Do not try the \
     browser again unless they ask; `ask_user` asks them again.";

pub(super) fn not_enabled(dir: &Path) -> String {
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

/// Why the agent cannot have Chrome now, as what it is told.
pub(super) enum Closed {
    Declined,
    NotEnabled(String),
}

/// Connects to the person's Chrome, which is when it asks them.
pub(super) async fn connect(
    user_data_dir: Option<&Path>,
) -> anyhow::Result<Result<Arc<Cdp>, Closed>> {
    let dir = user_data_dir
        .map(Path::to_path_buf)
        .or_else(default_user_data_dir)
        .context(
            "no Chrome user data directory on this OS: pass one to `Browser::user_data_dir`",
        )?;
    let Some(url) = devtools_url(&dir) else {
        return Ok(Err(Closed::NotEnabled(not_enabled(&dir))));
    };
    Ok(match Cdp::connect(&url, CONSENT_WAIT).await {
        Ok(cdp) => Ok(cdp),
        Err(Refusal::Unreachable) => Err(Closed::NotEnabled(not_enabled(&dir))),
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
            .context("Chrome opened no tab")?
            .to_string(),
    ])
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devtools_active_port_is_a_port_then_a_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("DevToolsActivePort"),
            "9222\n/devtools/browser/abc\n",
        )
        .unwrap();
        assert_eq!(
            devtools_url(dir.path()).as_deref(),
            Some("ws://127.0.0.1:9222/devtools/browser/abc")
        );
        std::fs::write(dir.path().join("DevToolsActivePort"), "x\n/y\n").unwrap();
        assert_eq!(devtools_url(dir.path()), None);
    }
}
