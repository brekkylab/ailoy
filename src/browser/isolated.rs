//! The agent's own browser, apart from the person's: [`Browser::isolated`](super::Browser::isolated).
//!
//! The browser named, else a Chromium-based one installed here ([`installed`]); started on first
//! use with a profile of its own and remote debugging on, and closed with the agent.

use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
};

use anyhow::Context as _;
use serde_json::json;

use super::cdp::Cdp;

/// A browser started for the agent, killed when dropped.
pub(super) struct Launched {
    child: Child,
    /// Its user data directory, where it writes `DevToolsActivePort`.
    pub dir: PathBuf,
}

impl Launched {
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for Launched {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Starts `executable` on the profile `dir`, with remote debugging on a port of its choosing.
pub(super) fn launch(executable: &Path, dir: &Path, headless: bool) -> anyhow::Result<Launched> {
    std::fs::create_dir_all(dir)?;
    // A file left by an earlier run would name a port the new one is not on.
    let _ = std::fs::remove_file(dir.join("DevToolsActivePort"));
    let mut command = Command::new(executable);
    command
        .args([
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
        ])
        .arg(format!("--user-data-dir={}", dir.display()));
    if headless {
        command.arg("--headless=new");
    }
    let child = command
        .arg("about:blank")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("starting the browser at {}", executable.display()))?;
    Ok(Launched {
        child,
        dir: dir.to_path_buf(),
    })
}

/// The tab a browser just started opened, which the agent takes as its first.
pub(super) async fn first_tab(cdp: &Cdp) -> anyhow::Result<Vec<String>> {
    let all = cdp.call("Target.getTargets", json!({}), None).await?;
    let first = all["targetInfos"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|info| info["type"] == "page")
        .and_then(|info| info["targetId"].as_str());
    if let Some(target) = first {
        return Ok(vec![target.to_string()]);
    }
    let created = cdp
        .call("Target.createTarget", json!({ "url": "about:blank" }), None)
        .await?;
    Ok(vec![
        created["targetId"]
            .as_str()
            .context("the browser opened no tab")?
            .to_string(),
    ])
}

/// A Chromium-based browser installed here: Chrome, Chromium, Edge or Brave, the first found.
pub(super) fn installed() -> Option<PathBuf> {
    let home = dirs::home_dir().unwrap_or_default();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if cfg!(target_os = "macos") {
        for app in [
            "Google Chrome.app/Contents/MacOS/Google Chrome",
            "Chromium.app/Contents/MacOS/Chromium",
            "Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "Brave Browser.app/Contents/MacOS/Brave Browser",
        ] {
            candidates.push(Path::new("/Applications").join(app));
            candidates.push(home.join("Applications").join(app));
        }
    } else if cfg!(target_os = "windows") {
        let roots: Vec<PathBuf> = ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"]
            .iter()
            .filter_map(|v| std::env::var_os(v).map(PathBuf::from))
            .collect();
        for app in [
            r"Google\Chrome\Application\chrome.exe",
            r"Chromium\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe",
            r"BraveSoftware\Brave-Browser\Application\brave.exe",
        ] {
            candidates.extend(roots.iter().map(|root| root.join(app)));
        }
    } else {
        // Linux and other Unix: by name on the PATH.
        let path: Vec<PathBuf> = std::env::var_os("PATH")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default();
        for name in [
            "google-chrome-stable",
            "google-chrome",
            "chromium",
            "chromium-browser",
            "microsoft-edge-stable",
            "microsoft-edge",
            "brave-browser",
        ] {
            candidates.extend(path.iter().map(|dir| dir.join(name)));
        }
    }
    candidates.into_iter().find(|c| c.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_installed_browser_is_a_file_there() {
        if let Some(found) = installed() {
            assert!(found.is_file(), "{found:?}");
        }
    }
}
