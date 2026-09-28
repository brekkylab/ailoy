//! `daemon.toml`: what one daemon instance is, apart from what is registered with it.

use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

pub const CONFIG_FILE: &str = "daemon.toml";
/// Secrets the daemon's own process needs, such as the API key a model is reached
/// with. Read at startup; a variable already in the environment wins.
pub const ENV_FILE: &str = ".env";
/// The token a request to the TCP address must bear. Set it and the address asks for
/// a credential; leave it unset and the address is open to whoever reaches it, which
/// is why `listen` defaults to loopback.
pub const TOKEN_ENV: &str = "AILOY_API_TOKEN";
/// Default name of the administration socket, inside the root.
pub const SOCKET_FILE: &str = "daemon.sock";
pub const DB_FILE: &str = "daemon.db";
/// Runner work directories, one per trigger.
pub const WORK_DIR: &str = "work";
/// Scratch of the shared trigger console, one subdirectory per trigger.
pub const TRIGGER_DIR: &str = "trigger";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// TCP address carrying `POST /events/{type}` alone. Reachable from wherever an
    /// event sender is.
    pub listen: String,
    /// Socket file carrying every route, relative to the daemon root unless absolute.
    /// Who may administer the daemon is who may open this file.
    pub socket: String,
    /// Events older than this are pruned.
    pub events_retention_secs: u64,
    pub console: ConsoleConfig,
    pub hooks: Vec<HookConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:7878".into(),
            socket: SOCKET_FILE.into(),
            events_retention_secs: 7 * 24 * 3600,
            console: ConsoleConfig::default(),
            hooks: Vec::new(),
        }
    }
}

/// The console server both the trigger scripts and the runs use.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsoleConfig {
    /// argv of the console server.
    pub program: Vec<String>,
    /// The server is host-local (`cortex-local-console`): images, network reach and
    /// secrets cannot be enforced. Development only.
    pub host: bool,
    /// Image of the shared trigger console.
    pub image: String,
}

impl Default for ConsoleConfig {
    fn default() -> Self {
        Self {
            program: vec!["cortex-uvm-console".into()],
            host: false,
            image: "python:3.12-slim".into(),
        }
    }
}

/// An outbound webhook fired on run status changes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HookConfig {
    pub url: String,
    /// Statuses to send; empty means all.
    #[serde(default)]
    pub statuses: Vec<String>,
    /// Triggers to send for; empty means all.
    #[serde(default)]
    pub triggers: Vec<String>,
}

impl Config {
    /// `daemon.toml` under `root`, or the defaults when there is none.
    pub fn load(root: &Path) -> anyhow::Result<Self> {
        let path = root.join(CONFIG_FILE);
        match fs::read_to_string(&path) {
            Ok(text) => {
                toml::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(anyhow::anyhow!("{}: {e}", path.display())),
        }
    }
}
