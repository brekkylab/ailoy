//! `ailoy-server serve --config <file>`: the built-in provider served as an
//! [`ailoy_runner::Server`], its admin let in with `AILOY_ADMIN_TOKEN`, keeping its agents and
//! sessions in files ([`FileStore`]). What it keeps outlives it: the agents and sessions in its store, and each session's
//! sandbox on virtx, reached again as the session's next turn begins.
//!
//! The config file:
//!
//! ```json
//! {
//!   "listen": "127.0.0.1:8480",
//!   "admin_listen": "127.0.0.1:8481",
//!   "home": "/var/lib/ailoy",
//!   "tls": { "cert": "cert.pem", "key": "key.pem" },
//!   "virtx": { "url": "https://virtx.example", "auth": { "type": "bearer", "token": "..." } },
//!   "models": { "anthropic": "api" }
//! }
//! ```
//!
//! - `home` is the directory the agents and sessions are kept in, as files ([`FileStore`]); the
//!   user's local data directory's `ailoy`, unless given. A server keeping them elsewhere, as in a
//!   database, is a binary of its own: an `ailoy_runner::Server` of a provider with an
//!   `AgentStore` of its own.
//! - `tls` serves both listeners over TLS; without it, each must be on this machine's loopback,
//!   as a client sends a token in the clear to no other host.
//! - `virtx` is the server sessions' sandboxes are made on: by the admin's sessions as `auth`
//!   and `project` let in, by an app's with the user's own credentials. Without it, a session
//!   of an agent that uses a sandbox is refused.
//! - `models` runs an org's models on the backend it names, in place of the default's.

use std::{
    collections::BTreeMap,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use ailoy_runner::{Server, Tls, agent::builtin_provider, model::Backend};
use anyhow::Context as _;
use clap::Parser;
use serde::Deserialize;

mod store;
use store::FileStore;

/// The config file, as the crate's docs show it.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    /// Where an app's users reach the sessions they start.
    listen: SocketAddr,
    /// Where the code that manages the server reaches the agents it keeps.
    admin_listen: SocketAddr,
    /// The directory the agents and sessions are kept in.
    #[serde(default)]
    home: Option<PathBuf>,
    /// The certificate and key both listeners are served over TLS with.
    #[serde(default)]
    tls: Option<TlsConfig>,
    /// The virtx server sessions' sandboxes are made on.
    #[serde(default)]
    virtx: Option<VirtxConfig>,
    /// The backend each org's models run on, in place of the default's.
    #[serde(default)]
    models: BTreeMap<String, Backend>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TlsConfig {
    /// The certificate chain, in PEM.
    cert: PathBuf,
    /// Its private key, in PEM.
    key: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VirtxConfig {
    /// Its base `http` or `https` URL.
    url: String,
    /// The credentials the admin's sessions' sandboxes are made with.
    #[serde(default)]
    auth: virtx::Auth,
    /// The project they are made in: the credentials' user's own without one.
    #[serde(default)]
    project: Option<String>,
}

impl Config {
    /// The config at `path`.
    fn read(path: &Path) -> anyhow::Result<Config> {
        let read =
            || -> anyhow::Result<Config> { Ok(serde_json::from_slice(&std::fs::read(path)?)?) };
        read().with_context(|| format!("reading {}", path.display()))
    }
}

#[derive(Parser)]
enum Command {
    /// Serve the agents and their sessions as `--config` says, until interrupted or terminated.
    Serve {
        #[arg(long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let Command::Serve { config } = Command::parse();
    let config = Config::read(&config)?;
    let token = std::env::var("AILOY_ADMIN_TOKEN")
        .context("AILOY_ADMIN_TOKEN is not set: the token the code managing this server sends")?;

    let mut provider = builtin_provider();
    let home = match config.home {
        Some(home) => home,
        None => dirs::data_local_dir()
            .context("this user has no local data directory: give `home`")?
            .join("ailoy"),
    };
    provider.store = Arc::new(FileStore::new(home));
    if let Some(virtx) = config.virtx {
        let url = virtx.url.clone();
        // Connecting blocks until the client is made.
        let client = tokio::task::spawn_blocking(move || {
            virtx::Client::connect(&virtx.url, virtx.auth, virtx.project.as_deref())
        })
        .await?
        .with_context(|| format!("reaching virtx at {url}"))?;
        provider.virtx = Some(client);
        provider.virtx_url = Some(url);
    }
    for (org, backend) in config.models {
        provider.models.set(org, backend);
    }

    let provider = Arc::new(RwLock::new(provider));
    let tls = (config.tls).map(|tls| Tls::new(tls.cert, tls.key));
    let server = Server::bind(provider, config.listen, config.admin_listen, token, tls).await?;
    eprintln!(
        "serving apps on {} and the admin on {}",
        server.local_addr()?,
        server.admin_addr()?
    );
    server.run(stopped()).await
}

/// Completes on Ctrl-C, or on `SIGTERM` where there is one.
async fn stopped() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminated = signal(SignalKind::terminate()).expect("a SIGTERM handler");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminated.recv() => {}
        }
    }
    #[cfg(not(unix))]
    let _ = tokio::signal::ctrl_c().await;
}
