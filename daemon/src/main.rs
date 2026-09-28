//! `ailoy-daemon`: the always-on side of ailoy automations.
//!
//! It owns events (everything that arrived, by type), triggers (which event types
//! wake which automation, and the sources a trigger runs for itself) and runs. The
//! automation directories it points at stay wherever their authors keep them and are
//! read only when a run or a trigger script needs them.

mod api;
mod cli;
mod config;
mod console;
mod db;
mod dispatcher;
mod hooks;
mod sources;
mod state;
mod trigger;
mod worker;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use std::{future::IntoFuture, os::unix::fs::PermissionsExt};

use anyhow::Context as _;
use clap::{Parser, Subcommand};

use crate::{config::Config, db::Db, state::AppState};

#[derive(Parser, Debug)]
#[command(
    name = "ailoy-daemon",
    about = "Runs ailoy automations when events arrive"
)]
struct Cli {
    /// The daemon's administration socket, for the client commands. Relative paths
    /// are from the current directory, so the default works from the daemon root.
    #[arg(long, global = true, env = "AILOY_DAEMON_SOCKET", default_value = config::SOCKET_FILE)]
    socket: PathBuf,

    /// Publish events over TCP instead of the socket, to this address. Only
    /// `events publish` is served there.
    #[arg(long, global = true, env = "AILOY_DAEMON_URL")]
    url: Option<String>,

    /// API token for the TCP address, when the daemon asks for one.
    #[arg(long, global = true, env = "AILOY_DAEMON_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the daemon: sources, dispatcher, worker and the HTTP API.
    Serve {
        #[arg(long)]
        root: PathBuf,
    },
    /// One pass: dispatch dirty triggers, execute pending runs, exit. No source runs
    /// and nothing listens for posted events.
    Once {
        #[arg(long)]
        root: PathBuf,
    },
    #[command(subcommand)]
    Events(cli::Events),
    #[command(subcommand)]
    Triggers(cli::Triggers),
    #[command(subcommand)]
    Runs(cli::Runs),
    /// Credentials for posting events of one type.
    #[command(subcommand)]
    Tokens(cli::Tokens),
    /// Print the JSON Schemas of the registration bodies.
    Schema,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let cli = Cli::parse();
    let client = client(&cli);
    match cli.command {
        Command::Serve { root } => serve(root).await,
        Command::Once { root } => once(root).await,
        Command::Events(cmd) => print(client.events(cmd).await?),
        Command::Triggers(cmd) => print(client.triggers(cmd).await?),
        Command::Runs(cmd) => print(client.runs(cmd).await?),
        Command::Tokens(cmd) => print(client.tokens(cmd).await?),
        Command::Schema => print(trigger::TriggerConfig::schema()),
    }
}

/// `--url` picks the TCP address, otherwise the socket.
fn client(cli: &Cli) -> cli::Client {
    let target = match &cli.url {
        Some(url) => cli::Target::Tcp(url.clone()),
        None => cli::Target::Socket(cli.socket.clone()),
    };
    cli::Client::new(target, cli.token.clone())
}

fn print(v: serde_json::Value) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

async fn open(root: PathBuf) -> anyhow::Result<Arc<AppState>> {
    let root = std::path::absolute(root)?;
    std::fs::create_dir_all(&root)?;
    load_env(&root)?;
    let config = Config::load(&root)?;
    let db = Db::open(&root.join(config::DB_FILE))?;
    let state = AppState::new(root, config, db);
    worker::recover(&state).await?;
    // Every registered automation is looked at once, so a directory that vanished
    // while the daemon was down shows up in `last_error` right away.
    for t in state.db.list_triggers()? {
        if let Err(e) = ailoy::automation::AutomationDef::load(&t.config.automation) {
            state.db.trigger_failed(&t.name, &e.to_string())?;
        }
    }
    Ok(state)
}

async fn serve(root: PathBuf) -> anyhow::Result<()> {
    let state = open(root).await?;
    {
        let mut tasks = state.sources.lock().await;
        for t in state.db.list_triggers()? {
            tasks.start(&t.name, &t.config);
        }
    }
    tokio::spawn(dispatcher::run_loop(state.clone()));
    tokio::spawn(worker::run_loop(state.clone()));

    let socket_path = socket_path(&state);
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Before the socket, so a taken port does not leave this root without the name its
    // daemon is reached by.
    let tcp = tokio::net::TcpListener::bind(&state.config.listen)
        .await
        .with_context(|| format!("listening on {}", state.config.listen))?;

    // A socket file outlives the process that made it. One nothing answers on is that
    // leftover and is replaced; one that answers belongs to a daemon already serving
    // this root, and taking its name away would leave it running and unreachable.
    if tokio::net::UnixStream::connect(&socket_path).await.is_ok() {
        anyhow::bail!("a daemon is already serving {}", socket_path.display());
    }
    let _ = std::fs::remove_file(&socket_path);
    let socket = tokio::net::UnixListener::bind(&socket_path)?;
    std::fs::set_permissions(&socket_path, PermissionsExt::from_mode(0o600))?;
    tracing::info!(
        listen = %state.config.listen,
        socket = %socket_path.display(),
        root = %state.root.display(),
        "ailoy-daemon up"
    );

    let shutdown = || async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let on_socket =
        axum::serve(socket, api::socket_router(state.clone())).with_graceful_shutdown(shutdown());
    let on_tcp =
        axum::serve(tcp, api::tcp_router(state.clone())).with_graceful_shutdown(shutdown());
    let result = tokio::try_join!(on_socket.into_future(), on_tcp.into_future()).map(|_| ());
    let _ = std::fs::remove_file(&socket_path);
    Ok(result?)
}

/// Put `<root>/.env` into this process's environment, leaving whatever is already
/// there. What a run needs is read from the environment — a model's API key, the
/// variables `console.json` names its secrets by — and the daemon is the process that
/// reads them.
fn load_env(root: &Path) -> anyhow::Result<()> {
    let path = root.join(config::ENV_FILE);
    match dotenvy::from_path(&path) {
        Ok(()) => {
            tracing::info!(file = %path.display(), "environment loaded");
            Ok(())
        }
        Err(dotenvy::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(anyhow::anyhow!("{}: {e}", path.display())),
    }
}

/// The administration socket, as `daemon.toml` places it.
fn socket_path(state: &AppState) -> PathBuf {
    let configured = PathBuf::from(&state.config.socket);
    if configured.is_absolute() {
        configured
    } else {
        state.root.join(configured)
    }
}

async fn once(root: PathBuf) -> anyhow::Result<()> {
    let state = open(root).await?;
    let dispatched = dispatcher::pass(&state).await?;
    tracing::info!(dispatched, "dispatch pass done");
    loop {
        worker::claim_all(&state).await?;
        if state.db.count_active_runs()? == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    Ok(())
}
