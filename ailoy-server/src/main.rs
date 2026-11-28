//! `ailoy-server serve --config <file>`: an Ailoy server over TCP, serving the built-in provider
//! on two listeners: the agents it keeps and their sessions to the code that manages it
//! (`ailoy_core::Client`), which sends `AILOY_ADMIN_TOKEN`; and the sessions an app's users
//! start to them (`ailoy_core::AppClient`), each reached with its grant. What it keeps outlives it: the agents and sessions in its store, and each session's
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
//!   database, is a binary of its own over `ailoy_runner`'s routers with an `AgentStore` of its
//!   own.
//! - `tls` serves both listeners over TLS; without it, each must be on this machine's loopback,
//!   as a client sends a token in the clear to no other host.
//! - `virtx` is the server sessions' sandboxes are made on: by the admin's sessions as `auth`
//!   and `project` let in, by an app's with the user's own credentials. Without it, a session
//!   of an agent that uses a sandbox is refused.
//! - `models` runs an org's models on the backend it names, in place of the default's.

use std::{
    collections::BTreeMap,
    future::Future,
    io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use ailoy_core::agent::AgentProvider;
use ailoy_runner::{admin_router, agent::builtin_provider, model::Backend, router};
use anyhow::{Context as _, bail};
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use clap::Parser;
use http_over_x::Tls;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpListener;

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

/// The server, its listeners bound, until it [`run`](Self::run)s.
struct Server {
    provider: Arc<RwLock<AgentProvider>>,
    app: TcpListener,
    admin: TcpListener,
    admin_token: Arc<str>,
    tls: Option<Tls>,
}

impl Server {
    /// Bind the listeners `config` names, the admin's let in with `admin_token` alone, serving
    /// `provider`, its agents and sessions kept in its own store, with `config`'s virtx and
    /// models set on it. Refused for an empty token, for a listener off this machine's loopback
    /// without TLS, and where virtx is not reached.
    async fn bind(
        config: Config,
        admin_token: String,
        mut provider: AgentProvider,
    ) -> anyhow::Result<Server> {
        if admin_token.is_empty() {
            bail!("the admin token is empty");
        }
        if config.tls.is_none() {
            for addr in [config.listen, config.admin_listen] {
                if !addr.ip().is_loopback() {
                    bail!("{addr} is not on this machine's loopback: serve it with `tls`");
                }
            }
        }
        let tls = (config.tls).map(|tls| Tls::new(tls.cert, tls.key));

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

        let bind = |addr: SocketAddr| async move {
            TcpListener::bind(addr)
                .await
                .with_context(|| format!("listening on {addr}"))
        };
        Ok(Server {
            provider,
            app: bind(config.listen).await?,
            admin: bind(config.admin_listen).await?,
            admin_token: admin_token.into(),
            tls,
        })
    }

    /// Where an app's users reach it.
    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.app.local_addr()
    }

    /// Where the code that manages it reaches it.
    fn admin_addr(&self) -> io::Result<SocketAddr> {
        self.admin.local_addr()
    }

    /// Serve both listeners until `shutdown` completes. What is kept stays, for the server
    /// started again to go on with. An `Err` is a listener's, or of `tls`, read as serving
    /// starts.
    async fn run(self, shutdown: impl Future<Output = ()>) -> anyhow::Result<()> {
        let app = router(self.provider.clone());
        let admin = admin_router(self.provider)
            .layer(middleware::from_fn_with_state(self.admin_token, authorize));
        let served = async {
            tokio::try_join!(
                serve(app, self.app, self.tls.as_ref()),
                serve(admin, self.admin, self.tls.as_ref()),
            )
        };
        tokio::select! {
            served = served => served.map(|_| ()).map_err(Into::into),
            () = shutdown => Ok(()),
        }
    }
}

async fn serve(router: Router, listener: TcpListener, tls: Option<&Tls>) -> io::Result<()> {
    match tls {
        Some(tls) => http_over_x::serve_tls(router, listener, tls).await,
        None => http_over_x::serve_tcp(router, listener).await,
    }
}

/// Let the request through where it sends `token` as its bearer token; `401` otherwise.
async fn authorize(State(token): State<Arc<str>>, request: Request, next: Next) -> Response {
    let sent = (request.headers().get(header::AUTHORIZATION))
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if sent.is_some_and(|sent| same(sent.as_bytes(), token.as_bytes())) {
        return next.run(request).await;
    }
    let why = "the admin token is not sent";
    let body = json!({"error": {"code": "unauthorized", "message": why}});
    let challenge = [(header::WWW_AUTHENTICATE, "Bearer")];
    (StatusCode::UNAUTHORIZED, challenge, Json(body)).into_response()
}

/// Whether `sent` is `token`, in a time that does not tell how much of it matched.
fn same(sent: &[u8], token: &[u8]) -> bool {
    let differ = sent
        .iter()
        .zip(token)
        .fold(0, |differ, (a, b)| differ | (a ^ b));
    sent.len() == token.len() && differ == 0
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
    let home = match &config.home {
        Some(home) => home.clone(),
        None => dirs::data_local_dir()
            .context("this user has no local data directory: give `home`")?
            .join("ailoy"),
    };
    let mut provider = builtin_provider();
    provider.store = Arc::new(FileStore::new(home));
    let server = Server::bind(config, token, provider).await?;
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

#[cfg(test)]
mod tests {
    //! The server over TCP, reached by both clients as a program elsewhere reaches it.

    use ailoy_core::{
        AppClient, Client,
        agent::{
            AgentPutArgs, AgentSpec, AgentState, AppSessionCreateArgs, AppSessionCreateOutput,
            SessionCreateArgs,
        },
        message::{FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, Role},
        model::{InferLangModel, LangModelOptions},
        tool::ToolDesc,
    };
    use futures::{
        StreamExt as _,
        future::BoxFuture,
        stream::{self, BoxStream},
    };
    use http_over_x::{Auth, http::HeaderMap};
    use tokio::sync::oneshot;

    use super::*;

    const TOKEN: &str = "admin-secret";

    /// Answers `pong` whatever it is asked.
    struct Parrot;

    impl InferLangModel for Parrot {
        fn infer(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
            unimplemented!("a turn streams")
        }

        fn infer_stream(
            &self,
            _: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
            let output = MessageOutput {
                message: Message::new(Role::Assistant).with_contents([Part::text("pong")]),
                finish_reason: FinishReason::Stop {},
                usage: None,
            };
            Box::pin(stream::once(async move { Ok(output.into()) }))
        }
    }

    fn ping() -> Message {
        Message::new(Role::User).with_contents([Part::text("ping")])
    }

    /// A server on this machine's loopback, with a parrot model, keeping its agents and sessions
    /// under `home`, served until the sender is dropped or sent; and the base URLs of its app's
    /// and its admin's listeners.
    async fn served(home: &Path) -> (oneshot::Sender<()>, String, String) {
        let mut provider = builtin_provider();
        provider.models.set(
            "tcp-parrot",
            |_: &str, _: Option<&str>| -> anyhow::Result<Arc<dyn InferLangModel>> {
                Ok(Arc::new(Parrot))
            },
        );
        provider.store = Arc::new(FileStore::new(home));
        let config: Config = serde_json::from_value(loopback()).unwrap();
        let server = Server::bind(config, TOKEN.to_owned(), provider)
            .await
            .unwrap();
        let url = |addr: SocketAddr| format!("http://{addr}");
        let (app, admin) = (
            url(server.local_addr().unwrap()),
            url(server.admin_addr().unwrap()),
        );
        let (stop, stopped) = oneshot::channel();
        tokio::spawn(server.run(async move {
            let _ = stopped.await;
        }));
        (stop, app, admin)
    }

    fn loopback() -> serde_json::Value {
        serde_json::json!({"listen": "127.0.0.1:0", "admin_listen": "127.0.0.1:0"})
    }

    /// An agent, a session's history and an app's grant outlive the server that kept them.
    #[tokio::test]
    async fn test_what_a_server_keeps_outlives_it() {
        let home = tempfile::tempdir().unwrap();
        let (stop, app, admin) = served(home.path()).await;
        let client = Client::connect(&admin, Auth::bearer(TOKEN), HeaderMap::new()).unwrap();
        let spec = AgentSpec::new("tcp-parrot/x");
        client
            .put_agent("c", &AgentPutArgs::new(spec))
            .await
            .unwrap();
        let session = client
            .create_session("c", &SessionCreateArgs::new())
            .await
            .unwrap();
        let deltas: Vec<_> = client.run_stream("c", &session.id, &ping()).collect().await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");
        let raw = http_over_x::Client::connect(&app, Auth::None, HeaderMap::new()).unwrap();
        let made: AppSessionCreateOutput = raw
            .post("/sessions/", &AppSessionCreateArgs::new("c"))
            .await
            .unwrap();
        stop.send(()).unwrap();

        let (_stop, app, admin) = served(home.path()).await;
        let client = Client::connect(&admin, Auth::bearer(TOKEN), HeaderMap::new()).unwrap();
        assert_eq!(client.list_agent().await.unwrap()[0].name, "c");
        let deltas: Vec<_> = client.run_stream("c", &session.id, &ping()).collect().await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");
        let history = client.get_session("c", &session.id).await.unwrap().history;
        assert_eq!(history.len(), 4);

        let raw =
            http_over_x::Client::connect(&app, Auth::bearer(made.grant), HeaderMap::new()).unwrap();
        let path = format!("/sessions/{}", made.session.id);
        let state: AgentState = raw.get(&path).await.unwrap();
        assert_eq!(state.id, made.session.id);
    }

    #[tokio::test]
    async fn test_the_admin_runs_a_session_s_turn_with_the_admin_token() {
        let home = tempfile::tempdir().unwrap();
        let (_stop, _, admin) = served(home.path()).await;
        let client = Client::connect(&admin, Auth::bearer(TOKEN), HeaderMap::new()).unwrap();

        let spec = AgentSpec::new("tcp-parrot/x");
        client
            .put_agent("a", &AgentPutArgs::new(spec))
            .await
            .unwrap();
        let session = client
            .create_session("a", &SessionCreateArgs::new())
            .await
            .unwrap();
        let deltas: Vec<_> = client.run_stream("a", &session.id, &ping()).collect().await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");

        let history = client.get_session("a", &session.id).await.unwrap().history;
        assert_eq!(history.last().unwrap().contents[0].as_text(), Some("pong"));
    }

    #[tokio::test]
    async fn test_the_admin_listener_refuses_any_other_token() {
        let home = tempfile::tempdir().unwrap();
        let (_stop, _, admin) = served(home.path()).await;
        for auth in [
            Auth::None,
            Auth::bearer("admin-secreT"),
            Auth::bearer("admin"),
        ] {
            let client = Client::connect(&admin, auth, HeaderMap::new()).unwrap();
            let error = client.list_tool().await.err().unwrap();
            assert_eq!(error.status(), Some(401));
        }
    }

    /// An app's user starts a session with no admin token, and reaches only it, on its own
    /// listener, which serves none of the admin's paths.
    #[tokio::test]
    async fn test_an_app_runs_its_session_on_its_own_listener() {
        let home = tempfile::tempdir().unwrap();
        let (_stop, app, admin) = served(home.path()).await;
        let client = Client::connect(&admin, Auth::bearer(TOKEN), HeaderMap::new()).unwrap();
        let spec = AgentSpec::new("tcp-parrot/x");
        client
            .put_agent("b", &AgentPutArgs::new(spec))
            .await
            .unwrap();

        let args = AppSessionCreateArgs::new("b");
        let session = AppClient::connect(&app, Auth::None, HeaderMap::new(), &args)
            .await
            .unwrap();
        let deltas: Vec<_> = session.run_stream(&ping()).collect().await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");
        assert_eq!(session.get_session().await.unwrap().history.len(), 2);

        let on_app = Client::connect(&app, Auth::bearer(TOKEN), HeaderMap::new()).unwrap();
        assert_eq!(on_app.list_agent().await.err().unwrap().status(), Some(404));
        session.close().await.unwrap();
    }

    #[tokio::test]
    async fn test_a_listener_off_the_loopback_needs_tls() {
        let config = serde_json::json!({"listen": "0.0.0.0:0", "admin_listen": "127.0.0.1:0"});
        let config: Config = serde_json::from_value(config).unwrap();
        let error = (Server::bind(config, TOKEN.to_owned(), builtin_provider()).await)
            .err()
            .unwrap();
        assert!(error.to_string().contains("tls"), "{error}");
    }

    #[tokio::test]
    async fn test_an_empty_admin_token_is_refused() {
        let config: Config = serde_json::from_value(loopback()).unwrap();
        let bound = Server::bind(config, String::new(), builtin_provider()).await;
        assert!(bound.is_err());
    }
}
