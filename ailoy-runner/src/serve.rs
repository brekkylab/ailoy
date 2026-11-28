//! [`Server`]: an [`AgentProvider`] served over TCP to a program elsewhere, [`router`] to an
//! app's users on one listener and [`admin_router`] to the code that manages it on another, let
//! in with the admin token alone. What the provider keeps is its store's: a server keeps it in
//! files, or a database, by the [`AgentStore`](ailoy_core::agent::AgentStore) it sets.

use std::{
    future::Future,
    io,
    net::SocketAddr,
    sync::{Arc, RwLock},
};

use ailoy_core::agent::AgentProvider;
use anyhow::{Context as _, bail};
use axum::{
    Json, Router,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
pub use http_over_x::Tls;
use serde_json::json;
use tokio::net::TcpListener;

use crate::{admin_router, router};

/// A provider served over TCP, its listeners bound, until it [`run`](Self::run)s.
pub struct Server {
    provider: Arc<RwLock<AgentProvider>>,
    app: TcpListener,
    admin: TcpListener,
    admin_token: Arc<str>,
    tls: Option<Tls>,
}

impl Server {
    /// Bind `listen`, where an app's users reach the sessions they start, and `admin_listen`,
    /// where the code that manages `provider` reaches it with `admin_token` alone; both over
    /// `tls` where it is given. Refused for an empty token, and for a listener off this
    /// machine's loopback without TLS, as a client sends a token in the clear to no other host.
    pub async fn bind(
        provider: Arc<RwLock<AgentProvider>>,
        listen: SocketAddr,
        admin_listen: SocketAddr,
        admin_token: String,
        tls: Option<Tls>,
    ) -> anyhow::Result<Server> {
        if admin_token.is_empty() {
            bail!("the admin token is empty");
        }
        if tls.is_none() {
            for addr in [listen, admin_listen] {
                if !addr.ip().is_loopback() {
                    bail!("{addr} is not on this machine's loopback: serve it over TLS");
                }
            }
        }
        let bind = |addr: SocketAddr| async move {
            TcpListener::bind(addr)
                .await
                .with_context(|| format!("listening on {addr}"))
        };
        Ok(Server {
            provider,
            app: bind(listen).await?,
            admin: bind(admin_listen).await?,
            admin_token: admin_token.into(),
            tls,
        })
    }

    /// Where an app's users reach it.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.app.local_addr()
    }

    /// Where the code that manages it reaches it.
    pub fn admin_addr(&self) -> io::Result<SocketAddr> {
        self.admin.local_addr()
    }

    /// Serve both listeners until `shutdown` completes. What the provider keeps stays, for a
    /// server started again on its store to go on with. An `Err` is a listener's, or of the
    /// TLS files, read as serving starts.
    pub async fn run(self, shutdown: impl Future<Output = ()>) -> anyhow::Result<()> {
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
    let differ = (sent.iter().zip(token)).fold(0, |differ, (a, b)| differ | (a ^ b));
    sent.len() == token.len() && differ == 0
}

#[cfg(test)]
mod tests {
    use ailoy_core::{
        AppClient, Auth, Client,
        agent::{AgentPutArgs, AgentSpec, AppSessionCreateArgs, SessionCreateArgs},
        message::{FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, Role},
        model::{InferLangModel, LangModelOptions},
        tool::ToolDesc,
    };
    use futures::{
        StreamExt as _,
        future::BoxFuture,
        stream::{self, BoxStream},
    };
    use http_over_x::http::HeaderMap;
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

    /// A provider whose `parrot` org's models are [`Parrot`]s.
    fn parrots() -> Arc<RwLock<AgentProvider>> {
        let mut provider = AgentProvider::new();
        provider.models.set(
            "parrot",
            |_: &str, _: Option<&str>| -> anyhow::Result<Arc<dyn InferLangModel>> {
                Ok(Arc::new(Parrot))
            },
        );
        Arc::new(RwLock::new(provider))
    }

    fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().unwrap()
    }

    /// `provider` served on this machine's loopback until the sender is dropped or sent; and
    /// the base URLs of its app's and its admin's listeners.
    async fn served(provider: Arc<RwLock<AgentProvider>>) -> (oneshot::Sender<()>, String, String) {
        let token = TOKEN.to_owned();
        let server = Server::bind(provider, loopback(), loopback(), token, None)
            .await
            .unwrap();
        let url = |addr: SocketAddr| format!("http://{addr}");
        let app = url(server.local_addr().unwrap());
        let admin = url(server.admin_addr().unwrap());
        let (stop, stopped) = oneshot::channel();
        tokio::spawn(server.run(async move {
            let _ = stopped.await;
        }));
        (stop, app, admin)
    }

    fn admin(url: &str) -> Client {
        Client::connect(url, Auth::bearer(TOKEN), HeaderMap::new()).unwrap()
    }

    #[tokio::test]
    async fn test_the_admin_runs_a_session_s_turn_with_the_admin_token() {
        let (_stop, _, url) = served(parrots()).await;
        let client = admin(&url);
        let spec = AgentSpec::new("parrot/x");
        client
            .put_agent("a", &AgentPutArgs::new(spec))
            .await
            .unwrap();
        let args = SessionCreateArgs::new();
        let session = client.create_session("a", &args).await.unwrap();
        let deltas: Vec<_> = client.run_stream("a", &session.id, &ping()).collect().await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");

        let history = client.get_session("a", &session.id).await.unwrap().history;
        assert_eq!(history.last().unwrap().contents[0].as_text(), Some("pong"));
    }

    #[tokio::test]
    async fn test_the_admin_listener_refuses_any_other_token() {
        let (_stop, _, url) = served(parrots()).await;
        for auth in [
            Auth::None,
            Auth::bearer("admin-secreT"),
            Auth::bearer("admin"),
        ] {
            let client = Client::connect(&url, auth, HeaderMap::new()).unwrap();
            let error = client.list_tool().await.err().unwrap();
            assert_eq!(error.status(), Some(401));
        }
    }

    /// An app's user starts a session with no admin token, and reaches only it, on its own
    /// listener, which serves none of the admin's paths.
    #[tokio::test]
    async fn test_an_app_runs_its_session_on_its_own_listener() {
        let (_stop, app, url) = served(parrots()).await;
        let spec = AgentSpec::new("parrot/x");
        admin(&url)
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

        let on_app = admin(&app);
        assert_eq!(on_app.list_agent().await.err().unwrap().status(), Some(404));
        session.close().await.unwrap();
    }

    /// What the provider keeps outlives a server of it, for the next to go on with.
    #[tokio::test]
    async fn test_what_is_kept_outlives_the_server() {
        let provider = parrots();
        let (stop, _, url) = served(provider.clone()).await;
        let client = admin(&url);
        let spec = AgentSpec::new("parrot/x");
        client
            .put_agent("c", &AgentPutArgs::new(spec))
            .await
            .unwrap();
        let args = SessionCreateArgs::new();
        let session = client.create_session("c", &args).await.unwrap();
        stop.send(()).unwrap();

        let (_stop, _, url) = served(provider).await;
        let deltas: Vec<_> = admin(&url)
            .run_stream("c", &session.id, &ping())
            .collect()
            .await;
        assert!(deltas.iter().all(Result::is_ok), "{deltas:?}");
    }

    #[tokio::test]
    async fn test_a_listener_off_the_loopback_needs_tls() {
        let anywhere = "0.0.0.0:0".parse().unwrap();
        let token = TOKEN.to_owned();
        let bound = Server::bind(parrots(), anywhere, loopback(), token, None).await;
        assert!(bound.err().unwrap().to_string().contains("TLS"));
    }

    #[tokio::test]
    async fn test_an_empty_admin_token_is_refused() {
        let bound = Server::bind(parrots(), loopback(), loopback(), String::new(), None).await;
        assert!(bound.is_err());
    }
}
