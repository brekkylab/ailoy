use futures::Stream;
use http_over_x::{
    Auth, Error,
    http::{HeaderMap, HeaderValue, header},
    segment,
};

use crate::{
    agent::{AgentState, AppSessionCreateArgs, AppSessionCreateOutput},
    message::{Message, MessageDeltaOutput},
};

/// One session an app's user starts of an agent a server keeps, and reaches with its grant
/// alone: the turns run in it, and its state, through this; and removed, with its sandbox, by
/// [`close`](Self::close), or on a task of its own as this drops. The grant reaches nothing else,
/// so neither does this.
pub struct AppClient {
    /// The client of the server, sending the grant; until the session is removed.
    inner: Option<http_over_x::Client>,
    /// The session's id.
    id: String,
}

impl AppClient {
    /// Start a session on the server at `url`, its base `http` or `https` URL, as `args` says,
    /// sending `auth`, the app's user's credentials, and `headers`, whatever else it asks of
    /// each request; and reach it with its grant from then on. Its sandbox is made of the
    /// package `args` names as `auth` lets in, on the server's virtx: refused as virtx refuses
    /// making it, `401` or `403` for one the package does not let in, `429` past its limits.
    /// Must be called inside a tokio runtime.
    pub async fn connect(
        url: &str,
        auth: Auth,
        headers: HeaderMap,
        args: &AppSessionCreateArgs,
    ) -> Result<AppClient, Error> {
        let inner = http_over_x::Client::connect(url, auth, headers)?;
        AppClient::over(inner, args).await
    }

    /// Serve `service` in this process, and start a session on it as `args` says, talking to
    /// it over an in-memory pipe: `service` is the server's router. No credentials are sent but
    /// the grant, so a sandbox is made of a package that lets anyone in alone. Must be called
    /// inside a tokio runtime.
    #[cfg(feature = "embed")]
    pub async fn embed<S, B>(service: S, args: &AppSessionCreateArgs) -> Result<AppClient, Error>
    where
        S: tower_service::Service<
                http_over_x::http::Request<hyper::body::Incoming>,
                Response = http_over_x::http::Response<B>,
            > + Clone
            + Send
            + 'static,
        S::Future: Send + 'static,
        S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
        B: http_body::Body<Data = bytes::Bytes> + Send + 'static,
        B::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        AppClient::over(http_over_x::Client::embed(service), args).await
    }

    /// Start a session as `args` says on the server `inner` reaches, however it was made, and
    /// reach it with its grant from then on, over either transport.
    pub async fn over(
        inner: http_over_x::Client,
        args: &AppSessionCreateArgs,
    ) -> Result<AppClient, Error> {
        let made: AppSessionCreateOutput = inner.post("/sessions/", args).await?;
        let grant = HeaderValue::from_str(&format!("Bearer {}", made.grant)).map_err(|e| {
            Error::Client(format!("the server gave a grant no header carries: {e}"))
        })?;
        // From here on, the grant alone: what started it reaches nothing more through this client.
        inner.set_header(header::AUTHORIZATION, Some(grant));
        Ok(AppClient {
            inner: Some(inner),
            id: made.session.id,
        })
    }

    /// The session's id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The session's state.
    pub async fn get_session(&self) -> Result<AgentState, Error> {
        self.inner().get(&self.path()).await
    }

    /// Run one turn on `query`, and stream what it produces: the model's output as it is
    /// generated, and each tool's result whole, one delta each. The server keeps the session's
    /// history once the turn ends, read or not. Refused while the session has a turn running.
    pub fn run_stream<'a>(
        &'a self,
        query: &Message,
    ) -> impl Stream<Item = Result<MessageDeltaOutput, Error>> + Send + 'a {
        super::run_stream(self.inner(), self.path(), query)
    }

    /// Remove the session, and with it its sandbox and its grant, answering once they are.
    pub async fn close(mut self) -> Result<(), Error> {
        let inner = self
            .inner
            .take()
            .expect("an app client holds its client until it drops");
        inner.delete(&self.path()).await
    }

    fn inner(&self) -> &http_over_x::Client {
        (self.inner.as_ref()).expect("an app client holds its client until it drops")
    }

    fn path(&self) -> String {
        format!("/sessions/{}", segment(&self.id))
    }
}

/// Removes the session, as [`AppClient::close`] does, on a task of its own: unawaited, and not
/// at all outside a tokio runtime.
impl Drop for AppClient {
    fn drop(&mut self) {
        let Some(inner) = self.inner.take() else {
            return;
        };
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let path = self.path();
        runtime.spawn(async move {
            let _ = inner.delete(&path).await;
        });
    }
}
