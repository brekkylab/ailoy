use std::{io, process::Command};

use futures::{Stream, StreamExt as _};
use http_over_x::{Auth, Error, Members, full, http, http::HeaderMap, refused, segment};
use sse_stream::SseStream;

use crate::{
    agent::{AgentState, SessionCreateArgs, SessionListOutputItem},
    message::{Message, MessageDeltaOutput},
};

/// What the code that runs an agent talks to its server with: the sessions of an agent the
/// server keeps, by the agent's name.
pub struct Client {
    inner: http_over_x::Client,
}

impl Client {
    /// Serve `service` in this process, and talk to it over an in-memory pipe: `service` is
    /// the server's router. No credentials are sent, and the service is dropped once the
    /// client is. Must be called inside a tokio runtime.
    #[cfg(feature = "embed")]
    pub fn embed<S, B>(service: S) -> Client
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
        let inner = http_over_x::Client::embed(service);
        Client { inner }
    }

    /// Start `command` as the server, and talk to it over its stdin and stdout. Must be called
    /// inside a tokio runtime.
    pub fn spawn(command: Command) -> io::Result<Client> {
        let inner = http_over_x::Client::spawn(command)?;
        Ok(Client { inner })
    }

    /// Reach the server at `url`, its base `http` or `https` URL, sending `auth`, the
    /// credentials it wants, and `headers`, whatever else it asks of each request, with every
    /// request. Refused when a token or a password would go in the clear: over `http` to any
    /// host but this machine.
    pub fn connect(url: &str, auth: Auth, headers: HeaderMap) -> Result<Client, Error> {
        let inner = http_over_x::Client::connect(url, auth, headers)?;
        Ok(Client { inner })
    }

    /// Start a session of the agent `agent`, as `args` describes it. The answer is its state,
    /// whose id, made by the server, the session is known by.
    pub async fn create_session(
        &self,
        agent: impl AsRef<str>,
        args: &SessionCreateArgs,
    ) -> Result<AgentState, Error> {
        let path = format!("/agents/{}/sessions/", segment(agent.as_ref()));
        self.inner.post(&path, args).await
    }

    /// Every session of the agent `agent`.
    pub async fn list_session(
        &self,
        agent: impl AsRef<str>,
    ) -> Result<Vec<SessionListOutputItem>, Error> {
        let path = format!("/agents/{}/sessions/", segment(agent.as_ref()));
        self.inner.get(&path).await
    }

    /// The state of the session `id` of the agent `agent`.
    pub async fn get_session(
        &self,
        agent: impl AsRef<str>,
        id: impl AsRef<str>,
    ) -> Result<AgentState, Error> {
        let (agent, id) = (segment(agent.as_ref()), segment(id.as_ref()));
        let path = format!("/agents/{agent}/sessions/{id}");
        self.inner.get(&path).await
    }

    /// Keep the session `id` of the agent `agent` no longer.
    pub async fn remove_session(
        &self,
        agent: impl AsRef<str>,
        id: impl AsRef<str>,
    ) -> Result<(), Error> {
        let (agent, id) = (segment(agent.as_ref()), segment(id.as_ref()));
        let path = format!("/agents/{agent}/sessions/{id}");
        self.inner.delete(&path).await
    }
}

impl Client {
    /// Run one turn of the session `id` of the agent `agent` on `query`, and stream what it
    /// produces: the model's output as it is generated, and each tool's result whole, one delta
    /// each. The server keeps the session's history once the turn ends, read or not. Refused
    /// while the session has a turn running.
    pub fn run_stream<'a>(
        &'a self,
        agent: impl AsRef<str>,
        id: impl AsRef<str>,
        query: &Message,
    ) -> impl Stream<Item = Result<MessageDeltaOutput, Error>> + Send + 'a {
        let (agent, id) = (segment(agent.as_ref()), segment(id.as_ref()));
        let path = format!("/agents/{agent}/sessions/{id}");
        let body = serde_json::to_vec(query);
        async_stream::try_stream! {
            let body = body.map_err(|e| Error::Client(e.to_string()))?;
            let request = http::Request::post(&path)
                .header(http::header::ACCEPT, "text/event-stream")
                .header(http::header::CONTENT_TYPE, "application/json")
                .body(full(body))
                .map_err(|e| Error::Client(format!("not a request: {e}")))?;
            let response = self.inner.send(request).await?;
            let response = match response.status().is_success() {
                true => response,
                false => Err(refused(response).await)?,
            };
            let mut events = SseStream::new(response.into_body());
            while let Some(event) = events.next().await {
                let event = event.map_err(|e| Error::Transport(format!("reading {path}: {e}")))?;
                let Some(data) = event.data else {
                    continue;
                };
                if event.event.as_deref() == Some("error") {
                    Err(failed(&data))?;
                }
                yield serde_json::from_str(&data).map_err(|e| Error::Client(e.to_string()))?;
            }
        }
    }
}

/// The error a turn ended on partway, from its `error` event's data, the
/// `{"error": {"code", "message"}}` a refusal answers with.
fn failed(data: &str) -> Error {
    let mut error: Members = serde_json::from_str::<serde_json::Value>(data)
        .ok()
        .and_then(|value| value.get("error")?.as_object().cloned())
        .unwrap_or_default();
    let mut field = |name| match error.remove(name) {
        Some(serde_json::Value::String(value)) => value,
        _ => String::new(),
    };
    let (code, message) = (field("code"), field("message"));
    Error::Status(500, format!("{code}: {message}"), error)
}

/// A client of the server `inner` reaches, however it was made.
impl From<http_over_x::Client> for Client {
    fn from(inner: http_over_x::Client) -> Self {
        Client { inner }
    }
}
