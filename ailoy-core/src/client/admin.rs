use std::{io, process::Command};

use futures::Stream;
use http_over_x::{Auth, Error, http::HeaderMap, segment};

use crate::{
    agent::{
        AgentListOutputItem, AgentPutArgs, AgentState, SessionCreateArgs, SessionListOutputItem,
    },
    message::{Message, MessageDeltaOutput},
    tool::ToolDesc,
};

/// What the code that manages a server talks to it with: the tools it provides, the agents it
/// keeps, and their sessions, each with the sandbox the server makes for it.
pub struct Client {
    inner: http_over_x::Client,
}

impl Client {
    /// Serve `service` in this process, and talk to it over an in-memory pipe: `service` is
    /// the server's admin router. No credentials are sent, and the service is dropped once
    /// the client is. Must be called inside a tokio runtime.
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

    /// Every tool the server provides, as it describes each to a model. A spec names the ones
    /// its agent uses.
    pub async fn list_tool(&self) -> Result<Vec<ToolDesc>, Error> {
        self.inner.get("/tools").await
    }

    /// Every agent the server keeps.
    pub async fn list_agent(&self) -> Result<Vec<AgentListOutputItem>, Error> {
        self.inner.get("/agents/").await
    }

    /// Keep the agent `args` describes under `name`, in place of one already there, whose
    /// sessions stay. Refused when the server lacks a tool the spec names.
    pub async fn put_agent(
        &self,
        name: impl AsRef<str>,
        args: &AgentPutArgs,
    ) -> Result<AgentListOutputItem, Error> {
        let path = format!("/agents/{}", segment(name.as_ref()));
        self.inner.put(&path, args).await
    }

    /// The agent `name`.
    pub async fn get_agent(&self, name: impl AsRef<str>) -> Result<AgentListOutputItem, Error> {
        let path = format!("/agents/{}", segment(name.as_ref()));
        self.inner.get(&path).await
    }

    /// Keep the agent `name` no longer, nor its sessions and their sandboxes.
    pub async fn remove_agent(&self, name: impl AsRef<str>) -> Result<(), Error> {
        let path = format!("/agents/{}", segment(name.as_ref()));
        self.inner.delete(&path).await
    }

    /// Start a session of the agent `agent`, as `args` describes it, its sandbox made on the
    /// server's virtx. The answer is its state, whose id, made by the server, the session is
    /// known by. Refused when the spec uses a sandbox and `args` makes none, or the other way
    /// round; and as virtx refuses making it.
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

    /// Keep the session `id` of the agent `agent` no longer, nor its sandbox.
    pub async fn remove_session(
        &self,
        agent: impl AsRef<str>,
        id: impl AsRef<str>,
    ) -> Result<(), Error> {
        let (agent, id) = (segment(agent.as_ref()), segment(id.as_ref()));
        let path = format!("/agents/{agent}/sessions/{id}");
        self.inner.delete(&path).await
    }

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
        super::run_stream(&self.inner, format!("/agents/{agent}/sessions/{id}"), query)
    }
}

/// A client of the server `inner` reaches, however it was made.
impl From<http_over_x::Client> for Client {
    fn from(inner: http_over_x::Client) -> Self {
        Client { inner }
    }
}
