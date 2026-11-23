use std::{io, process::Command};

use http_over_x::{Auth, Error, http::HeaderMap, segment};

use crate::{
    agent::{AgentListOutputItem, AgentPutArgs},
    tool::ToolDesc,
};

/// What the code that manages a server talks to it with: the tools it provides and the agents
/// it keeps.
pub struct AdminClient {
    inner: http_over_x::Client,
}

impl AdminClient {
    /// Serve `service` in this process, and talk to it over an in-memory pipe: `service` is
    /// the server's admin router. No credentials are sent, and the service is dropped once
    /// the client is. Must be called inside a tokio runtime.
    #[cfg(feature = "embed")]
    pub fn embed<S, B>(service: S) -> AdminClient
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
        AdminClient { inner }
    }

    /// Start `command` as the server, and talk to it over its stdin and stdout. Must be called
    /// inside a tokio runtime.
    pub fn spawn(command: Command) -> io::Result<AdminClient> {
        let inner = http_over_x::Client::spawn(command)?;
        Ok(AdminClient { inner })
    }

    /// Reach the server at `url`, its base `http` or `https` URL, sending `auth`, the
    /// credentials it wants, and `headers`, whatever else it asks of each request, with every
    /// request. Refused when a token or a password would go in the clear: over `http` to any
    /// host but this machine.
    pub fn connect(url: &str, auth: Auth, headers: HeaderMap) -> Result<AdminClient, Error> {
        let inner = http_over_x::Client::connect(url, auth, headers)?;
        Ok(AdminClient { inner })
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

    /// Keep the agent `name` no longer, nor its sessions.
    pub async fn remove_agent(&self, name: impl AsRef<str>) -> Result<(), Error> {
        let path = format!("/agents/{}", segment(name.as_ref()));
        self.inner.delete(&path).await
    }
}

/// A client of the server `inner` reaches, however it was made.
impl From<http_over_x::Client> for AdminClient {
    fn from(inner: http_over_x::Client) -> Self {
        AdminClient { inner }
    }
}
