use url::Url;

use crate::{
    agent::default_provider,
    tool::{MCPConnection, MCPServer, ToolDesc, ToolFunc, ToolProvider, prefixed_tool_name},
};

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug)]
pub enum MCPTransport {
    /// Spawns a child process and communicates over its stdio.
    ///
    /// `command` is the executable, not a shell line: `args` are passed through
    /// as written, so nothing here is word-split or glob-expanded.
    Stdio { command: String, args: Vec<String> },

    /// Connects to a remote MCP server over HTTP streaming.
    StreamableHTTP { url: Url },
}

/// Provide each tool `provider` describes through the [`default_provider`], in place of one of
/// its name. Answers their names, which a spec lists to give an agent the tools.
///
/// A function of this process is `(desc, func)`.
pub fn set_tool_provider(provider: impl ToolProvider + 'static) -> Vec<String> {
    default_provider()
        .write()
        .unwrap()
        .set_tool_provider(provider)
}

/// Stop providing the tool `name` through the [`default_provider`]. Answers whether it was
/// provided.
pub fn remove_tool_provider(name: &str) -> bool {
    default_provider()
        .write()
        .unwrap()
        .remove_tool_provider(name)
}

/// Make the tool `name` through the [`default_provider`], with the description it is shown to
/// the model by.
pub fn create_tool(name: &str) -> anyhow::Result<(ToolDesc, ToolFunc)> {
    default_provider().read().unwrap().create_tool(name)
}

/// Connect to a stdio MCP server and provide each of its tools under `prefix` through the
/// [`default_provider`], in place of a server already there under it, answering their names to
/// put in an [`AgentSpec`](crate::agent::AgentSpec).
///
/// The server runs on the host, outside the [`Sandbox`](crate::console::Sandbox) that the
/// console tools use, where a process runs as a job and is not one to hold the stdio of. An MCP
/// server therefore has whatever access this process has — register only servers
/// the caller trusts.
pub async fn register_mcp_stdio(
    prefix: impl AsRef<str>,
    command: impl AsRef<str>,
    args: impl IntoIterator<Item = impl AsRef<str>>,
) -> anyhow::Result<Vec<String>> {
    let conn = MCPConnection::stdio(command, args).await?;
    Ok(set_mcp_server(prefix.as_ref(), conn))
}

/// Connect to a streamable-HTTP MCP server and provide each of its tools under `prefix`, as
/// [`register_mcp_stdio`] does.
pub async fn register_mcp_streamable_http(
    prefix: impl AsRef<str>,
    url: impl AsRef<str>,
) -> anyhow::Result<Vec<String>> {
    let conn = MCPConnection::streamable_http(url).await?;
    Ok(set_mcp_server(prefix.as_ref(), conn))
}

/// The server `conn` in place of the one under `prefix`, whose session ends once its tools go.
fn set_mcp_server(prefix: &str, conn: MCPConnection) -> Vec<String> {
    let mut provider = default_provider().write().unwrap();
    provider.remove_tool_providers_by_prefix(&prefixed_tool_name(prefix, ""));
    provider.set_tool_provider(MCPServer::new(prefix, conn))
}

/// Remove an MCP server's tools from the [`default_provider`], which ends its session. Returns
/// how many were removed; an unknown prefix removes nothing.
///
/// Worth calling: the provider lives as long as the process, so a server registered there
/// outlives every agent that used it unless something says otherwise.
pub fn unregister_mcp(prefix: impl AsRef<str>) -> usize {
    let prefix = prefixed_tool_name(prefix.as_ref(), "");
    default_provider()
        .write()
        .unwrap()
        .remove_tool_providers_by_prefix(&prefix)
}
