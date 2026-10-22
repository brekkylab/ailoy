use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use url::Url;

use crate::tool::{
    MCPConnection, MCPToolEntry, ToolDesc, ToolFunc,
    r#impl::{get_builtin_tool_factories, mcp_tool_desc, prefixed_tool_name},
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

/// Builds a tool's [`ToolFunc`] from the [`ToolDesc`] a spec requests; may specialise on it
/// or ignore it. An error fails the agent's construction.
pub type ToolFactory = Arc<dyn Fn(&ToolDesc) -> anyhow::Result<ToolFunc> + Send + Sync>;

/// How a tool is built, set under its name with [`set_tool_provider`].
#[derive(Clone)]
pub enum ToolProvider {
    /// A function of this process.
    Function(ToolFactory),

    /// One tool served by an external MCP server, over a connection opened at
    /// registration time and shared with that server's other tools.
    MCP(MCPToolEntry),
}

impl ToolProvider {
    /// A tool whose [`ToolFunc`] is built from the requested [`ToolDesc`].
    pub fn function(
        f: impl Fn(&ToolDesc) -> anyhow::Result<ToolFunc> + Send + Sync + 'static,
    ) -> Self {
        Self::Function(Arc::new(f))
    }

    /// Materialise this provider into a [`ToolFunc`] bound to `desc`.
    pub(crate) fn provide(&self, desc: &ToolDesc) -> anyhow::Result<ToolFunc> {
        match self {
            Self::Function(factory) => factory(desc),
            // The description is ignored: an MCP tool's behaviour is fixed by the
            // server, and the desc in the spec is a copy of what the server
            // already reported at registration.
            Self::MCP(entry) => Ok(entry.tool_func()),
        }
    }
}

/// A tool whose behaviour is a fixed [`ToolFunc`], regardless of the [`ToolDesc`] the
/// spec requests.
impl From<ToolFunc> for ToolProvider {
    fn from(f: ToolFunc) -> Self {
        Self::function(move |_| Ok(f.clone()))
    }
}

/// Providers keyed by tool name, seeded with every built-in tool.
static TOOL_PROVIDERS: LazyLock<RwLock<HashMap<String, ToolProvider>>> = LazyLock::new(|| {
    RwLock::new(
        get_builtin_tool_factories()
            .into_iter()
            .map(|(name, factory)| (name.to_owned(), ToolProvider::Function(factory)))
            .collect(),
    )
});

/// Builds the tool `name` through `provider`, replacing the provider it had.
pub fn set_tool_provider(name: impl Into<String>, provider: impl Into<ToolProvider>) {
    TOOL_PROVIDERS
        .write()
        .unwrap()
        .insert(name.into(), provider.into());
}

/// Removes the provider for `name`, returning it if there was one.
pub fn remove_tool_provider(name: &str) -> Option<ToolProvider> {
    TOOL_PROVIDERS.write().unwrap().remove(name)
}

/// Builds the tool `desc` names through the provider set for it.
pub fn create_tool(desc: &ToolDesc) -> anyhow::Result<ToolFunc> {
    // Clone out so the lock is not held while building.
    let provider = TOOL_PROVIDERS
        .read()
        .unwrap()
        .get(&desc.name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no provider set for tool '{}'", desc.name))?;
    provider.provide(desc)
}

/// Sets a provider for every tool an MCP server reported, under `prefix`.
///
/// One entry per tool, prefixed because the registry is a flat name-keyed map
/// and two servers may both offer a `search`. The returned [`ToolDesc`]s are
/// exactly those entries, ready for [`AgentSpec::tools`](crate::agent::AgentSpec::tools);
/// a spec never learns that an MCP server was involved.
///
/// An existing entry of the same name is replaced, as with any other set.
pub(crate) fn insert_mcp(
    providers: &mut HashMap<String, ToolProvider>,
    prefix: &str,
    conn: MCPConnection,
) -> Vec<ToolDesc> {
    let conn = Arc::new(conn);

    let mut descs = Vec::with_capacity(conn.tools().len());
    for tool in conn.tools() {
        let desc = mcp_tool_desc(prefix, tool);
        providers.insert(
            desc.name.clone(),
            ToolProvider::MCP(MCPToolEntry::new(conn.clone(), tool.name.to_string())),
        );
        descs.push(desc);
    }
    descs
}

/// Drops every MCP entry under `prefix` and closes the session behind them — which,
/// for a stdio server, ends the child process. Returns how many entries were removed.
pub(crate) fn remove_mcp(providers: &mut HashMap<String, ToolProvider>, prefix: &str) -> usize {
    let wanted = prefixed_tool_name(prefix, "");
    // `prefixed_tool_name("github", "")` is "github__", so this matches the
    // separator too and a prefix cannot swallow a longer one beside it.
    let names: Vec<String> = providers
        .iter()
        .filter(|(name, provider)| {
            matches!(provider, ToolProvider::MCP(_)) && name.starts_with(&wanted)
        })
        .map(|(name, _)| name.clone())
        .collect();

    let mut conn = None;
    for name in &names {
        if let Some(ToolProvider::MCP(entry)) = providers.remove(name) {
            conn.get_or_insert_with(|| entry.conn().clone());
        }
    }
    // After the entries are gone this is the last handle unless an agent is
    // mid-call, and cancelling is what stops the session either way.
    if let Some(conn) = conn {
        conn.shutdown();
    }
    names.len()
}

/// Connect to a stdio MCP server and set a provider for each of its tools under
/// `prefix`, returning the [`ToolDesc`]s to put in an [`AgentSpec`](crate::agent::AgentSpec).
///
/// The server runs on the host, outside the [`ConsoleClient`](crate::console::ConsoleClient)
/// sandbox that the built-in tools use: `ConsoleClient::exec` is one-shot, so there is
/// nowhere inside it to keep a process that must hold its stdio open. An MCP
/// server therefore has whatever access this process has — register only servers
/// the caller trusts.
pub async fn register_mcp_stdio(
    prefix: impl AsRef<str>,
    command: impl AsRef<str>,
    args: impl IntoIterator<Item = impl AsRef<str>>,
) -> anyhow::Result<Vec<ToolDesc>> {
    let conn = MCPConnection::stdio(command, args).await?;
    Ok(insert_mcp(
        &mut TOOL_PROVIDERS.write().unwrap(),
        prefix.as_ref(),
        conn,
    ))
}

/// Connect to a streamable-HTTP MCP server and set a provider for each of its tools
/// under `prefix`.
pub async fn register_mcp_streamable_http(
    prefix: impl AsRef<str>,
    url: impl AsRef<str>,
) -> anyhow::Result<Vec<ToolDesc>> {
    let conn = MCPConnection::streamable_http(url).await?;
    Ok(insert_mcp(
        &mut TOOL_PROVIDERS.write().unwrap(),
        prefix.as_ref(),
        conn,
    ))
}

/// Remove an MCP server's tools and close its session. Returns how many were removed;
/// an unknown prefix removes nothing.
///
/// Worth calling: the registry lives as long as the process, so a server registered
/// there outlives every agent that used it unless something says otherwise.
pub fn unregister_mcp(prefix: impl AsRef<str>) -> usize {
    remove_mcp(&mut TOOL_PROVIDERS.write().unwrap(), prefix.as_ref())
}
