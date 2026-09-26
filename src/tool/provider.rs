use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use url::Url;

use crate::tool::{
    MCPConnection, MCPToolEntry, ToolDesc, ToolFunc,
    r#impl::{
        get_a2a_tool_desc, get_a2a_tool_func, get_builtin_tool_factories, mcp_tool_desc,
        prefixed_tool_name,
    },
};

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug)]
pub enum MCPToolProviderElem {
    /// Spawns a child process and communicates over its stdio.
    ///
    /// `command` is the executable, not a shell line: `args` are passed through
    /// as written, so nothing here is word-split or glob-expanded.
    Stdio { command: String, args: Vec<String> },

    /// Connects to a remote MCP server over HTTP streaming.
    StreamableHTTP { url: Url },
}

/// One entry in a [`ToolProvider`] — describes where a tool's implementation
/// comes from. Resolved into a [`ToolFunc`] by [`ToolProvider::provide`] at
/// agent startup.
#[derive(Clone)]
pub enum ToolProviderElem {
    /// A function-backed tool. The closure receives the [`ToolDesc`] requested
    /// by the [`AgentSpec`] and returns the [`ToolFunc`] to bind to it. This
    /// lets the function specialise behaviour to the requested description
    /// (e.g. by inspecting parameters), or simply ignore the argument and
    /// return a fixed [`ToolFunc`].
    Function(Arc<dyn Fn(&ToolDesc) -> ToolFunc + Send + Sync + 'static>),

    /// One tool served by an external MCP server, over a connection opened at
    /// registration time and shared with that server's other tools.
    MCP(MCPToolEntry),

    /// A remote A2A (Agent-to-Agent) server exposed as a callable tool.
    ///
    /// At startup the runtime fetches the agent card from
    /// `{url}/.well-known/agent-card.json` to learn its name and description,
    /// then exposes it as a tool that the orchestrating agent can call with a
    /// plain-text task string.
    A2A { url: Url },
}

impl ToolProviderElem {
    /// Materialise this entry into a [`ToolFunc`] bound to `desc`.
    fn provide(&self, desc: &ToolDesc) -> anyhow::Result<ToolFunc> {
        match self {
            ToolProviderElem::Function(factory) => Ok(factory(desc)),
            // The description is ignored: an MCP tool's behaviour is fixed by the
            // server, and the desc in the spec is a copy of what the server
            // already reported at registration.
            ToolProviderElem::MCP(entry) => Ok(entry.tool_func()),
            // Unlike MCP, nothing has to be discovered to *call* an A2A agent:
            // the task is in the arguments and the address is in the entry. Only
            // the description needs the agent card, and that was fetched at
            // registration (see `register_a2a`).
            ToolProviderElem::A2A { url } => Ok(get_a2a_tool_func(url)),
        }
    }
}

/// Registry of tool sources that an agent can draw from at startup.
///
/// `ToolProvider` is the `tools` field of [`AgentProvider`](crate::agent::AgentProvider).
/// Each entry is keyed by tool name and contributes a [`ToolFunc`] when an
/// agent's [`AgentSpec`] requests it (see [`ToolProvider::provide`]).
///
/// The default constructor pre-registers every built-in tool under its
/// canonical name; start from [`ToolProvider::empty`] to opt out.
#[derive(Clone)]
pub struct ToolProvider {
    inner: BTreeMap<String, ToolProviderElem>,
}

impl Default for ToolProvider {
    fn default() -> Self {
        let mut inner = BTreeMap::new();
        for (name, factory) in get_builtin_tool_factories() {
            inner.insert(name.to_string(), ToolProviderElem::Function(factory));
        }
        Self { inner }
    }
}

impl ToolProvider {
    /// Create a provider pre-populated with every built-in tool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a provider with no entries, including no built-ins.
    pub fn empty() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Register a tool whose behaviour is a fixed [`ToolFunc`], regardless of
    /// the [`ToolDesc`] the spec requests.
    pub fn insert_func(
        &mut self,
        name: impl Into<String>,
        f: ToolFunc,
    ) -> Option<ToolProviderElem> {
        self.inner.insert(
            name.into(),
            ToolProviderElem::Function(Arc::new(move |_| f.clone())),
        )
    }

    /// Register a tool whose [`ToolFunc`] is constructed lazily from the
    /// [`ToolDesc`] the spec requests. Useful when the function needs to
    /// inspect the parameters schema or other metadata supplied by the spec.
    pub fn insert_func_factory(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&ToolDesc) -> ToolFunc + Send + Sync + 'static,
    ) -> Option<ToolProviderElem> {
        self.inner
            .insert(name.into(), ToolProviderElem::Function(Arc::new(f)))
    }

    /// Register a remote A2A agent under `name`.
    ///
    /// Only the entry: the [`ToolDesc`] a spec needs comes from the agent's
    /// card, which is a network fetch. [`register_a2a`] does both halves and
    /// hands back the desc.
    pub fn insert_a2a(
        &mut self,
        name: impl Into<String>,
        url: impl Into<Url>,
    ) -> Option<ToolProviderElem> {
        self.inner
            .insert(name.into(), ToolProviderElem::A2A { url: url.into() })
    }

    /// Register every tool an MCP server reported, under `prefix`.
    ///
    /// One server becomes many entries — `{prefix}__{remote name}` each — because
    /// the registry is a flat name-keyed map and two servers may well both offer
    /// a `search`. The returned [`ToolDesc`]s are exactly those entries, ready to
    /// hand to [`AgentSpec::tools`](crate::agent::AgentSpec::tools); a spec never
    /// learns that an MCP server was involved.
    ///
    /// Connecting is the caller's step ([`MCPToolProviderElem::connect`]) and not
    /// part of this one, because it awaits and this registry lives behind a
    /// `std` lock: a guard held across an `.await` would make the whole future
    /// `!Send`. [`register_mcp_stdio`] and [`register_mcp_streamable_http`] do
    /// both halves in the right order for callers who want one call.
    ///
    /// An existing entry of the same name is replaced, as with any other insert.
    pub fn insert_mcp(&mut self, prefix: impl AsRef<str>, conn: MCPConnection) -> Vec<ToolDesc> {
        let prefix = prefix.as_ref();
        let conn = Arc::new(conn);

        let mut descs = Vec::with_capacity(conn.tools().len());
        for tool in conn.tools() {
            let desc = mcp_tool_desc(prefix, tool);
            self.inner.insert(
                desc.name.clone(),
                ToolProviderElem::MCP(MCPToolEntry::new(conn.clone(), tool.name.to_string())),
            );
            descs.push(desc);
        }
        descs
    }

    /// Drop every entry registered under `prefix` and close the session behind
    /// them — which, for a stdio server, ends the child process.
    ///
    /// Worth calling: a [`ToolProvider`] in the process-wide registry lives as
    /// long as the process, so a server registered there outlives every agent
    /// that used it unless something says otherwise.
    ///
    /// Returns how many entries were removed.
    pub fn remove_mcp(&mut self, prefix: impl AsRef<str>) -> usize {
        let wanted = prefixed_tool_name(prefix.as_ref(), "");
        // `prefixed_tool_name("github", "")` is "github__", so this matches the
        // separator too and a prefix cannot swallow a longer one beside it.
        let names: Vec<String> = self
            .inner
            .iter()
            .filter(|(name, elem)| {
                matches!(elem, ToolProviderElem::MCP(_)) && name.starts_with(&wanted)
            })
            .map(|(name, _)| name.clone())
            .collect();

        let mut conn = None;
        for name in &names {
            if let Some(ToolProviderElem::MCP(entry)) = self.inner.remove(name) {
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

    /// Look up a registered entry by name.
    pub fn get(&self, name: &str) -> Option<&ToolProviderElem> {
        self.inner.get(name)
    }

    /// Iterate over all registered `(name, entry)` pairs.
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, String, ToolProviderElem> {
        self.inner.iter()
    }

    /// Resolve every [`ToolDesc`] listed in `spec.tools` to a [`ToolFunc`]
    /// by looking up the matching entry in this provider. The returned vector
    /// matches `spec.tools` element-for-element. Returns an error if any
    /// requested tool name is not registered.
    ///
    /// Called by [`Agent::try_with_provider_and_state`](crate::agent::Agent::try_with_provider_and_state)
    /// during agent construction.
    pub fn provide(&self, spec: &[ToolDesc]) -> anyhow::Result<HashMap<String, ToolFunc>> {
        let mut funcs = HashMap::with_capacity(spec.len());
        for desc in spec {
            let elem = self.inner.get(&desc.name).ok_or_else(|| {
                anyhow::anyhow!("tool '{}' not registered in ToolProvider", desc.name)
            })?;
            funcs.insert(desc.name.clone(), elem.provide(desc)?);
        }
        Ok(funcs)
    }
}

/// Process-wide named registry of [`ToolProvider`] instances.
///
/// Populated at first access with a single `"default"` entry built from
/// [`ToolProvider::default`] (i.e. the registry pre-loaded with every built-in
/// tool).  Additional named providers can be registered via
/// [`get_tool_providers_mut`], and looked up via [`get_tool_providers`].
static TOOL_PROVIDERS: LazyLock<RwLock<HashMap<String, ToolProvider>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert("default".to_string(), ToolProvider::default());
    RwLock::new(map)
});

/// Borrow the process-wide [`ToolProvider`] registry for reading.
pub fn get_tool_providers() -> RwLockReadGuard<'static, HashMap<String, ToolProvider>> {
    TOOL_PROVIDERS.read().expect("tool_providers lock poisoned")
}

/// Borrow the process-wide [`ToolProvider`] registry for writing.
pub fn get_tool_providers_mut() -> RwLockWriteGuard<'static, HashMap<String, ToolProvider>> {
    TOOL_PROVIDERS
        .write()
        .expect("tool_providers lock poisoned")
}

/// Connect to a stdio MCP server and register its tools under `prefix` in the
/// named provider, returning the [`ToolDesc`]s to put in an
/// [`AgentSpec`](crate::agent::AgentSpec).
///
/// The two halves in the order that keeps the future `Send`: the connection is
/// opened first, and the registry lock is taken only afterwards, for the
/// insert. Doing it by hand in the other order — holding the guard from
/// [`get_tool_providers_mut`] across the `.await` — compiles but poisons the
/// future for [`tokio::spawn`].
///
/// The server runs on the host, outside the [`ConsoleClient`](crate::console::ConsoleClient)
/// sandbox that the built-in tools use: `ConsoleClient::exec` is one-shot, so there is
/// nowhere inside it to keep a process that must hold its stdio open. An MCP
/// server therefore has whatever access this process has — register only servers
/// the caller trusts.
pub async fn register_mcp_stdio(
    provider: impl AsRef<str>,
    prefix: impl AsRef<str>,
    command: impl AsRef<str>,
    args: impl IntoIterator<Item = impl AsRef<str>>,
) -> anyhow::Result<Vec<ToolDesc>> {
    let conn = MCPConnection::stdio(command, args).await?;
    register_mcp_connection(provider, prefix, conn)
}

/// Connect to a streamable-HTTP MCP server and register its tools under
/// `prefix` in the named provider. The HTTP counterpart of
/// [`register_mcp_stdio`].
pub async fn register_mcp_streamable_http(
    provider: impl AsRef<str>,
    prefix: impl AsRef<str>,
    url: impl AsRef<str>,
) -> anyhow::Result<Vec<ToolDesc>> {
    let conn = MCPConnection::streamable_http(url).await?;
    register_mcp_connection(provider, prefix, conn)
}

/// The lock-holding half of the two `register_mcp_*` helpers: no `.await`
/// inside, so the guard never crosses a suspension point.
fn register_mcp_connection(
    provider: impl AsRef<str>,
    prefix: impl AsRef<str>,
    conn: MCPConnection,
) -> anyhow::Result<Vec<ToolDesc>> {
    let provider = provider.as_ref();
    let mut registry = get_tool_providers_mut();
    let tp = registry
        .get_mut(provider)
        .ok_or_else(|| anyhow::anyhow!("tool_provider '{}' not registered", provider))?;
    Ok(tp.insert_mcp(prefix, conn))
}

/// Remove an MCP server's tools from the named provider and close its session.
///
/// Returns how many entries were removed; an unknown provider name is an error,
/// but an unknown prefix simply removes nothing.
pub fn unregister_mcp(provider: impl AsRef<str>, prefix: impl AsRef<str>) -> anyhow::Result<usize> {
    let provider = provider.as_ref();
    let mut registry = get_tool_providers_mut();
    let tp = registry
        .get_mut(provider)
        .ok_or_else(|| anyhow::anyhow!("tool_provider '{}' not registered", provider))?;
    Ok(tp.remove_mcp(prefix))
}

/// Fetch a remote A2A agent's card, register it under `name` in the named
/// provider, and return the [`ToolDesc`] to put in an
/// [`AgentSpec`](crate::agent::AgentSpec).
///
/// The A2A counterpart of [`register_mcp_stdio`], and split the same way for
/// the same reason: the card is fetched before the registry lock is taken, so
/// no guard is held across an `.await`.
///
/// One agent is one tool, so unlike an MCP server there is no prefix and no
/// fan-out — `name` is the tool name the model will see, with any character the
/// model APIs refuse mapped to `_`.
pub async fn register_a2a(
    provider: impl AsRef<str>,
    name: impl AsRef<str>,
    url: Url,
) -> anyhow::Result<ToolDesc> {
    let desc = get_a2a_tool_desc(name.as_ref(), &url).await?;

    let provider = provider.as_ref();
    let mut registry = get_tool_providers_mut();
    let tp = registry
        .get_mut(provider)
        .ok_or_else(|| anyhow::anyhow!("tool_provider '{}' not registered", provider))?;

    // Keyed by the sanitised name the desc ended up with, not the raw argument:
    // `provide` looks entries up by `ToolDesc::name`, so the two must agree.
    tp.insert_a2a(desc.name.clone(), url);
    Ok(desc)
}
