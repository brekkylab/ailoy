use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use url::Url;

use crate::{
    tool::{ToolDesc, ToolFunc},
    tool_impl::get_builtin_tool_factories,
};

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug)]
pub enum MCPToolProviderElem {
    /// Spawns a child process and communicates over its stdio.
    Stdio { command: String },

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

    /// A tool served by an external MCP server.
    MCP(MCPToolProviderElem),

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
            ToolProviderElem::MCP(_) => {
                todo!("MCP factory construction is not yet implemented")
            }
            ToolProviderElem::A2A { url: _ } => {
                todo!("A2A factory construction is not yet implemented")
            }
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

    /// Register a remote A2A agent under `name`. The actual tool description
    /// is discovered from the agent's card at resolve time.
    pub fn insert_a2a(
        &mut self,
        name: impl Into<String>,
        url: impl Into<Url>,
    ) -> Option<ToolProviderElem> {
        self.inner
            .insert(name.into(), ToolProviderElem::A2A { url: url.into() })
    }

    /// Register an MCP server reachable via stdio. Not yet implemented.
    pub fn insert_mcp_stdio(
        &mut self,
        _name: impl Into<String>,
        _command: impl Into<String>,
    ) -> Option<ToolProviderElem> {
        todo!("MCP stdio registration is not yet implemented")
    }

    /// Register an MCP server reachable over streamable HTTP. Not yet implemented.
    pub fn insert_mcp_streamable_http(
        &mut self,
        _name: impl Into<String>,
        _url: impl Into<Url>,
    ) -> Option<ToolProviderElem> {
        todo!("MCP streamable HTTP registration is not yet implemented")
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
    /// Called by [`Agent::try_with_provider_and_runenv`](crate::agent::Agent::try_with_provider_and_runenv)
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
