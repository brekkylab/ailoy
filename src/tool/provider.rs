use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::{
    agent::AgentSpec,
    tool::{Tool, ToolFactory},
};

/// One built-in tool that ships with the runtime.  Each variant maps to a
/// specific factory in `tool_impl/builtins`.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuiltinToolProviderElem {
    WebSearch {},
    PythonRepl {},
    Bash {},
    Read {},
    Write {},
    Edit {},
    ApplyPatch {},
}

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPToolProviderElem {
    /// Spawns a child process and communicates over its stdio
    Stdio { command: String },

    /// Connects to a remote MCP server over HTTP streaming
    StreamableHTTP { url: Url },
}

/// One entry in a [`ToolProvider`] — describes where a tool's implementation
/// comes from.  Resolved into a runtime [`Tool`] at agent startup.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolProviderElem {
    /// A tool baked into the agent runtime (e.g. `bash`, `python_repl`).
    Builtin(BuiltinToolProviderElem),

    /// A tool served by an external MCP server.
    MCP(MCPToolProviderElem),

    /// A remote A2A (Agent-to-Agent) server exposed as a callable tool.
    ///
    /// At startup the runtime fetches the agent card from
    /// `{url}/.well-known/agent-card.json` to learn its name and description,
    /// then exposes it as a tool that the orchestrating agent can call with a
    /// plain-text task string.
    A2A { url: Url },

    /// A pre-built [`ToolFactory`] supplied by the host application.
    /// Skipped during serialisation.
    #[serde(skip)]
    #[schemars(skip)]
    Custom(ToolFactory),
}

impl ToolProviderElem {
    async fn make(&self, spec: &AgentSpec) -> anyhow::Result<Tool> {
        let tool = match self {
            ToolProviderElem::Builtin(b) => crate::tool_impl::make_builtin_tool_factory(b)
                .await?
                .make(spec),
            ToolProviderElem::MCP(_) => {
                todo!("MCP factory construction is not yet implemented")
            }
            ToolProviderElem::A2A { url } => crate::tool_impl::make_a2a_tool_factory(url.clone())
                .await?
                .make(spec),
            ToolProviderElem::Custom(factory) => factory.make(spec),
        };
        Ok(tool)
    }
}

/// Ordered list of tool sources that an agent should expose at startup.
///
/// `ToolProvider` is the `tools` field of [`AgentProvider`](crate::agent::AgentProvider).
/// Each [`ToolProviderElem`] inside contributes one runtime [`Tool`] when the agent is
/// constructed (see [`ToolProvider::make_runtime`]).  Use the chained builder methods
/// below to assemble it, or push entries directly via serde / `inner` access.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct ToolProvider {
    inner: Vec<ToolProviderElem>,
}

impl ToolProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn web_search(mut self) -> Self {
        self.inner.push(ToolProviderElem::Builtin(
            BuiltinToolProviderElem::WebSearch {},
        ));
        self
    }

    pub fn python_repl(mut self) -> Self {
        self.inner.push(ToolProviderElem::Builtin(
            BuiltinToolProviderElem::PythonRepl {},
        ));
        self
    }

    pub fn bash(mut self) -> Self {
        self.inner
            .push(ToolProviderElem::Builtin(BuiltinToolProviderElem::Bash {}));
        self
    }

    pub fn read(mut self) -> Self {
        self.inner
            .push(ToolProviderElem::Builtin(BuiltinToolProviderElem::Read {}));
        self
    }

    pub fn write(mut self) -> Self {
        self.inner
            .push(ToolProviderElem::Builtin(BuiltinToolProviderElem::Write {}));
        self
    }

    pub fn edit(mut self) -> Self {
        self.inner
            .push(ToolProviderElem::Builtin(BuiltinToolProviderElem::Edit {}));
        self
    }

    pub fn apply_patch(mut self) -> Self {
        self.inner.push(ToolProviderElem::Builtin(
            BuiltinToolProviderElem::ApplyPatch {},
        ));
        self
    }

    pub fn mcp_stdio(mut self, command: impl Into<String>) -> Self {
        self.inner
            .push(ToolProviderElem::MCP(MCPToolProviderElem::Stdio {
                command: command.into(),
            }));
        self
    }

    pub fn mcp_streamable_http(mut self, url: impl Into<Url>) -> Self {
        self.inner
            .push(ToolProviderElem::MCP(MCPToolProviderElem::StreamableHTTP {
                url: url.into(),
            }));
        self
    }

    pub fn a2a(mut self, url: impl Into<Url>) -> Self {
        self.inner.push(ToolProviderElem::A2A { url: url.into() });
        self
    }

    pub fn custom(mut self, factory: ToolFactory) -> Self {
        self.inner.push(ToolProviderElem::Custom(factory));
        self
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ToolProviderElem> {
        self.inner.iter()
    }

    /// Resolve every [`ToolProviderElem`] in registration order to a runtime [`Tool`]
    /// bound to `spec`.  Called by [`Agent::try_with_provider_and_runenv`](crate::agent::Agent::try_with_provider_and_runenv)
    /// during agent construction.
    pub async fn make_runtime(&self, spec: &AgentSpec) -> anyhow::Result<Vec<Tool>> {
        let mut tools = Vec::with_capacity(self.inner.len());
        for elem in &self.inner {
            tools.push(elem.make(spec).await?);
        }
        Ok(tools)
    }
}
