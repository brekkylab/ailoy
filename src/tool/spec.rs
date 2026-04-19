use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::agent::AgentSpec;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BuiltinToolProvider {
    WebSearch {},

    PythonRepl {
        /// Python version to provision (e.g. `"3.12"`). `None` → latest stable.
        python_version: Option<String>,
        /// Persistent venv path. `None` → temp dir cleaned up when the tool is dropped.
        /// Supports `~` expansion.
        venv_path: Option<String>,
        /// Packages to pre-install before the first tool call.
        #[serde(default)]
        packages: Vec<String>,
    },
}

/// Transport configuration for an MCP (Model Context Protocol) tool server.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MCPToolProvider {
    /// Spawns a child process and communicates over its stdio
    Stdio { command: String },

    /// Connects to a remote MCP server over HTTP streaming
    StreamableHTTP { url: Url },
}

/// Identifies where a tool's implementation lives at runtime.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ToolProvider {
    /// A tool baked into the agent runtime, referenced by `name`
    Builtin(BuiltinToolProvider),

    /// A tool served by an external MCP server described by [`MCPToolProvider`]
    MCP(MCPToolProvider),

    /// A sub-agent exposed as a callable tool.
    ///
    /// The `spec` must have a [`card`](AgentSpec::card) — its `name` becomes
    /// the tool name the orchestrating agent calls, and its `description` is
    /// shown to the model to help it decide when to delegate.
    SubAgent(AgentSpec),

    /// A remote A2A (Agent-to-Agent) server exposed as a callable tool.
    ///
    /// At startup the runtime fetches the agent card from
    /// `{url}/.well-known/agent-card.json` to learn its name and description,
    /// then exposes it as a tool that the orchestrating agent can call with a
    /// plain-text task string.
    A2A { url: Url },
}
