use std::sync::LazyLock;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{lang_model::LangModelProvider, tool::ToolProvider};

/// Top-level configuration shared across agents.
///
/// `AgentProvider` is a **shared, top-level configuration object** that is passed
/// unchanged to every agent in a session.  It answers two questions:
///
/// * **How do I call a model?** — [`models`](AgentProvider::models) maps each model
///   identifier that an [`AgentSpec`](crate::agent::AgentSpec) might reference to its
///   API schema, endpoint URL, and credentials.  An agent looks up its
///   [`AgentSpec::model`](crate::agent::AgentSpec::model) here at construction.
///
/// * **How do I initialise a tool?** — [`tools`](AgentProvider::tools) is a
///   [`ToolProvider`] that describes the tool sources to materialise (built-ins,
///   MCP servers, remote A2A agents, or custom factories).  Every entry is built
///   into a runtime [`Tool`](crate::tool::Tool) at agent startup and added to the
///   agent's tool list.
///
/// `AgentProvider` is separate from [`AgentSpec`](crate::agent::AgentSpec) because
/// these settings describe *how* to run an agent, not *what* the agent is.  Swapping
/// the API endpoint or key does not change the agent's identity; swapping the model
/// or instruction does.
///
/// Both fields are public — populate them directly, e.g.
/// `provider.models.insert("openai/gpt-4o".into(), LangModelProvider::openai(key))`
/// or `provider.tools = ToolProvider::new().bash().web_search()`.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentProvider {
    /// Registry of all available language models, keyed by model identifier
    /// (e.g. `"openai/gpt-4o"` or `"anthropic/claude-sonnet-4-6"`). Lookups
    /// are by exact match against [`AgentSpec::model`](crate::agent::AgentSpec::model).
    pub models: LangModelProvider,

    /// Tool sources to materialise at agent startup.  Each [`ToolProviderElem`](crate::tool::ToolProviderElem)
    /// in the underlying list contributes exactly one runtime [`Tool`](crate::tool::Tool).
    pub tools: ToolProvider,
}

impl AgentProvider {
    pub fn new() -> Self {
        Self {
            models: Default::default(),
            tools: Default::default(),
        }
    }
}

impl Default for AgentProvider {
    fn default() -> Self {
        Self::new()
    }
}

static DEFAULT_PROVIDER: LazyLock<RwLock<AgentProvider>> =
    LazyLock::new(|| RwLock::new(AgentProvider::new()));

pub async fn default_provider() -> RwLockReadGuard<'static, AgentProvider> {
    DEFAULT_PROVIDER.read().await
}

pub async fn default_provider_mut() -> RwLockWriteGuard<'static, AgentProvider> {
    DEFAULT_PROVIDER.write().await
}
