use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
///   [`ToolProvider`] keyed by tool name (built-ins, MCP servers, remote A2A
///   agents, or custom function tools).  When an agent is constructed, the
///   [`ToolDesc`](crate::tool::ToolDesc)s listed in its
///   [`AgentSpec::tools`](crate::agent::AgentSpec::tools) are resolved against
///   this registry to produce the [`ToolFunc`](crate::tool::ToolFunc)s that
///   actually run.
///
/// `AgentProvider` is separate from [`AgentSpec`](crate::agent::AgentSpec) because
/// these settings describe *how* to run an agent, not *what* the agent is.  Swapping
/// the API endpoint or key does not change the agent's identity; swapping the model
/// or instruction does.
///
/// Both fields are public — populate them directly, e.g.
/// `provider.models.insert_api("openai/gpt-4o".into(), LangModelAPISchema::OpenAI, "https://api.openai.com/v1/responses", Some(key))?`.
/// [`tools`](AgentProvider::tools) starts pre-loaded with every built-in tool;
/// use [`ToolProvider::empty`](crate::tool::ToolProvider::empty) to opt out.
#[derive(Clone)]
pub struct AgentProvider {
    /// Registry of all available language models, keyed by model identifier
    /// (e.g. `"openai/gpt-4o"` or `"anthropic/claude-sonnet-4-6"`). Lookups
    /// are by exact match against [`AgentSpec::model`](crate::agent::AgentSpec::model).
    pub models: LangModelProvider,

    /// Registry of tool sources, keyed by tool name. Each
    /// [`ToolProviderElem`](crate::tool::ToolProviderElem) is resolved to a
    /// [`ToolFunc`](crate::tool::ToolFunc) when an
    /// [`AgentSpec`](crate::agent::AgentSpec) requests a matching name.
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

/// Borrow the process-wide default [`AgentProvider`] for reading.
///
/// Holds a [`std::sync::RwLockReadGuard`]; drop it before performing long
/// operations to avoid blocking writers.  Use [`default_provider_owned`]
/// when you need to release the lock immediately.
pub fn default_provider() -> RwLockReadGuard<'static, AgentProvider> {
    DEFAULT_PROVIDER
        .read()
        .expect("default_provider lock poisoned")
}

/// Borrow the process-wide default [`AgentProvider`] for writing.
pub fn default_provider_mut() -> RwLockWriteGuard<'static, AgentProvider> {
    DEFAULT_PROVIDER
        .write()
        .expect("default_provider lock poisoned")
}
