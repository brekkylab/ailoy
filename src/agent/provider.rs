use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Named bundle that ties an agent to a [`LangModelProvider`](crate::lang_model::LangModelProvider) and a
/// [`ToolProvider`](crate::tool::ToolProvider) by **name** rather than by value.
///
/// It stores only keys into the process-wide registries
/// ([`get_lm_providers`](crate::lang_model::get_lm_providers),
/// [`get_tool_providers`](crate::tool::get_tool_providers)), so a registry update is immediately
/// visible to every `AgentProvider` naming that entry. The names are resolved at agent
/// construction time (e.g. [`AgentBuilder::build`](crate::agent::AgentBuilder::build)).
///
/// Bundles themselves are registered in [`get_agent_providers`] / [`get_agent_providers_mut`].
#[derive(Clone, Debug)]
pub struct AgentProvider {
    /// Key into [`get_lm_providers`](crate::lang_model::get_lm_providers).
    pub lang_model_provider: String,

    /// Key into [`get_tool_providers`](crate::tool::get_tool_providers).
    pub tool_provider: String,
}

impl AgentProvider {
    /// Bundle two provider names, unvalidated; both must be registered by
    /// agent-construction time.
    pub fn new(lang_model_provider: impl Into<String>, tool_provider: impl Into<String>) -> Self {
        Self {
            lang_model_provider: lang_model_provider.into(),
            tool_provider: tool_provider.into(),
        }
    }
}

impl Default for AgentProvider {
    /// `{ "default", "default" }`, the entries auto-registered in both registries.
    fn default() -> Self {
        Self::new("default", "default")
    }
}

/// Process-wide named registry of [`AgentProvider`] bundles.
///
/// Pre-populated with `"default"` = [`AgentProvider::default`].
static AGENT_PROVIDERS: LazyLock<RwLock<HashMap<String, AgentProvider>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert("default".to_string(), AgentProvider::default());
    RwLock::new(map)
});

/// Borrow the process-wide [`AgentProvider`] registry for reading.
pub fn get_agent_providers() -> RwLockReadGuard<'static, HashMap<String, AgentProvider>> {
    AGENT_PROVIDERS
        .read()
        .expect("agent_providers lock poisoned")
}

/// Borrow the process-wide [`AgentProvider`] registry for writing.
pub fn get_agent_providers_mut() -> RwLockWriteGuard<'static, HashMap<String, AgentProvider>> {
    AGENT_PROVIDERS
        .write()
        .expect("agent_providers lock poisoned")
}
