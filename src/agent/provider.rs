use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Names a [`LangModelProvider`](crate::lang_model::LangModelProvider) and a
/// [`ToolProvider`](crate::tool::ToolProvider) in the process-wide registries rather than
/// holding them, so a registry update reaches every bundle naming that entry.
///
/// Registered in [`get_agent_providers`] / [`get_agent_providers_mut`].
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

/// Process-wide registry of [`AgentProvider`] bundles, seeded with [`AgentProvider::default`].
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
