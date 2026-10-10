use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Names a [`ToolProvider`](crate::tool::ToolProvider) in the process-wide registry rather
/// than holding it, so a registry update reaches every bundle naming that entry.
///
/// Models are not named here: a spec's model is built through the provider set for it with
/// [`set_lang_model_provider`](crate::experimental::model::set_lang_model_provider).
///
/// Registered in [`get_agent_providers`] / [`get_agent_providers_mut`].
#[derive(Clone, Debug)]
pub struct AgentProvider {
    /// Key into [`get_tool_providers`](crate::tool::get_tool_providers).
    pub tool_provider: String,
}

impl AgentProvider {
    /// Bundle a tool provider name, unvalidated; it must be registered by
    /// agent-construction time.
    pub fn new(tool_provider: impl Into<String>) -> Self {
        Self {
            tool_provider: tool_provider.into(),
        }
    }
}

impl Default for AgentProvider {
    /// `{ "default" }`, the entry auto-registered in the tool provider registry.
    fn default() -> Self {
        Self::new("default")
    }
}

/// Process-wide registry of [`AgentProvider`] bundles, seeded with [`AgentProvider::default`].
static AGENT_PROVIDERS: LazyLock<RwLock<HashMap<String, AgentProvider>>> = LazyLock::new(|| {
    RwLock::new(HashMap::from([(
        "default".to_owned(),
        AgentProvider::default(),
    )]))
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
