use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Named bundle that ties an agent to a [`LangModelProvider`] and a
/// [`ToolProvider`] by **name** rather than by value.
///
/// `AgentProvider` itself does not hold model or tool definitions — those live
/// in their own process-wide registries
/// ([`lang_model_providers`](crate::lang_model::lang_model_providers),
/// [`tool_providers`](crate::tool::tool_providers)).  This struct only stores
/// the *keys* into those registries, so a single update to e.g.
/// [`lang_model_providers_mut`](crate::lang_model::lang_model_providers_mut)
/// is immediately visible to every `AgentProvider` that references that name.
///
/// Mirror registries:
/// * [`get_agent_providers`] / [`get_agent_providers_mut`] — the global map of
///   `AgentProvider`s, pre-populated with a single `"default"` entry that
///   points at the `"default"` lang-model and tool providers.
///
/// At agent construction time
/// (e.g. [`AgentBuilder::build`](crate::agent::AgentBuilder::build)) the
/// builder looks up the chosen [`AgentProvider`] by name, then resolves the
/// nested `lang_model_provider` / `tool_provider` names against their
/// respective registries.
#[derive(Clone, Debug)]
pub struct AgentProvider {
    /// Key into [`lang_model_providers`](crate::lang_model::lang_model_providers).
    pub lang_model_provider: String,

    /// Key into [`tool_providers`](crate::tool::tool_providers).
    pub tool_provider: String,
}

impl AgentProvider {
    /// Bundle the two given provider names.  Both names must exist in their
    /// respective registries at agent-construction time; this constructor does
    /// not validate them.
    pub fn new(lang_model_provider: impl Into<String>, tool_provider: impl Into<String>) -> Self {
        Self {
            lang_model_provider: lang_model_provider.into(),
            tool_provider: tool_provider.into(),
        }
    }
}

impl Default for AgentProvider {
    /// Returns the canonical bundle `{ "default", "default" }`, matching the
    /// `"default"` entries auto-registered in the lang-model and tool
    /// provider registries.
    fn default() -> Self {
        Self::new("default", "default")
    }
}

/// Process-wide named registry of [`AgentProvider`] bundles.
///
/// Pre-populated with a single `"default"` entry equal to
/// [`AgentProvider::default`].  Look up additional named bundles via
/// [`get_agent_providers`]; register new ones via [`get_agent_providers_mut`].
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
