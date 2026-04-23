use std::{collections::HashMap, sync::Arc};

use tokio::sync::Mutex;

use crate::{
    agent::{Agent, AgentProvider, AgentSpec},
    tool::{MCPToolProvider, Tool, ToolFactory, ToolProvider},
    tool_impl::{make_a2a_tool, make_builtin_tool, make_subagent_tool},
};

/// Agent-independent registry of [`ToolFactory`] instances, keyed by tool name.
///
/// Created before any agent is bound and shared across the system.
/// Call [`ToolSet::make_runtime`] with an [`AgentSpec`] to materialise a
/// concrete [`Tool`] for a specific agent instance.
pub struct ToolSet {
    tools: HashMap<String, ToolFactory>,
}

impl ToolSet {
    /// Build an empty [`ToolSet`].
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Build a [`ToolSet`] from the tool providers declared in `provider` and
    /// the sub-agents declared in `spec`.
    pub async fn from_providers(
        spec: &AgentSpec,
        provider: &AgentProvider,
    ) -> anyhow::Result<Self> {
        let mut this = Self::new();

        // Initialise all provider tools.
        for tool_provider in &provider.tools {
            match tool_provider {
                ToolProvider::Builtin(builtin_tool_provider) => {
                    let factory = make_builtin_tool(builtin_tool_provider).await?;
                    this.tools.insert(factory.get_name().into(), factory);
                }
                ToolProvider::A2A { url } => {
                    let factory = make_a2a_tool(url.clone()).await?;
                    this.tools.insert(factory.get_name().into(), factory);
                }
                ToolProvider::MCP(mcptool_provider) => match mcptool_provider {
                    MCPToolProvider::Stdio { command: _ } => todo!(),
                    MCPToolProvider::StreamableHTTP { url: _ } => todo!(),
                },
            }
        }

        // Initialise sub-agents using the toolset built above.
        for sub_spec in &spec.subagents {
            let card = sub_spec.card.clone().ok_or_else(|| {
                anyhow::anyhow!(
                    "SubAgent spec must have a card (name + description) to be \
                     registered as a tool"
                )
            })?;
            let sub_agent = Agent::try_with_tools(sub_spec.clone(), provider, &this).await?;
            let sub_agent = Arc::new(Mutex::new(sub_agent));
            let factory = make_subagent_tool(card, sub_agent);
            this.tools.insert(factory.get_name().into(), factory);
        }

        Ok(this)
    }

    /// Insert a tool by key.
    ///
    /// `f` can be a [`ToolFunc`] (wrapped in an `Arc` automatically) or an
    /// existing `Arc<ToolFunc>` — e.g. one obtained from [`Tool::get_func`].
    pub fn insert(&mut self, key: impl Into<String>, factory: ToolFactory) -> Option<ToolFactory> {
        self.tools.insert(key.into(), factory)
    }

    pub fn remove(&mut self, key: impl AsRef<str>) -> Option<ToolFactory> {
        self.tools.remove(key.as_ref())
    }

    pub fn make_runtime(&self, key: impl AsRef<str>, spec: &AgentSpec) -> Option<Tool> {
        self.tools
            .get(key.as_ref())
            .map(|factory| factory.make(spec))
    }

    /// Merge another [`ToolSet`] into this one.
    ///
    /// Tools from `other` are inserted into `self`. If a key already exists in
    /// `self`, the entry from `other` overwrites it.
    pub fn merge(&mut self, other: ToolSet) {
        self.tools.extend(other.tools);
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, String, ToolFactory> {
        self.tools.iter()
    }

    pub fn iter_mut(&mut self) -> std::collections::hash_map::IterMut<'_, String, ToolFactory> {
        self.tools.iter_mut()
    }

    /// Return the names of all registered tools.
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

impl FromIterator<(String, ToolFactory)> for ToolSet {
    fn from_iter<I: IntoIterator<Item = (String, ToolFactory)>>(iter: I) -> Self {
        Self {
            tools: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for ToolSet {
    type Item = (String, ToolFactory);
    type IntoIter = std::collections::hash_map::IntoIter<String, ToolFactory>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

impl IntoIterator for &ToolSet {
    type Item = (String, ToolFactory);
    type IntoIter = std::vec::IntoIter<(String, ToolFactory)>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}
