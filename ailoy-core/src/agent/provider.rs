use std::{fmt, sync::Arc};

use indexmap::IndexMap;

use crate::{
    agent::{
        AgentListOutputItem, AgentPutArgs, AgentState, SessionCreateArgs, SessionListOutputItem,
    },
    tool::{ToolDesc, ToolFunc, ToolProvider},
};

/// Why an [`AgentProvider`] did not do what was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// No agent is kept under this name.
    NoAgent(String),
    /// The agent has no session of this id.
    NoSession(String),
    /// A spec names this tool, and nothing provides it.
    NoTool(String),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::NoAgent(name) => write!(f, "there is no agent '{name}'"),
            Refusal::NoSession(id) => write!(f, "the agent has no session '{id}'"),
            Refusal::NoTool(name) => write!(f, "nothing provides the tool '{name}'"),
        }
    }
}

impl std::error::Error for Refusal {}

/// An agent kept, and the state of each of its sessions by id.
struct Kept {
    args: AgentPutArgs,
    sessions: IndexMap<String, AgentState>,
}

/// What an agent runs with where it runs: the providers of the tools a spec names, and the
/// agents kept by name, each with its sessions. The agents and sessions are kept in memory.
#[derive(Default)]
pub struct AgentProvider {
    /// Each tool's description and the provider that makes it, by the tool's name.
    tools: IndexMap<String, (ToolDesc, Arc<dyn ToolProvider>)>,
    agents: IndexMap<String, Kept>,
}

impl AgentProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide each tool `provider` describes, in place of one of its name already provided.
    /// Answers their names, which a spec lists to give an agent the tools.
    pub fn set_tool_provider(&mut self, provider: impl ToolProvider + 'static) -> Vec<String> {
        let provider: Arc<dyn ToolProvider> = Arc::new(provider);
        let descs = provider.descs();
        let mut names = Vec::with_capacity(descs.len());
        for desc in descs {
            names.push(desc.name.clone());
            self.tools
                .insert(desc.name.clone(), (desc, provider.clone()));
        }
        names
    }

    /// Stop providing the tool `name`. Answers whether it was provided.
    pub fn remove_tool_provider(&mut self, name: &str) -> bool {
        self.tools.shift_remove(name).is_some()
    }

    /// Stop providing every tool whose name starts with `prefix`. Answers how many there were.
    pub fn remove_tool_providers_by_prefix(&mut self, prefix: &str) -> usize {
        let before = self.tools.len();
        self.tools.retain(|name, _| !name.starts_with(prefix));
        before - self.tools.len()
    }

    /// The description of every tool provided, in the order each name was first provided.
    pub fn tool_descs(&self) -> Vec<ToolDesc> {
        self.tools.values().map(|(desc, _)| desc.clone()).collect()
    }

    /// Make the tool `name` through its provider, with the description it is shown to a model
    /// by.
    pub fn create_tool(&self, name: &str) -> anyhow::Result<(ToolDesc, ToolFunc)> {
        let (desc, provider) = self
            .tools
            .get(name)
            .ok_or_else(|| Refusal::NoTool(name.to_owned()))?;
        Ok((desc.clone(), provider.make(name)?))
    }

    /// Keep the agent `args` describes under `name`, in place of one already there, whose
    /// sessions stay. Refused when its spec names a tool nothing provides.
    pub fn put_agent(
        &mut self,
        name: &str,
        args: AgentPutArgs,
    ) -> Result<AgentListOutputItem, Refusal> {
        if let Some(tool) = args
            .spec
            .tools
            .iter()
            .find(|t| !self.tools.contains_key(*t))
        {
            return Err(Refusal::NoTool(tool.clone()));
        }
        let item = AgentListOutputItem {
            name: name.to_owned(),
            spec: args.spec.clone(),
            model_key_provider: args.model_key_provider,
        };
        match self.agents.get_mut(name) {
            Some(kept) => kept.args = args,
            None => {
                let sessions = IndexMap::new();
                self.agents.insert(name.to_owned(), Kept { args, sessions });
            }
        }
        Ok(item)
    }

    /// The agent kept under `name`.
    pub fn get_agent(&self, name: &str) -> Result<AgentListOutputItem, Refusal> {
        let kept = self
            .agents
            .get(name)
            .ok_or_else(|| Refusal::NoAgent(name.to_owned()))?;
        Ok(AgentListOutputItem {
            name: name.to_owned(),
            spec: kept.args.spec.clone(),
            model_key_provider: kept.args.model_key_provider,
        })
    }

    /// Every agent kept, in the order each name was first put.
    pub fn list_agents(&self) -> Vec<AgentListOutputItem> {
        self.agents
            .iter()
            .map(|(name, kept)| AgentListOutputItem {
                name: name.clone(),
                spec: kept.args.spec.clone(),
                model_key_provider: kept.args.model_key_provider,
            })
            .collect()
    }

    /// Keep the agent `name` no longer, nor its sessions. Nothing happens when there is none.
    pub fn remove_agent(&mut self, name: &str) {
        self.agents.shift_remove(name);
    }

    /// Start a session of the agent `agent`, as `args` describes it: a state with a new id.
    pub fn create_session(
        &mut self,
        agent: &str,
        args: SessionCreateArgs,
    ) -> Result<AgentState, Refusal> {
        let kept = self
            .agents
            .get_mut(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let state = AgentState::new().with_history(args.history);
        kept.sessions.insert(state.id.clone(), state.clone());
        Ok(state)
    }

    /// The state of the session `id` of the agent `agent`.
    pub fn get_session(&self, agent: &str, id: &str) -> Result<AgentState, Refusal> {
        let kept = self
            .agents
            .get(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let session = kept.sessions.get(id);
        session
            .cloned()
            .ok_or_else(|| Refusal::NoSession(id.to_owned()))
    }

    /// Every session of the agent `agent`, in the order each was started.
    pub fn list_sessions(&self, agent: &str) -> Result<Vec<SessionListOutputItem>, Refusal> {
        let kept = self
            .agents
            .get(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let sessions = kept.sessions.values();
        Ok(sessions
            .map(|state| SessionListOutputItem {
                id: state.id.clone(),
                message_count: state.history.len(),
            })
            .collect())
    }

    /// Keep the session `id` of the agent `agent` no longer. Nothing happens when there is
    /// none.
    pub fn remove_session(&mut self, agent: &str, id: &str) {
        if let Some(kept) = self.agents.get_mut(agent) {
            kept.sessions.shift_remove(id);
        }
    }
}
