use std::fmt;

use indexmap::IndexMap;

use crate::{
    agent::{AgentListOutputItem, AgentPutArgs, AgentSpec, AgentState, SessionListOutputItem},
    message::Message,
    model::LangModelProvider,
    tool::ToolProvider,
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
    /// A spec names a model of this org, and nothing makes its models.
    NoModel(String),
    /// The session has a turn running.
    Busy(String),
    /// A sandbox is given to a session of an agent whose spec uses none, or none to one whose
    /// spec does.
    Sandbox(bool),
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::NoAgent(name) => write!(f, "there is no agent '{name}'"),
            Refusal::NoSession(id) => write!(f, "the agent has no session '{id}'"),
            Refusal::NoTool(name) => write!(f, "nothing provides the tool '{name}'"),
            Refusal::NoModel(org) => write!(f, "nothing makes the models of '{org}'"),
            Refusal::Busy(id) => write!(f, "the session '{id}' has a turn running"),
            Refusal::Sandbox(true) => write!(f, "the spec uses a sandbox, and none is given"),
            Refusal::Sandbox(false) => write!(f, "a sandbox is given, and the spec uses none"),
        }
    }
}

impl std::error::Error for Refusal {}

/// An agent kept, and each of its sessions by id.
struct Kept {
    args: AgentPutArgs,
    sessions: IndexMap<String, Session>,
}

/// A session kept: its state, the sandbox made for it, the grant an app reaches it with, and
/// whether a turn runs in it.
struct Session {
    state: AgentState,
    sandbox: Option<virtx::Sandbox>,
    grant: Option<String>,
    running: bool,
}

/// What an agent runs with where it runs: the providers of the models and tools a spec names,
/// the virtx server its sessions' sandboxes are made on, and the agents kept by name, each with
/// its sessions. The agents and sessions are kept in memory.
#[derive(Default)]
pub struct AgentProvider {
    pub tools: ToolProvider,
    pub models: LangModelProvider,
    /// A client of the virtx server, as one of its project's members, that makes the sandboxes
    /// of the sessions the code managing this provider starts.
    pub virtx: Option<virtx::Client>,
    /// The base URL of the virtx server, where an app's session's sandbox is made with the
    /// app's own credentials: none over stdio, which has no apps.
    pub virtx_url: Option<String>,
    agents: IndexMap<String, Kept>,
}

impl AgentProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keep the agent `args` describes under `name`, in place of one already there, whose
    /// sessions stay. Refused when its spec names a model or a tool nothing provides.
    pub fn put_agent(
        &mut self,
        name: &str,
        args: AgentPutArgs,
    ) -> Result<AgentListOutputItem, Refusal> {
        if self.models.get(&args.spec.model).is_none() {
            let org = args
                .spec
                .model
                .split_once('/')
                .map_or(&*args.spec.model, |(org, _)| org);
            return Err(Refusal::NoModel(org.to_owned()));
        }
        if let Some(tool) = args.spec.tools.iter().find(|t| !self.tools.contains(t)) {
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

    /// Keep the agent `name` no longer, nor its sessions; the sandboxes made for those are
    /// answered, for the caller to remove. Nothing happens when there is none.
    pub fn remove_agent(&mut self, name: &str) -> Vec<virtx::Sandbox> {
        let Some(kept) = self.agents.shift_remove(name) else {
            return Vec::new();
        };
        let sessions = kept.sessions.into_values();
        sessions.filter_map(|session| session.sandbox).collect()
    }

    /// Start a session of the agent `agent`, starting with `history` and working in `sandbox`,
    /// which goes with it: a state with a new id. Refused when the agent's spec uses a sandbox
    /// and none is given, or the other way round.
    pub fn create_session(
        &mut self,
        agent: &str,
        history: Vec<Message>,
        sandbox: Option<virtx::Sandbox>,
    ) -> Result<AgentState, Refusal> {
        let kept = self
            .agents
            .get_mut(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let uses = kept.args.spec.use_sandbox;
        if uses != sandbox.is_some() {
            return Err(Refusal::Sandbox(uses));
        }
        let mut state = AgentState::new().with_history(history);
        state.sandbox_id = sandbox.as_ref().map(|sandbox| sandbox.id().to_owned());
        let session = Session {
            state: state.clone(),
            sandbox,
            grant: None,
            running: false,
        };
        kept.sessions.insert(state.id.clone(), session);
        Ok(state)
    }

    /// Give out the grant that reaches the session `id` of the agent `agent` alone, in place of
    /// one given before.
    pub fn grant_session(&mut self, agent: &str, id: &str) -> Result<String, Refusal> {
        let session = self.session_mut(agent, id)?;
        let grant = uuid::Uuid::new_v4().simple().to_string();
        session.grant = Some(grant.clone());
        Ok(grant)
    }

    /// The session `grant` reaches, as its agent and id.
    pub fn find_grant(&self, grant: &str) -> Option<(String, String)> {
        self.agents.iter().find_map(|(name, kept)| {
            let (id, _) = (kept.sessions.iter())
                .find(|(_, session)| session.grant.as_deref() == Some(grant))?;
            Some((name.clone(), id.clone()))
        })
    }

    /// The state of the session `id` of the agent `agent`.
    pub fn get_session(&self, agent: &str, id: &str) -> Result<AgentState, Refusal> {
        let kept = self
            .agents
            .get(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let session = kept.sessions.get(id);
        session
            .map(|session| session.state.clone())
            .ok_or_else(|| Refusal::NoSession(id.to_owned()))
    }

    /// Begin a turn of the session `id` of the agent `agent`: what it runs with, the agent's
    /// spec, the session's state and its sandbox. Refused while a turn runs in it, until
    /// [`end_turn`](Self::end_turn).
    pub fn begin_turn(
        &mut self,
        agent: &str,
        id: &str,
    ) -> Result<(AgentSpec, AgentState, Option<virtx::Sandbox>), Refusal> {
        let kept = self
            .agents
            .get_mut(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let spec = kept.args.spec.clone();
        let session = kept
            .sessions
            .get_mut(id)
            .ok_or_else(|| Refusal::NoSession(id.to_owned()))?;
        if session.running {
            return Err(Refusal::Busy(id.to_owned()));
        }
        session.running = true;
        Ok((spec, session.state.clone(), session.sandbox.clone()))
    }

    /// End the turn running in the session of `state`'s id of the agent `agent`, keeping
    /// `state` in place of the one there. A session removed while its turn ran stays removed.
    pub fn end_turn(&mut self, agent: &str, state: AgentState) {
        if let Ok(session) = self.session_mut(agent, &state.id) {
            session.state = state;
            session.running = false;
        }
    }

    /// Every session of the agent `agent`, in the order each was started.
    pub fn list_sessions(&self, agent: &str) -> Result<Vec<SessionListOutputItem>, Refusal> {
        let kept = self
            .agents
            .get(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        let sessions = kept.sessions.values();
        Ok(sessions
            .map(|session| SessionListOutputItem {
                id: session.state.id.clone(),
                message_count: session.state.history.len(),
            })
            .collect())
    }

    /// Keep the session `id` of the agent `agent` no longer; the sandbox made for it is answered,
    /// for the caller to remove. Nothing happens when there is none.
    pub fn remove_session(&mut self, agent: &str, id: &str) -> Option<virtx::Sandbox> {
        let kept = self.agents.get_mut(agent)?;
        kept.sessions.shift_remove(id)?.sandbox
    }

    fn session_mut(&mut self, agent: &str, id: &str) -> Result<&mut Session, Refusal> {
        let kept = self
            .agents
            .get_mut(agent)
            .ok_or_else(|| Refusal::NoAgent(agent.to_owned()))?;
        kept.sessions
            .get_mut(id)
            .ok_or_else(|| Refusal::NoSession(id.to_owned()))
    }
}
