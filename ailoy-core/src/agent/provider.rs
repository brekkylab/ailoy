use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
};

use crate::{
    agent::{AgentPutArgs, AgentStore, MemoryStore},
    model::{LangModelProvider, ModelKeyProvider},
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
    /// A spec's model, of this org, is to be called with the client's key, and its models are
    /// made with none.
    NoClientKey(String),
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
            Refusal::NoClientKey(org) => {
                write!(f, "the models of '{org}' are not made with a client's key")
            }
            Refusal::Busy(id) => write!(f, "the session '{id}' has a turn running"),
            Refusal::Sandbox(true) => write!(f, "the spec uses a sandbox, and none is given"),
            Refusal::Sandbox(false) => write!(f, "a sandbox is given, and the spec uses none"),
        }
    }
}

impl std::error::Error for Refusal {}

/// What an agent runs with where it runs: the providers of the models and tools a spec names,
/// the virtx server its sessions' sandboxes are made on, and the store the agents put to it and
/// their sessions are kept in. What is not kept, as which session has a turn running, is this
/// process's alone.
pub struct AgentProvider {
    pub tools: ToolProvider,
    pub models: LangModelProvider,
    /// A client of the virtx server, as one of its project's members, that makes the sandboxes
    /// of the sessions the code managing this provider starts.
    pub virtx: Option<virtx::Client>,
    /// The base URL of the virtx server, where an app's session's sandbox is made with the
    /// app's own credentials: none over stdio, which has no apps.
    pub virtx_url: Option<String>,
    /// Where the agents and their sessions are kept: in memory, unless set.
    pub store: Arc<dyn AgentStore>,
    /// The sessions with a turn running, by id.
    running: HashSet<String>,
    /// The sandboxes of the sessions reached since this process began, by session id.
    sandboxes: HashMap<String, virtx::Sandbox>,
}

impl Default for AgentProvider {
    fn default() -> Self {
        AgentProvider {
            tools: ToolProvider::default(),
            models: LangModelProvider::default(),
            virtx: None,
            virtx_url: None,
            store: Arc::new(MemoryStore::default()),
            running: HashSet::new(),
            sandboxes: HashMap::new(),
        }
    }
}

impl AgentProvider {
    pub fn new() -> Self {
        Self::default()
    }

    /// Refused when the agent `args` describes names a model or a tool nothing here provides,
    /// or its model is to be called with the client's key and is made with none.
    pub fn check_agent(&self, args: &AgentPutArgs) -> Result<(), Refusal> {
        let spec = &args.spec;
        let org = (spec.model.split_once('/')).map_or(&*spec.model, |(org, _)| org);
        let Some(set) = self.models.get(&spec.model) else {
            return Err(Refusal::NoModel(org.to_owned()));
        };
        if args.model_key_provider == ModelKeyProvider::Client && !set.takes_key(org) {
            return Err(Refusal::NoClientKey(org.to_owned()));
        }
        if let Some(tool) = spec.tools.iter().find(|t| !self.tools.contains(t)) {
            return Err(Refusal::NoTool(tool.clone()));
        }
        Ok(())
    }

    /// Begin a turn of the session `id`: refused while one runs in it, until
    /// [`end_turn`](Self::end_turn).
    pub fn begin_turn(&mut self, id: &str) -> Result<(), Refusal> {
        match self.running.insert(id.to_owned()) {
            true => Ok(()),
            false => Err(Refusal::Busy(id.to_owned())),
        }
    }

    /// End the turn running in the session `id`.
    pub fn end_turn(&mut self, id: &str) {
        self.running.remove(id);
    }

    /// The sandbox of the session `id`, if it was reached since this process began.
    pub fn sandbox(&self, id: &str) -> Option<virtx::Sandbox> {
        self.sandboxes.get(id).cloned()
    }

    /// Hold `sandbox` as the session `id`'s, for its turns to work in.
    pub fn hold_sandbox(&mut self, id: &str, sandbox: virtx::Sandbox) {
        self.sandboxes.insert(id.to_owned(), sandbox);
    }

    /// Let go of the sandbox of the session `id`, answering it, if it was held.
    pub fn release_sandbox(&mut self, id: &str) -> Option<virtx::Sandbox> {
        self.sandboxes.remove(id)
    }
}
