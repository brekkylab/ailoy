use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::message::Message;

/// What an agent carries from one turn to the next, as data: a session of an agent a server
/// keeps is one of these, kept there with it.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AgentState {
    /// What this state is known by, a UUID unless given another.
    pub id: String,

    pub history: Vec<Message>,

    /// The context's size in tokens as of the last model call, its input as the model
    /// reported it, which a [`TruncationPolicy`](super::TruncationPolicy) checks against its limit as a turn begins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,

    /// The id of the sandbox the agent works in, on the virtx server where it runs; none, for an
    /// agent without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_id: Option<String>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            history: Vec::new(),
            context_tokens: None,
            sandbox_id: None,
        }
    }
}

impl AgentState {
    /// A new state, with an empty history, a new id and no sandbox.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }
}

/// What starting a session takes: `POST /agents/{name}/sessions/`'s body.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct SessionCreateArgs {
    /// The history the session's state starts with; empty, unless given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Message>,

    /// What the server makes the session's sandbox of, which goes with the session. A session of
    /// an agent whose spec uses a sandbox needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<SessionSandbox>,
}

impl SessionCreateArgs {
    /// A session that starts with an empty history.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history.extend(history);
        self
    }

    pub fn sandbox(mut self, sandbox: SessionSandbox) -> Self {
        self.sandbox = Some(sandbox);
        self
    }
}

/// What a session's sandbox is made of, on the server's virtx: `{"package": "<id>"}` or
/// `{"create": <args>}`.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionSandbox {
    /// The virtx package of this id, which virtx counts the sandbox under.
    Package(String),
    /// These args, as virtx's `POST /sandboxes/` takes them.
    Create(#[schemars(with = "serde_json::Value")] virtx::SandboxCreateArgs),
}

/// What an app starting a session takes: `POST /sessions/`'s body.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppSessionCreateArgs {
    /// The agent the session is of, by the name it is kept under.
    pub agent: String,

    /// The history the session's state starts with; empty, unless given.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Message>,

    /// The id of the virtx package the session's sandbox is made of, as the app's own
    /// credentials let it: virtx refuses an app its package does not let in, or one past its
    /// limits. A session of an agent whose spec uses a sandbox needs one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
}

impl AppSessionCreateArgs {
    /// A session of the agent `agent` that starts with an empty history.
    pub fn new(agent: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            history: Vec::new(),
            package: None,
        }
    }

    pub fn history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history.extend(history);
        self
    }

    pub fn package(mut self, id: impl Into<String>) -> Self {
        self.package = Some(id.into());
        self
    }
}

/// A session an app started: `POST /sessions/`'s answer.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppSessionCreateOutput {
    /// Its state, whose id, made by the server, the session is known by.
    pub session: AgentState,

    /// The token that reaches this session alone, sent as `Authorization: Bearer <grant>`.
    pub grant: String,
}

/// A session of an agent, as the server lists it.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SessionListOutputItem {
    /// Its state's id.
    pub id: String,

    /// How many messages its history has.
    pub message_count: usize,
}
