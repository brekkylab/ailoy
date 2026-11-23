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

    /// The id of the sandbox the agent works in, on the virtx server of where the agent runs;
    /// none, for an agent without one.
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
}

/// A session of an agent, as the server lists it.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SessionListOutputItem {
    /// Its state's id.
    pub id: String,

    /// How many messages its history has.
    pub message_count: usize,
}
