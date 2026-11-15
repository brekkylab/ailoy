use serde::{Deserialize, Serialize};

use crate::message::Message;

/// What an [`Agent`](super::Agent) carries from one turn to the next, as data to keep and
/// build the agent from again. The sandbox and the browser it works with are handed to it
/// beside the state; a sandbox's files stay on its virtx server.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    /// What this state is known by, a UUID unless given another.
    pub id: String,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Message>,

    /// The context's size in tokens as of the last model call, its input as the model
    /// reported it, which a [`TruncationPolicy`](super::TruncationPolicy) checks against its
    /// limit as a turn begins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_tokens: Option<u64>,

    /// The virtx snapshot holding what this state's work wrote in its sandbox, for the
    /// sandbox to be made again on. Nothing here makes or reads it yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<String>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            history: Vec::new(),
            context_tokens: None,
            snapshot: None,
        }
    }
}

impl AgentState {
    /// A new state, with an empty history and a new id.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Hold `snapshot`, the virtx snapshot of this state's sandbox changes.
    pub fn with_snapshot(mut self, snapshot: impl Into<String>) -> Self {
        self.snapshot = Some(snapshot.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Part, Role};

    #[test]
    fn test_a_state_round_trips_and_new_ones_differ() {
        let state = AgentState::new()
            .with_history([Message::new(Role::User).with_contents([Part::text("hi")])])
            .with_snapshot("work");
        let back: AgentState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(back.id, state.id);
        assert_eq!(back.history.len(), 1);
        assert_eq!(back.snapshot.as_deref(), Some("work"));
        assert_ne!(AgentState::new().id, state.id);
    }
}
