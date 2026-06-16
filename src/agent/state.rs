use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    message::Message,
    runenv::{Local, Machine, MachineDyn},
};

pub struct AgentState {
    pub history: Vec<Message>,

    /// Shared machine handle. Defaults to [`Local`] wrapped in `Arc<Mutex<>>`.
    /// Sub-agents inherit this via `Arc::clone` so they share the same VM.
    pub runenv: Arc<Mutex<dyn MachineDyn>>,

    /// Token count from the most recent model API call; used to decide when to truncate history.
    pub last_input_tokens: Option<u64>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            runenv: Arc::new(Mutex::new(Local::new())),
            last_input_tokens: None,
        }
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Replace the shared machine. Accepts any `Arc<Mutex<M>>` where `M: Machine`
    /// and stores it erased as `Arc<Mutex<dyn MachineDyn>>` via unsizing coercion,
    /// so callers can keep passing the concrete handle they already hold.
    pub fn with_runenv<M: Machine>(mut self, runenv: Arc<Mutex<M>>) -> Self {
        self.runenv = runenv;
        self
    }
}
