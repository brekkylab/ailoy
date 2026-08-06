use std::sync::Arc;

use crate::{
    message::Message,
    runenv::{Console, LocalConsole},
};

pub struct AgentState {
    pub history: Vec<Message>,

    /// Shared exec backend. Defaults to [`LocalConsole`] (host). Sub-agents
    /// inherit this via `Arc::clone`. `Console::exec` takes `&self`, so no lock
    /// is needed — implementations synchronize internally as required.
    pub runenv: Arc<dyn Console>,

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
            runenv: Arc::new(LocalConsole::new()),
            last_input_tokens: None,
        }
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Replace the exec backend with any `Arc<dyn Console>` (host, krun sandbox, …).
    pub fn with_runenv(mut self, runenv: Arc<dyn Console>) -> Self {
        self.runenv = runenv;
        self
    }
}
