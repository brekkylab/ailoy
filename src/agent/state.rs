use crate::{
    message::Message,
    runenv::{Local, SharedMachine},
};

pub struct AgentState {
    pub history: Vec<Message>,

    /// Shared machine handle. Defaults to [`Local`] wrapped in `Arc<Mutex<>>`.
    /// Sub-agents inherit this via `Arc::clone` so they share the same VM.
    pub machine: SharedMachine,

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
            machine: SharedMachine::new(Local::new()),
            last_input_tokens: None,
        }
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    pub fn with_runenv(mut self, machine: SharedMachine) -> Self {
        self.machine = machine;
        self
    }
}
