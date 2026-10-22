use std::sync::Arc;

use tokio::sync::Mutex;
use virtx::console::ConsoleClient;

use crate::{memory::Memory, message::Message};

/// What an [`Agent`](super::Agent) carries from one turn to the next.
#[derive(Default)]
pub struct AgentState {
    pub history: Vec<Message>,

    /// Where this agent's console tools run.
    ///
    /// With `None`, pure tools still run and console tools fail with an error; nothing
    /// fills this in, since building a console means choosing a console server.
    ///
    /// Need not be started: each tool batch that has a console tool starts it.
    ///
    /// `Arc` because tool calls run as `'static` futures that each carry a handle;
    /// `Mutex` because the protocol allows one outstanding request at a time (every
    /// `ConsoleClient` method takes `&mut self`).
    pub console: Arc<Mutex<Option<ConsoleClient>>>,

    /// What this agent remembers of the user; `None` gives it no memory tools.
    pub memory: Option<Memory>,

    /// Input tokens of the last model call, which a
    /// [`ContextManager`](super::ContextManager) checks against its limit.
    pub last_input_tokens: Option<u64>,
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Run console tools in `console`; it need not be started.
    pub fn with_console(mut self, console: ConsoleClient) -> Self {
        self.console = Arc::new(Mutex::new(Some(console)));
        self
    }

    /// Share `console`'s slot with whoever else holds it, such as another agent.
    pub fn with_console_slot(mut self, console: Arc<Mutex<Option<ConsoleClient>>>) -> Self {
        self.console = console;
        self
    }

    /// Remember into `memory`. See [`AgentBuilder::memory`](super::AgentBuilder::memory).
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }
}
