use std::sync::Arc;

use cortex::console::ConsoleClient;
use tokio::sync::Mutex;

use crate::{memory::Memory, message::Message};

pub struct AgentState {
    pub history: Vec<Message>,

    /// Where this agent's console tools run.
    ///
    /// With `None`, pure tools still run and console tools fail with an error; nothing
    /// fills this in, since building a console means choosing a console server.
    ///
    /// Need not be started: the first command that needs a booted session boots it.
    ///
    /// `Arc` because tool execution hands out `'static` streams that each carry a handle;
    /// `Mutex` because the protocol allows one outstanding request at a time (every
    /// `ConsoleClient` method takes `&mut self`).
    pub console: Arc<Mutex<Option<ConsoleClient>>>,

    /// The memory store this agent remembers into; `None` means no memory tools.
    ///
    /// A [`Memory`] is a path, not a handle, so it is cheap to clone into the memory tools
    /// and safe to share with a sub-agent.
    pub memory: Option<Memory>,

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
            console: Arc::new(Mutex::new(None)),
            memory: None,
            last_input_tokens: None,
        }
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

    /// Share an existing console slot, e.g. a sub-agent using its parent's.
    pub fn with_console_slot(mut self, console: Arc<Mutex<Option<ConsoleClient>>>) -> Self {
        self.console = console;
        self
    }

    /// Remember into `memory`, which adds the `mem_search` and `mem_insert` tools.
    ///
    /// The store must already exist (`mem init`); a missing file surfaces on the first
    /// memory tool call.
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }
}
