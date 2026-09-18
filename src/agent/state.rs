use std::sync::Arc;

use cortex::console::Console;
use tokio::sync::Mutex;

use crate::{memory::Memory, message::Message};

pub struct AgentState {
    pub history: Vec<Message>,

    /// Where this agent's console tools run, if it has one.
    ///
    /// `None` is an agent with no console: its pure tools still run, and a tool that
    /// needs one fails saying so. Building a console means choosing a console server
    /// to start, which is the caller's decision — so nothing here ever fills this in.
    ///
    /// Supplied already started. `start` boots the backend and the server re-boots on
    /// a second one, so it is owed exactly once, by whoever built the console.
    ///
    /// `Arc<Mutex<..>>` because tool execution hands out `'static` streams that each
    /// carry a handle, and because every `Console` method takes `&mut self` — the
    /// protocol has one outstanding request at a time, and the lock is how that is
    /// spelled across concurrent tool futures.
    pub console: Arc<Mutex<Option<Console>>>,

    /// The memory store this agent remembers into, if it has one.
    ///
    /// `None` is an agent that does not remember: it is given no memory tools, and
    /// nothing is written down when the conversation ends. Which store — and whether
    /// there is one at all — is the caller's decision, the same way the console is, so
    /// nothing here fills this in either.
    ///
    /// A [`Memory`] is a path and not a handle, so this is cheap to clone into the
    /// memory tools at construction and safe to share with a sub-agent that should
    /// remember into the same store.
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

    /// Run console tools in `console`, which must already be started.
    pub fn with_console(mut self, console: Console) -> Self {
        self.console = Arc::new(Mutex::new(Some(console)));
        self
    }

    /// Share an existing slot — the way a sub-agent is put in its parent's console
    /// rather than given one of its own.
    pub fn with_console_slot(mut self, console: Arc<Mutex<Option<Console>>>) -> Self {
        self.console = console;
        self
    }

    /// Remember into `memory`, which brings the `mem_search` and `mem_insert` tools with
    /// it — see [`Agent::try_with_provider_and_state`](crate::agent::Agent::try_with_provider_and_state).
    ///
    /// The store is expected to exist already: `mem init` makes one, and an agent that
    /// was pointed at a file that is not there hears so from its first memory tool call.
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }
}
