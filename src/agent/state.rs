use std::sync::Arc;

use crate::{browser::BrowserSession, console::Sandbox, memory::Memory, message::Message};

/// What an [`Agent`](super::Agent) carries from one turn to the next.
#[derive(Default)]
pub struct AgentState {
    pub history: Vec<Message>,

    /// The sandbox this agent's sandbox tools run in.
    ///
    /// With `None`, pure tools still run and sandbox tools fail with an error; nothing
    /// fills this in, since making a sandbox means choosing a virtx server.
    pub sandbox: Option<Sandbox>,

    /// The memory store this agent remembers into; `None` means no memory tools.
    pub memory: Option<Memory>,

    /// Input tokens of the last model call, which a
    /// [`ContextManager`](super::ContextManager) checks against its limit as a turn begins.
    pub last_input_tokens: Option<u64>,

    /// The browser this agent drives; `None` means no browser tools.
    pub(crate) browser: Option<Arc<BrowserSession>>,
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_history(mut self, history: impl IntoIterator<Item = Message>) -> Self {
        self.history = history.into_iter().collect();
        self
    }

    /// Run sandbox tools in `sandbox`.
    pub fn with_sandbox(mut self, sandbox: Sandbox) -> Self {
        self.sandbox = Some(sandbox);
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

    pub(crate) fn with_browser(mut self, browser: Arc<BrowserSession>) -> Self {
        self.browser = Some(browser);
        self
    }
}
