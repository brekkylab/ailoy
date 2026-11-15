use std::sync::Arc;

use crate::{
    browser::{Browser, BrowserSession},
    console::Sandbox,
    message::Message,
};

/// What an [`Agent`](super::Agent) carries from one turn to the next: its history, and the
/// sandbox and browser it works with.
///
/// A clone shares the sandbox and the browser: they are handles, so two agents built from
/// clones work in the same sandbox and drive the same browser.
#[derive(Clone)]
pub struct AgentState {
    /// What this state is known by, a UUID unless given another.
    pub id: String,

    pub history: Vec<Message>,

    /// The context's size in tokens as of the last model call, its input as the model
    /// reported it, which a [`TruncationPolicy`](super::TruncationPolicy) checks against its
    /// limit as a turn begins.
    pub context_tokens: Option<u64>,

    /// The sandbox this agent's sandbox tools run in.
    ///
    /// Nothing fills this in, since making a sandbox means choosing a virtx server.
    pub sandbox: Option<Sandbox>,

    /// The browser this agent's `browser_*` tools drive.
    pub(crate) browser: Option<Arc<BrowserSession>>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            history: Vec::new(),
            context_tokens: None,
            sandbox: None,
            browser: None,
        }
    }
}

impl AgentState {
    /// A new state, with an empty history, a new id, and no sandbox or browser.
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

    /// Drive `browser`, which is not started until the agent first uses it.
    pub fn with_browser(mut self, browser: Browser) -> Self {
        self.browser = Some(BrowserSession::new(browser));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_states_differ() {
        assert_ne!(AgentState::new().id, AgentState::new().id);
    }
}
