use crate::{message::Message, state::Shell};

pub struct AgentState {
    pub history: Vec<Message>,

    pub shell: Option<Shell>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            shell: None,
        }
    }

    pub fn shell(mut self) -> Self {
        self.shell = Some(Shell::new());
        self
    }
}
