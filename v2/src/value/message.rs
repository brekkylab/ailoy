use serde::{Deserialize, Serialize};

use crate::value::Part;

/// The author of a message (or streaming delta) in a chat.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System instructions and constraints provided to the assistant.
    System,
    /// Content authored by the end user.
    User,
    /// Content authored by the assistant/model.
    Assistant,
    /// Outputs produced by external tools/functions
    Tool,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Option<Role>,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for Message {
    fn default() -> Self {
        Self {
            role: None,
            parts: Vec::default(),
        }
    }
}
