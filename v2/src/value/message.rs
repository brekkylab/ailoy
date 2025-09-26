use serde::{Deserialize, Serialize};

use crate::value::{Delta, Part, PartDelta};

/// The author of a message (or streaming delta) in a chat.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, strum::Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub id: Option<String>,
    pub thinking: String,
    pub contents: Vec<Part>,
    pub tool_calls: Vec<Part>,
    pub signature: Option<String>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            id: None,
            thinking: String::new(),
            contents: Vec::new(),
            tool_calls: Vec::new(),
            signature: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = thinking.into();
        self
    }

    pub fn with_thinking_signature(
        mut self,
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.thinking = thinking.into();
        self.signature = Some(signature.into());
        self
    }

    pub fn with_contents(mut self, contents: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        self.contents = contents.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn with_tool_calls(
        mut self,
        tool_calls: impl IntoIterator<Item = impl Into<Part>>,
    ) -> Self {
        self.tool_calls = tool_calls.into_iter().map(|v| v.into()).collect();
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct MessageDelta {
    pub role: Option<Role>,
    pub id: Option<String>,
    pub thinking: String,
    pub contents: Vec<PartDelta>,
    pub tool_calls: Vec<PartDelta>,
    pub signature: Option<String>,
}

impl MessageDelta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn to_message(self) -> Result<Message, String> {
        self.finish()
    }
}

impl Delta for MessageDelta {
    type Item = Message;

    fn aggregate(self, other: Self) -> Result<Self, ()> {
        let Self {
            mut role,
            mut id,
            mut thinking,
            mut contents,
            mut tool_calls,
            mut signature,
        } = self;

        // Merge role
        if let Some(lhs) = &role
            && let Some(rhs) = &other.role
        {
            if lhs != rhs {
                return Err(());
            };
        } else if let Some(rhs) = other.role {
            role = Some(rhs);
        };

        // Merge ID
        if let Some(id_incoming) = other.id {
            id = Some(id_incoming);
        }

        // Merge think
        if !other.thinking.is_empty() {
            thinking.push_str(&other.thinking);
        }

        // Merge content
        for part_incoming in other.contents {
            if let Some(part_last) = contents.last() {
                match (part_last, &part_incoming) {
                    (PartDelta::Text { .. }, PartDelta::Text { .. })
                    | (PartDelta::Function { .. }, PartDelta::Function { .. }) => {
                        let v = contents.pop().unwrap().aggregate(part_incoming)?;
                        contents.push(v);
                    }
                    _ => contents.push(part_incoming),
                }
            } else {
                contents.push(part_incoming);
            }
        }

        // Merge tool calls
        for part_incoming in other.tool_calls {
            if let Some(part_last) = tool_calls.last() {
                match (part_last, &part_incoming) {
                    (PartDelta::Text { .. }, PartDelta::Text { .. }) => {
                        let v = tool_calls.pop().unwrap().aggregate(part_incoming)?;
                        tool_calls.push(v);
                    }
                    (PartDelta::Function { id: id1, .. }, PartDelta::Function { id: id2, .. }) => {
                        if let Some(id1) = id1
                            && let Some(id2) = id2
                            && id1 != id2
                        {
                            tool_calls.push(part_incoming);
                        } else {
                            let v = tool_calls.pop().unwrap().aggregate(part_incoming)?;
                            tool_calls.push(v);
                        }
                    }
                    _ => tool_calls.push(part_incoming),
                }
            } else {
                tool_calls.push(part_incoming);
            }
        }

        // Merge signature
        if let Some(sig_incoming) = other.signature {
            signature = Some(sig_incoming);
        }

        // Return
        Ok(Self {
            role,
            thinking,
            id,
            contents,
            tool_calls,
            signature,
        })
    }

    fn finish(self) -> Result<Self::Item, String> {
        let Self {
            role,
            id,
            thinking,
            mut contents,
            mut tool_calls,
            signature,
        } = self;

        let Some(role) = role else {
            return Err("Role not specified".to_owned());
        };
        let contents = {
            let mut contents_new = Vec::with_capacity(contents.len());
            for v in contents.drain(..) {
                contents_new.push(v.finish()?);
            }
            contents_new
        };
        let tool_calls = {
            let mut tool_calls_new = Vec::with_capacity(tool_calls.len());
            for v in tool_calls.drain(..) {
                tool_calls_new.push(v.finish()?);
            }
            tool_calls_new
        };
        Ok(Message {
            role,
            id,
            thinking,
            contents,
            tool_calls,
            signature,
        })
    }
}
