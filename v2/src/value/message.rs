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

#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            parts: Vec::new(),
        }
    }

    pub fn with_parts(role: Role, parts: impl IntoIterator<Item = impl Into<Part>>) -> Self {
        Self {
            role,
            parts: parts.into_iter().map(|v| v.into()).collect(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MessageDelta {
    pub role: Option<Role>,
    pub parts: Vec<PartDelta>,
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
            mut parts,
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

        // Merge parts
        for part_new in other.parts {
            if let Some(last) = parts.last() {
                // Aggregate parts if same type
                match (last, &part_new) {
                    (PartDelta::TextReasoning { .. }, PartDelta::TextReasoning { .. })
                    | (PartDelta::TextContent { .. }, PartDelta::TextContent { .. }) => {
                        let part_updated = parts.pop().unwrap().aggregate(part_new)?;
                        parts.push(part_updated);
                    }
                    (
                        PartDelta::FunctionToolCall { id: id1, .. },
                        PartDelta::FunctionToolCall { id: id2, .. },
                    ) => {
                        match (id1, id2) {
                            // If ID is different, push as a new pert
                            (Some(lhs), Some(rhs)) if lhs != rhs => {
                                parts.push(part_new);
                            }
                            // Otherwise aggregate
                            _ => {
                                let part_updated = parts.pop().unwrap().aggregate(part_new)?;
                                parts.push(part_updated);
                            }
                        };
                    }
                    _ => {
                        parts.push(part_new);
                    }
                }
            } else {
                parts.push(part_new);
            }
        }

        // Return
        Ok(Self { role, parts })
    }

    fn finish(self) -> Result<Self::Item, String> {
        let Some(role) = self.role else {
            return Err("Role not specified".to_owned());
        };
        Ok(Message {
            role,
            parts: self
                .parts
                .into_iter()
                .map(|v| v.finish().unwrap())
                .collect(),
        })
    }
}
