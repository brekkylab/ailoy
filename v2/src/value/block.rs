use crate::value::Part;

/// A set of part + some purpose("reasoning"/ "content" / etc...)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Reasoning(String),
    Content(Vec<Part>),
    ToolCalls(Vec<Part>),
    Refusal(String),
}

impl Block {
    pub fn reasoning(reasoning: impl Into<String>) -> Self {
        Self::Reasoning(reasoning.into())
    }

    pub fn content(content: impl IntoIterator<Item = Part>) -> Self {
        Self::Content(content.into_iter().collect())
    }

    pub fn tool_calls(tool_calls: impl IntoIterator<Item = Part>) -> Self {
        Self::ToolCalls(tool_calls.into_iter().collect())
    }

    pub fn refusal(refusal: impl Into<String>) -> Self {
        Self::Refusal(refusal.into())
    }
}
