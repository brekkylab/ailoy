use crate::value::{Value, delta::Delta};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Part {
    TextReasoning {
        text: String,
        signature: Option<String>,
    },
    TextContent(String),
    ImageContent(Vec<u8>),
    FunctionToolCall {
        id: Option<String>,
        name: String,
        arguments: Value,
    },
    TextRefusal(String),
}

impl Part {
    pub fn text_reasoning(v: impl Into<String>) -> Self {
        Self::TextReasoning {
            text: v.into(),
            signature: None,
        }
    }

    pub fn text_reasoning_with_signature(v: impl Into<String>, sig: impl Into<String>) -> Self {
        Self::TextReasoning {
            text: v.into(),
            signature: Some(sig.into()),
        }
    }

    pub fn text_content(v: impl Into<String>) -> Self {
        Self::TextContent(v.into())
    }

    pub fn image_content(v: impl IntoIterator<Item = u8>) -> Self {
        Self::ImageContent(v.into_iter().collect())
    }

    pub fn function_tool_call(name: impl Into<String>, args: impl Into<Value>) -> Self {
        Self::FunctionToolCall {
            id: None,
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn function_tool_call_with_id(
        name: impl Into<String>,
        args: impl Into<Value>,
        id: impl Into<String>,
    ) -> Self {
        Self::FunctionToolCall {
            id: Some(id.into()),
            name: name.into(),
            arguments: args.into(),
        }
    }

    pub fn text_refusal(v: impl Into<String>) -> Self {
        Self::TextRefusal(v.into())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartDelta {
    Null,
    TextReasoning {
        text: String,
        signature: Option<String>,
    },
    TextContent(String),
    TextToolCall(String),
    FunctionToolCall {
        name: String,
        arguments: String,
    },
}

impl PartDelta {
    pub fn is_reasoning(&self) -> bool {
        match self {
            PartDelta::TextReasoning { .. } => true,
            _ => false,
        }
    }

    pub fn is_content(&self) -> bool {
        match self {
            PartDelta::TextContent { .. } => true,
            _ => false,
        }
    }

    pub fn is_tool_call(&self) -> bool {
        match self {
            PartDelta::TextToolCall { .. } | PartDelta::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            PartDelta::TextReasoning { .. }
            | PartDelta::TextContent(..)
            | PartDelta::TextToolCall(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            PartDelta::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            PartDelta::TextReasoning { text, .. } => Some(text.as_str()),
            PartDelta::TextContent(text) => Some(text.as_str()),
            PartDelta::TextToolCall(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            PartDelta::TextReasoning { text, .. } => Some(text),
            PartDelta::TextContent(text) => Some(text),
            PartDelta::TextToolCall(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(&str, &str)> {
        match self {
            PartDelta::FunctionToolCall { name, arguments } => {
                Some((name.as_str(), arguments.as_str()))
            }
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(&mut String, &mut String)> {
        match self {
            PartDelta::FunctionToolCall { name, arguments } => Some((name, arguments)),
            _ => None,
        }
    }
}

impl Default for PartDelta {
    fn default() -> Self {
        Self::Null
    }
}

impl Delta for PartDelta {
    type Item = Part;

    fn aggregate(self, other: Self) -> Result<Self, ()> {
        match (self, other) {
            (PartDelta::Null, other) => Ok(other),
            (
                PartDelta::TextReasoning {
                    text: mut ltext,
                    signature: lsig,
                },
                PartDelta::TextReasoning {
                    text: rtext,
                    signature: rsig,
                },
            ) => {
                ltext.push_str(&rtext);
                let sig = match (lsig, rsig) {
                    (None, None) => None,
                    (None, Some(rsig)) => Some(rsig),
                    (Some(lsig), None) => Some(lsig),
                    (Some(_), Some(rsig)) => Some(rsig),
                };
                Ok(PartDelta::TextReasoning {
                    text: ltext,
                    signature: sig,
                })
            }
            (PartDelta::TextContent(mut lhs), PartDelta::TextContent(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::TextContent(lhs))
            }
            (PartDelta::TextToolCall(mut lhs), PartDelta::TextToolCall(rhs)) => {
                lhs.push_str(&rhs);
                Ok(PartDelta::TextToolCall(lhs))
            }
            (
                PartDelta::FunctionToolCall {
                    name: mut lname,
                    arguments: mut largs,
                },
                PartDelta::FunctionToolCall {
                    name: rname,
                    arguments: rargs,
                },
            ) => {
                lname.push_str(&rname);
                largs.push_str(&rargs);
                Ok(PartDelta::FunctionToolCall {
                    name: lname,
                    arguments: largs,
                })
            }
            _ => Err(()),
        }
    }

    fn finish(self) -> Result<Self::Item, String> {
        match self {
            PartDelta::Null => Ok(Part::text_content(String::new())),
            PartDelta::TextReasoning { text, signature } => {
                Ok(Part::TextReasoning { text, signature })
            }
            PartDelta::TextContent(s) => Ok(Part::text_content(s)),
            PartDelta::TextToolCall(s) => match serde_json::from_str::<Value>(&s) {
                Ok(root) => {
                    match (
                        root.pointer_as::<str>("/name"),
                        root.pointer_as::<str>("/arguments"),
                    ) {
                        (Some(name), Some(args)) => Ok(Part::function_tool_call(name, args)),
                        _ => Err(String::from("Invalid function JSON")),
                    }
                }
                Err(_) => Err(String::from("Invalid JSON")),
            },
            PartDelta::FunctionToolCall { name, arguments } => {
                Ok(Part::function_tool_call(name, arguments))
            }
        }
    }
}
