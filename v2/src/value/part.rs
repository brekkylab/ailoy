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

    pub fn is_reasoning(&self) -> bool {
        match self {
            Self::TextReasoning { .. } => true,
            _ => false,
        }
    }

    pub fn is_content(&self) -> bool {
        match self {
            Self::TextContent { .. } => true,
            _ => false,
        }
    }

    pub fn is_tool_call(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::TextReasoning { .. } | Self::TextContent(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextReasoning { text, .. } => Some(text.as_str()),
            Self::TextContent(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::TextReasoning { text, .. } => Some(text),
            Self::TextContent(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &Value)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((id.as_ref().map(|x| x.as_str()), name.as_str(), arguments)),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(Option<&mut String>, &mut String, &mut Value)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((id.as_mut(), name, arguments)),
            _ => None,
        }
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
        id: Option<String>,
        name: String,
        arguments: String,
    },
    TextRefusal(String),
}

impl PartDelta {
    pub fn is_reasoning(&self) -> bool {
        match self {
            Self::TextReasoning { .. } => true,
            _ => false,
        }
    }

    pub fn is_content(&self) -> bool {
        match self {
            Self::TextContent { .. } => true,
            _ => false,
        }
    }

    pub fn is_tool_call(&self) -> bool {
        match self {
            Self::TextToolCall { .. } | Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn is_text(&self) -> bool {
        match self {
            Self::TextReasoning { .. } | Self::TextContent(..) | Self::TextToolCall(..) => true,
            _ => false,
        }
    }

    pub fn is_function(&self) -> bool {
        match self {
            Self::FunctionToolCall { .. } => true,
            _ => false,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::TextReasoning { text, .. } => Some(text.as_str()),
            Self::TextContent(text) => Some(text.as_str()),
            Self::TextToolCall(text) => Some(text.as_str()),
            _ => None,
        }
    }

    pub fn as_text_mut(&mut self) -> Option<&mut String> {
        match self {
            Self::TextReasoning { text, .. } => Some(text),
            Self::TextContent(text) => Some(text),
            Self::TextToolCall(text) => Some(text),
            _ => None,
        }
    }

    pub fn as_function(&self) -> Option<(Option<&str>, &str, &str)> {
        match self {
            Self::FunctionToolCall {
                id,
                name,
                arguments,
            } => Some((
                id.as_ref().map(|x| x.as_str()),
                name.as_str(),
                arguments.as_str(),
            )),
            _ => None,
        }
    }

    pub fn as_function_mut(&mut self) -> Option<(&mut String, &mut String)> {
        match self {
            Self::FunctionToolCall {
                name, arguments, ..
            } => Some((name, arguments)),
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
                    id: lid,
                    name: mut lname,
                    arguments: mut largs,
                },
                PartDelta::FunctionToolCall {
                    id: rid,
                    name: rname,
                    arguments: rargs,
                },
            ) => {
                let id = match (lid, rid) {
                    (None, None) => None,
                    (None, Some(rid)) => Some(rid),
                    (Some(lid), None) => Some(lid),
                    (Some(lid), Some(rid)) if lid == rid => Some(lid),
                    (Some(_), Some(_)) => return Err(()),
                };
                lname.push_str(&rname);
                largs.push_str(&rargs);
                Ok(PartDelta::FunctionToolCall {
                    id,
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
                        (Some(name), Some(args)) => Ok(Part::function_tool_call(
                            name,
                            serde_json::from_str::<Value>(&args)
                                .map_err(|_| String::from("Invalid JSON"))?,
                        )),
                        _ => Err(String::from("Invalid function JSON")),
                    }
                }
                Err(_) => Err(String::from("Invalid JSON")),
            },
            PartDelta::FunctionToolCall {
                id,
                name,
                arguments,
            } => Ok(Part::FunctionToolCall {
                id,
                name,
                arguments: serde_json::from_str::<Value>(&arguments)
                    .map_err(|_| String::from("Invalid JSON"))?,
            }),
            PartDelta::TextRefusal(text) => Ok(Part::text_refusal(text)),
        }
    }
}
