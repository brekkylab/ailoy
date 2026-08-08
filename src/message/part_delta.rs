use std::fmt;

use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{Delta, Part, PartFunction, PartImage},
};

/// Represents an incremental update (delta) of a function part.
///
/// This type is used during streaming or partial message generation, when function calls are being streamed as text chunks or partial JSON fragments.
///
/// # Variants
/// * `Verbatim(String)` — Raw text content, typically a partial JSON fragment.
/// * `WithStringArgs { name, arguments }` — Function name and its serialized arguments as strings.
/// * `WithParsedArgs { name, arguments }` — Function name and parsed arguments as a `Value`.
///
/// # Use Case
/// When the model streams out a function call response (e.g., `"function_call":{"name":...}`),
/// the incremental deltas can be accumulated until the full function payload is formed.
///
/// # Example
/// ```rust
/// # use ailoy::message::PartDeltaFunction;
/// let delta = PartDeltaFunction::WithStringArgs {
///     name: "translate".into(),
///     arguments: r#"{"text":"hi"}"#.into(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartDeltaFunction {
    Verbatim { text: String },
    WithStringArgs { name: String, arguments: String },
    WithParsedArgs { name: String, arguments: Value },
}

/// Represents a partial or incremental update (delta) of a [`Part`].
///
/// This type enables composable, streaming updates to message parts.
/// For example, text may be produced token-by-token, or a function call
/// may be emitted gradually as its arguments stream in.
///
/// # Example
///
/// ## Rust
/// ```rust
/// # use ailoy::message::{PartDelta, Delta};
/// let d1 = PartDelta::Text { text: "Hel".into() };
/// let d2 = PartDelta::Text { text: "lo".into() };
/// let merged = d1.accumulate(d2).unwrap();
/// assert_eq!(merged.to_text().unwrap(), "Hello");
/// ```
///
/// # Error Handling
/// Accumulation or finalization may return an error if incompatible deltas
/// (e.g. mismatched function IDs) are combined or invalid JSON arguments are given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartDelta {
    /// Incremental text fragment.
    Text { text: String },

    /// Incremental function call fragment.
    Function {
        id: Option<String>,
        function: PartDeltaFunction,
    },

    /// JSON-like value update.
    Value { value: Value },

    /// A complete image part. Images are not produced incrementally (a tool
    /// result may carry one whole), so this delta always holds a finished image.
    Image { image: PartImage },

    /// Placeholder representing no data yet.
    Null {},
}

impl PartDelta {
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }
    pub fn is_verbatim_function(&self) -> bool {
        matches!(
            self,
            Self::Function {
                function: PartDeltaFunction::Verbatim { .. },
                ..
            }
        )
    }

    pub fn is_function(&self) -> bool {
        matches!(
            self,
            Self::Function {
                function: PartDeltaFunction::WithStringArgs { .. },
                ..
            }
        )
    }

    pub fn is_parsed_function(&self) -> bool {
        matches!(
            self,
            Self::Function {
                function: PartDeltaFunction::WithParsedArgs { .. },
                ..
            }
        )
    }

    pub fn is_value(&self) -> bool {
        matches!(self, Self::Value { .. })
    }

    pub fn to_text(self) -> Option<String> {
        match self {
            Self::Text { text } => Some(text),
            Self::Function {
                function: PartDeltaFunction::Verbatim { text },
                ..
            } => Some(text),
            _ => None,
        }
    }

    pub fn to_function(self) -> Option<(Option<String>, String, String)> {
        match self {
            Self::Function {
                id,
                function: PartDeltaFunction::WithStringArgs { name, arguments },
            } => Some((id, name, arguments)),
            _ => None,
        }
    }

    pub fn to_parsed_function(self) -> Option<(Option<String>, String, Value)> {
        match self {
            Self::Function {
                id,
                function: PartDeltaFunction::WithParsedArgs { name, arguments },
            } => Some((id, name, arguments)),
            Self::Function {
                id,
                function: PartDeltaFunction::WithStringArgs { name, arguments },
            } => {
                // A no-arg tool call streams empty arguments (e.g. Anthropic
                // sends a single `partial_json: ""`); treat blank as an empty
                // object. Non-empty but malformed JSON stays `None` so callers
                // can surface it as an error instead of guessing.
                let arguments = if arguments.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str::<Value>(&arguments).ok()?
                };
                Some((id, name, arguments))
            }
            Self::Function {
                id,
                function: PartDeltaFunction::Verbatim { text },
            } => serde_json::from_str::<Value>(&text).ok().and_then(|root| {
                Some((
                    id,
                    root.pointer_as::<str>("/name")?.to_owned(),
                    root.pointer("/arguments")?.to_owned(),
                ))
            }),
            _ => None,
        }
    }

    pub fn to_value(self) -> Option<Value> {
        match self {
            Self::Value { value } => Some(value),
            _ => None,
        }
    }
}

impl Default for PartDelta {
    fn default() -> Self {
        Self::Null {}
    }
}

impl From<Part> for PartDelta {
    /// Wraps a finalized [`Part`] as a complete delta (used when a tool result,
    /// which is produced whole, is surfaced on the streaming path). A function's
    /// already-parsed arguments map to [`PartDeltaFunction::WithParsedArgs`].
    fn from(part: Part) -> Self {
        match part {
            Part::Text { text } => PartDelta::Text { text },
            Part::Function {
                id,
                function: PartFunction { name, arguments },
            } => PartDelta::Function {
                id: Some(id),
                function: PartDeltaFunction::WithParsedArgs { name, arguments },
            },
            Part::Value { value } => PartDelta::Value { value },
            Part::Image { image } => PartDelta::Image { image },
        }
    }
}

impl Delta for PartDelta {
    type Item = Part;
    type Err = anyhow::Error; // TODO: Define custom error for this.

    fn accumulate(self, other: Self) -> anyhow::Result<Self> {
        match (self, other) {
            (PartDelta::Null {}, other) => Ok(other),
            (PartDelta::Text { text: mut t1 }, PartDelta::Text { text: t2 }) => {
                t1.push_str(&t2);
                Ok(PartDelta::Text { text: t1 })
            }
            (
                PartDelta::Function {
                    id: id1,
                    function: f1,
                },
                PartDelta::Function {
                    id: id2,
                    function: f2,
                },
            ) => {
                let id = match (id1, id2) {
                    (Some(id1), Some(id2)) => {
                        if id1 != id2 {
                            bail!(
                                "Cannot accumulate two functions with different ids. ({} != {}).",
                                id1,
                                id2
                            )
                        }
                        Some(id1)
                    }
                    (None, Some(id2)) => Some(id2),
                    (Some(id1), None) => Some(id1),
                    (None, None) => None,
                };
                let f = match (f1, f2) {
                    (
                        PartDeltaFunction::Verbatim { text: mut t1 },
                        PartDeltaFunction::Verbatim { text: t2 },
                    ) => {
                        t1.push_str(&t2);
                        PartDeltaFunction::Verbatim { text: t1 }
                    }
                    (
                        PartDeltaFunction::WithStringArgs {
                            name: mut n1,
                            arguments: mut a1,
                        },
                        PartDeltaFunction::WithStringArgs {
                            name: n2,
                            arguments: a2,
                        },
                    ) => {
                        n1.push_str(&n2);
                        a1.push_str(&a2);
                        PartDeltaFunction::WithStringArgs {
                            name: n1,
                            arguments: a1,
                        }
                    }
                    (
                        PartDeltaFunction::WithParsedArgs {
                            name: mut n1,
                            arguments: _,
                        },
                        PartDeltaFunction::WithParsedArgs {
                            name: n2,
                            arguments: a2,
                        },
                    ) => {
                        // @jhlee: Rather than just replacing, merge logic could be helpful
                        n1.push_str(&n2);
                        PartDeltaFunction::WithParsedArgs {
                            name: n1,
                            arguments: a2,
                        }
                    }
                    (f1, f2) => bail!(
                        "Accumulation between those two function delta {:?}, {:?} is not defined.",
                        f1,
                        f2
                    ),
                };
                Ok(PartDelta::Function { id, function: f })
            }
            (pd1, pd2) => {
                bail!(
                    "Accumulation between those two part delta {:?}, {:?} is not defined.",
                    pd1,
                    pd2
                )
            }
        }
    }

    fn finish(self) -> anyhow::Result<Self::Item> {
        match &self {
            PartDelta::Null {} => Ok(Part::Text {
                text: String::new(),
            }),
            PartDelta::Text { text } => Ok(Part::Text {
                text: text.to_owned(),
            }),
            PartDelta::Function { .. } => {
                // Malformed (non-empty, non-JSON) arguments — e.g. a tool call
                // truncated mid-stream by a token limit — return an error rather
                // than panic, so the caller can handle it (the agent loop rolls
                // the turn back).
                let Some((id, name, arguments)) = self.to_parsed_function() else {
                    bail!("tool-call arguments are not valid JSON");
                };
                let id = id.unwrap_or("fn".into());
                Ok(Part::Function {
                    id,
                    function: PartFunction { name, arguments },
                })
            }
            PartDelta::Value { value } => Ok(Part::Value {
                value: value.to_owned(),
            }),
            PartDelta::Image { image } => Ok(Part::Image {
                image: image.to_owned(),
            }),
        }
    }
}

impl fmt::Display for PartDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "PartDelta {}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A no-arg tool call streams empty accumulated arguments (Anthropic sends a
    /// single `partial_json: ""`). finish() must yield an empty-object call, not
    /// panic on `serde_json::from_str("")`.
    #[test]
    fn test_finish_function_empty_args_is_empty_object() {
        let delta = PartDelta::Function {
            id: Some("get_time/call_1".into()),
            function: PartDeltaFunction::WithStringArgs {
                name: "get_time".into(),
                arguments: String::new(),
            },
        };
        let part = delta.finish().expect("empty args must finish as {}");
        let (_, name, args) = part.as_function().expect("expected a function part");
        assert_eq!(name, "get_time");
        assert!(args.is_object());
        assert_eq!(args.as_object().map(|o| o.len()), Some(0));
    }

    /// Non-empty but malformed arguments (e.g. a call truncated mid-stream by a
    /// token limit) must return an error, not panic, so the caller can roll back.
    #[test]
    fn test_finish_function_malformed_args_errors() {
        let delta = PartDelta::Function {
            id: None,
            function: PartDeltaFunction::WithStringArgs {
                name: "get_weather".into(),
                arguments: "{\"location\":".into(), // truncated JSON
            },
        };
        assert!(
            delta.finish().is_err(),
            "malformed args must error, not panic"
        );
    }
}
