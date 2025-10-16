use anyhow::bail;
use serde::{Deserialize, Serialize};

use crate::value::{Delta, Part, PartDelta};

/// The author of a message (or streaming delta) in a chat.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, strum::Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
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
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct Message {
    pub role: Role,

    pub contents: Vec<Part>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "nodejs", napi_derive::napi(js_name = "tool_calls"))]
    pub tool_calls: Option<Vec<Part>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            contents: Vec::new(),
            id: None,
            thinking: None,
            tool_calls: None,
            signature: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn with_thinking_signature(
        mut self,
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.thinking = Some(thinking.into());
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
        self.tool_calls = Some(tool_calls.into_iter().map(|v| v.into()).collect());
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct MessageDelta {
    pub role: Option<Role>,
    pub id: Option<String>,
    pub thinking: Option<String>,
    pub contents: Vec<PartDelta>,
    #[cfg_attr(feature = "nodejs", napi_derive::napi(js_name = "tool_calls"))]
    pub tool_calls: Vec<PartDelta>,
    pub signature: Option<String>,
}

impl MessageDelta {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_role(mut self, role: impl Into<Role>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_thinking(mut self, thinking: impl Into<String>) -> Self {
        self.thinking = Some(thinking.into());
        self
    }

    pub fn with_thinking_signature(
        mut self,
        thinking: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        self.thinking = Some(thinking.into());
        self.signature = Some(signature.into());
        self
    }

    pub fn with_contents(
        mut self,
        contents: impl IntoIterator<Item = impl Into<PartDelta>>,
    ) -> Self {
        self.contents = contents.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn with_tool_calls(
        mut self,
        tool_calls: impl IntoIterator<Item = impl Into<PartDelta>>,
    ) -> Self {
        self.tool_calls = tool_calls.into_iter().map(|v| v.into()).collect();
        self
    }

    pub fn to_message(self) -> anyhow::Result<Message> {
        self.finish()
    }
}

impl Delta for MessageDelta {
    type Item = Message;
    type Err = anyhow::Error; // TODO: Define custom error for this.

    fn aggregate(self, other: Self) -> anyhow::Result<Self> {
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
                bail!(
                    "Cannot aggregate two message deltas with differenct roles. ({} != {})",
                    lhs,
                    rhs
                );
            };
        } else if let Some(rhs) = other.role {
            role = Some(rhs);
        };

        // Merge ID
        if let Some(id_incoming) = other.id {
            id = Some(id_incoming);
        }

        // Merge think
        if let Some(thinking_rhs) = other.thinking {
            if let Some(mut thinking_lhs) = thinking {
                thinking_lhs.push_str(&thinking_rhs);
                thinking = Some(thinking_lhs);
            } else {
                thinking = Some(thinking_rhs);
            }
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

    fn finish(self) -> anyhow::Result<Self::Item> {
        let Self {
            role,
            id,
            thinking,
            mut contents,
            mut tool_calls,
            signature,
        } = self;

        let Some(role) = role else {
            bail!("Role not specified")
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
            Some(tool_calls_new)
        };
        Ok(Message {
            role,
            contents,
            id,
            thinking,
            tool_calls,
            signature,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(
    feature = "nodejs",
    napi_derive::napi(discriminant_case = "snake_case")
)]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum FinishReason {
    Stop {},
    Length {}, // max_output_tokens
    ToolCall {},
    Refusal { reason: String }, // content_filter, refusal
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct MessageOutput {
    pub delta: MessageDelta,
    #[cfg_attr(feature = "nodejs", napi_derive::napi(js_name = "finish_reason"))]
    pub finish_reason: Option<FinishReason>,
}

impl MessageOutput {
    pub fn new() -> Self {
        Self {
            delta: MessageDelta::new(),
            finish_reason: None,
        }
    }
}
