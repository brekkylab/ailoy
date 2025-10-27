use anyhow::Ok;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{PyRepr, python_to_value, value_to_python},
    value::{
        Delta, Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, Part, PartDelta,
        Role, ToolDesc,
    },
};

#[gen_stub_pymethods]
#[pymethods]
impl Part {
    pub fn __repr__(&self) -> String {
        let s = match &self {
            Part::Text { text } => format!("Text(\"{}\")", text.replace('\n', "\\n")),
            Part::Function { .. } => {
                format!(
                    "Function({})",
                    serde_json::to_string(self).unwrap_or("".to_owned())
                )
            }
            Part::Value { value } => format!(
                "Value({})",
                serde_json::to_string(value).unwrap_or("{...}".to_owned())
            ),
            Part::Image { image } => {
                format!(
                    "Image(\"{}\")",
                    serde_json::to_string(image).unwrap_or("".to_owned())
                )
            }
        };
        format!("Part.{}", s)
    }

    #[getter]
    fn part_type(&self) -> &'static str {
        match &self {
            Part::Text { .. } => "text",
            Part::Function { .. } => "function",
            Part::Value { .. } => "value",
            Part::Image { .. } => "image",
        }
    }

    // #[getter]
    // fn text(&self) -> Option<String> {
    //     match &self {
    //         Part::Text { .. } => Some(self.to_string()),
    //         _ => None,
    //     }
    // }

    // #[getter]
    // fn function(&self) -> Option<String> {
    //     match &self {
    //         Part::Function { .. } => Some(self.to_string()),
    //         _ => None,
    //     }
    // }

    // #[getter]
    // fn image(&self) -> Option<String> {
    //     match &self {
    //         Part::Image { .. } => Some(self.to_string()),
    //         _ => None,
    //     }
    // }

    // #[getter]
    // fn mime_type(&self) -> Option<String> {
    //     match &self {
    //         Part::ImageData { mime_type, .. } => Some(mime_type.clone()),
    //         _ => None,
    //     }
    // }
}

#[gen_stub_pymethods]
#[pymethods]
impl PartDelta {
    pub fn __repr__(&self) -> String {
        let s = match &self {
            PartDelta::Text { text } => format!("Text(\"{}\")", text.replace('\n', "\\n")),
            PartDelta::Function { .. } => {
                format!(
                    "Function({})",
                    serde_json::to_string(self).unwrap_or("".to_owned())
                )
            }
            PartDelta::Value { value } => format!(
                "Value({})",
                serde_json::to_string(value).unwrap_or("{...}".to_owned())
            ),
            PartDelta::Null {} => "Null()".to_owned(),
        };
        format!("PartDelta.{}", s)
    }

    #[getter]
    fn part_type(&self) -> &'static str {
        match &self {
            PartDelta::Text { .. } => "text",
            PartDelta::Function { .. } => "function",
            PartDelta::Value { .. } => "value",
            PartDelta::Null {} => "null",
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Message {
    #[new]
    #[pyo3(signature = (role, contents = None, id = None, thinking = None, tool_calls = None, signature = None))]
    fn __new__(
        role: Role,
        contents: Option<Vec<Part>>,
        id: Option<String>,
        thinking: Option<String>,
        tool_calls: Option<Vec<Part>>,
        signature: Option<String>,
    ) -> Self {
        Self {
            role,
            contents: contents.unwrap_or_default(),
            id,
            thinking,
            tool_calls,
            signature,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Message(role={}, contents=[{}], id={}, thinking={}, tool_calls=[{}], signature={})",
            self.role.__repr__(),
            self.contents
                .iter()
                .map(|content| content.__repr__())
                .collect::<Vec<_>>()
                .join(", "),
            self.id.__repr__(),
            self.thinking.__repr__(),
            self.tool_calls.as_ref().map_or(String::new(), |calls| {
                calls
                    .iter()
                    .map(|tool_part| tool_part.__repr__())
                    .collect::<Vec<_>>()
                    .join(", ")
            }),
            self.signature.__repr__(),
        )
    }

    fn append_contents(&mut self, part: Part) {
        self.contents.push(part);
    }

    fn append_tool_call(&mut self, part: Part) {
        match &mut self.tool_calls {
            Some(tool_calls) => tool_calls.push(part),
            None => self.tool_calls = vec![part].into(),
        };
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MessageDelta {
    #[new]
    #[pyo3(signature = (role=None, contents = None, id = None, thinking = None, tool_calls = None, signature = None))]
    fn __new__(
        role: Option<Role>,
        contents: Option<Vec<PartDelta>>,
        id: Option<String>,
        thinking: Option<String>,
        tool_calls: Option<Vec<PartDelta>>,
        signature: Option<String>,
    ) -> Self {
        Self {
            role,
            contents: contents.unwrap_or_default(),
            id,
            thinking,
            tool_calls: tool_calls.unwrap_or_default(),
            signature,
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MessageDelta(role={}, contents=[{}], id={}, thinking={}, tool_calls=[{}], signature={})",
            self.role.__repr__(),
            self.contents
                .iter()
                .map(|content| content.__repr__())
                .collect::<Vec<_>>()
                .join(", "),
            self.id.__repr__(),
            self.thinking.__repr__(),
            self.tool_calls
                .iter()
                .map(|tool| tool.__repr__())
                .collect::<Vec<_>>()
                .join(", "),
            self.signature.__repr__(),
        )
    }

    fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.clone().accumulate(other.clone()).map_err(Into::into)
    }

    #[pyo3(name = "to_message")]
    pub fn to_message_py(&self) -> PyResult<Message> {
        self.clone().to_message().map_err(Into::into)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl FinishReason {
    pub fn __repr__(&self) -> String {
        match self {
            FinishReason::Stop {} => "FinishReason.Stop()".to_owned(),
            FinishReason::Length {} => "FinishReason.Length()".to_owned(),
            FinishReason::ToolCall {} => "FinishReason.ToolCall()".to_owned(),
            FinishReason::Refusal { reason } => {
                format!("FinishReason.Refusal(reason={})", reason)
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MessageDeltaOutput {
    pub fn __repr__(&self) -> String {
        format!(
            "MessageOutput(delta={}, finish_reason={})",
            self.delta.__repr__(),
            self.finish_reason
                .clone()
                .map(|finish_reason| finish_reason.__repr__())
                .unwrap_or("None".to_owned())
        )
    }

    #[getter]
    fn delta(&self) -> MessageDelta {
        self.delta.clone()
    }

    #[getter]
    fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl ToolDesc {
    #[new]
    #[pyo3(signature = (name, description, parameters, *, returns=None))]
    fn __new__(
        py: Python<'_>,
        name: String,
        description: Option<String>,
        parameters: Py<PyDict>,
        returns: Option<Py<PyDict>>,
    ) -> PyResult<Self> {
        let parameters = python_to_value(parameters.bind(py))?;
        let returns = if let Some(returns) = returns {
            Some(python_to_value(returns.bind(py))?)
        } else {
            None
        };
        Ok(Self::new(name, description, parameters, returns)).map_err(Into::into)
    }

    pub fn __repr__(&self) -> String {
        let returns_display = self
            .returns
            .clone()
            .and_then(|v| serde_json::to_string(&v).ok());

        if let Some(returns_display) = returns_display {
            format!(
                "ToolDesc(name=\"{}\", description={}, parameters={}, returns={})",
                self.name,
                self.description.clone().unwrap_or_default(),
                serde_json::to_string(&self.parameters).expect("Invalid parameters"),
                returns_display,
            )
        } else {
            format!(
                "ToolDesc(name=\"{}\", description={}, parameters={})",
                self.name,
                self.description.clone().unwrap_or_default(),
                serde_json::to_string(&self.parameters).expect("Invalid parameters"),
            )
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn description(&self) -> Option<String> {
        self.description.clone()
    }

    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        value_to_python(py, &self.parameters)
            .unwrap()
            .cast_into()
            .unwrap()
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        self.returns
            .as_ref()
            .and_then(|returns| value_to_python(py, returns).ok()?.cast_into().ok())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Document {
    #[new]
    #[pyo3(signature = (id, text, title=None))]
    fn __new__(id: String, text: String, title: Option<String>) -> Self {
        match title {
            Some(title) => Self::new(id, text).with_title(title),
            None => Self::new(id, text),
        }
    }
}
