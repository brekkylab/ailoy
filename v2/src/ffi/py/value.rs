use std::str::FromStr;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyString},
};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{json_to_pydict, pydict_to_json},
    value::{FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolDesc},
};

#[gen_stub_pymethods]
#[pymethods]
impl Part {
    fn __repr__(&self) -> String {
        let s = match &self {
            Part::Text(text) => format!("Text(\"{}\")", text),
            Part::FunctionString(_) | Part::Function { .. } => {
                format!("Function({})", self.to_string())
            }
            Part::ImageURL(url) => format!("ImageURL(\"{}\")", url),
            Part::ImageData { .. } => {
                let mut s = self.to_string();
                if s.len() > 30 {
                    s.truncate(30);
                    s += "...";
                }
                format!("ImageData(\"{}\")", s)
            }
        };
        format!("Part.{}", s)
    }

    #[getter]
    fn part_type(&self) -> &'static str {
        match &self {
            Part::Text(_) => "text",
            Part::FunctionString(_) | Part::Function { .. } => "function",
            Part::ImageURL(_) => "image_url",
            Part::ImageData { .. } => "image_data",
        }
    }

    #[getter]
    fn text(&self) -> Option<String> {
        match &self {
            Part::Text(..) => Some(self.to_string()),
            _ => None,
        }
    }

    #[getter]
    fn function(&self) -> Option<String> {
        match &self {
            Part::Function { .. } | Part::FunctionString(..) => Some(self.to_string()),
            _ => None,
        }
    }

    #[getter]
    fn url(&self) -> Option<String> {
        match &self {
            Part::ImageURL(..) => Some(self.to_string()),
            _ => None,
        }
    }

    #[getter]
    fn data(&self) -> Option<String> {
        match &self {
            Part::ImageData { data, .. } => Some(data.clone()),
            _ => None,
        }
    }

    #[getter]
    fn mime_type(&self) -> Option<String> {
        match &self {
            Part::ImageData { mime_type, .. } => Some(mime_type.clone()),
            _ => None,
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl Message {
    #[new]
    fn __new__() -> PyResult<Self> {
        Ok(Self::default())
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    fn role(&self) -> Option<Role> {
        self.role.clone()
    }

    #[setter]
    fn set_role(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "Role | typing.Literal[\"system\",\"user\",\"assistant\",\"tool\"]"
        ))]
        role: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            if let Ok(role) = role.downcast::<PyString>() {
                self.role = Some(
                    Role::from_str(&role.to_string())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
                Ok(())
            } else if let Ok(role) = role.downcast::<Role>() {
                self.role = Some(role.clone().unbind().borrow(py).clone());
                Ok(())
            } else {
                return Err(PyTypeError::new_err("role should be either Role or str"));
            }
        })
    }

    #[getter]
    fn contents(&self) -> Vec<Part> {
        self.contents.clone()
    }

    #[setter]
    fn set_contents(&mut self, contents: Vec<Part>) {
        self.contents = contents;
    }

    fn append_contents(&mut self, part: Part) {
        self.contents.push(part);
    }

    #[getter]
    fn reasoning(&self) -> String {
        self.reasoning.clone()
    }

    #[setter]
    fn set_reasoning(&mut self, reasoning: String) {
        self.reasoning = reasoning;
    }

    #[getter]
    fn tool_calls(&self) -> Vec<Part> {
        self.tool_calls.clone()
    }

    #[setter]
    fn set_tool_calls(&mut self, tool_calls: Vec<Part>) {
        self.tool_calls = tool_calls;
    }

    fn append_tool_call(&mut self, part: Part) {
        self.tool_calls.push(part);
    }

    #[getter]
    fn tool_call_id(&self) -> Option<String> {
        self.tool_call_id.clone()
    }

    #[setter]
    fn set_tool_call_id(&mut self, tool_call_id: Option<String>) {
        self.tool_call_id = tool_call_id;
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MessageOutput {
    fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    fn delta(&self) -> Message {
        self.delta.clone()
    }

    #[getter]
    fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason.clone()
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl MessageAggregator {
    #[new]
    fn __new__() -> Self {
        Self::new()
    }

    #[getter]
    fn buffer(&self) -> Option<Message> {
        self.buffer.clone()
    }

    #[pyo3(name = "update")]
    fn update_(&mut self, msg_out: MessageOutput) -> Option<Message> {
        self.update(msg_out)
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
        description: String,
        parameters: Py<PyDict>,
        returns: Option<Py<PyDict>>,
    ) -> PyResult<Self> {
        let parameters = serde_json::Value::Object(pydict_to_json(py, parameters.bind(py))?);
        let returns = if let Some(returns) = returns {
            Some(serde_json::Value::Object(pydict_to_json(
                py,
                returns.bind(py),
            )?))
        } else {
            None
        };
        Self::new(name, description, parameters, returns).map_err(|e| PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        format!(
            "ToolDesc(name=\"{}\", description=\"{}\")",
            self.name, self.description
        )
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    fn description(&self) -> String {
        self.description.clone()
    }

    #[getter]
    fn parameters<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let json_value = serde_json::to_value(self.parameters.clone()).unwrap();
        json_to_pydict(py, &json_value).unwrap().unwrap()
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        match &self.returns {
            Some(returns) => {
                let json_value = serde_json::to_value(returns).unwrap();
                Some(json_to_pydict(py, &json_value).unwrap().unwrap())
            }
            None => None,
        }
    }
}
