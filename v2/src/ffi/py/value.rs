use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::{json_to_pydict, pydict_to_json},
    value::{
        FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolDesc, ToolDescArg,
    },
};

#[gen_stub_pymethods]
#[pymethods]
impl Part {
    /// Part(part_type, *, id=None, text=None, url=None, data=None, function=None)
    ///
    /// Examples:
    /// - Part(part_type="text", text="hello")
    /// - Part(part_type="image", url="https://example.com/cat.png")
    /// - Part(part_type="image", data="<base64>", mime_type="image/jpeg")  # 'base64=' alias also accepted
    /// - Part(part_type="function", function='{"name":"foo","arguments":"{}"}')
    #[new]
    #[pyo3(signature = (part_type, *, text=None, url=None, data=None, mime_type=None, function=None, id=None, name=None, arguments=None))]
    fn __new__(
        part_type: &str,
        text: Option<String>,
        url: Option<String>,
        data: Option<String>,
        mime_type: Option<String>,
        function: Option<String>,
        id: Option<String>,
        name: Option<String>,
        arguments: Option<String>,
    ) -> PyResult<Self> {
        let inner = match part_type {
            "text" => Part::Text(text.ok_or_else(|| PyTypeError::new_err("text= required"))?),
            "function" => {
                if function.is_some() {
                    Part::new_function_string(function.unwrap())
                } else if name.is_some() || arguments.is_some() {
                    Part::new_function(
                        id.unwrap_or_default(),
                        name.unwrap_or_default(),
                        arguments.unwrap_or_default(),
                    )
                } else {
                    Err(PyTypeError::new_err(
                        "function= or name= or arguments= required",
                    ))?
                }
            }
            "image" => {
                if let Some(u) = url {
                    Part::ImageURL(u)
                } else if let Some(data) = data
                    && let Some(mime_type) = mime_type
                {
                    Part::ImageData { data, mime_type }
                } else {
                    return Err(PyTypeError::new_err(
                        "image needs url= or data= with mime_type=",
                    ));
                }
            }
            other => return Err(PyValueError::new_err(format!("unknown type: {other}"))),
        };
        Ok(inner)
    }

    fn __repr__(&self) -> String {
        self.to_string().unwrap_or("".into())
    }

    #[getter]
    fn part_type(&self) -> &'static str {
        match &self {
            Part::Text(_) => "text",
            Part::FunctionString(_) => "function",
            Part::Function { .. } => "function",
            Part::ImageURL(_) | Part::ImageData { .. } => "image",
            // Part::AudioURL(_) | Part::AudioData(_) => "audio",
            // Part::Audio { .. } => "audio",
        }
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        match &self {
            Part::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn function(&self) -> Option<&str> {
        match &self {
            Part::FunctionString(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn url(&self) -> Option<&str> {
        match &self {
            Part::ImageURL(u) => Some(u.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn data(&self) -> Option<&str> {
        match &self {
            Part::ImageData { data, .. } => Some(data.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn mime_type(&self) -> Option<&str> {
        match &self {
            Part::ImageData { mime_type, .. } => Some(mime_type.as_str()),
            _ => None,
        }
    }

    // #[staticmethod]
    // fn from_json(s: &str) -> PyResult<Self> {
    //     Ok(PyPart {
    //         inner: serde_json::from_str::<Part>(s)
    //             .map_err(|e| PyValueError::new_err(e.to_string()))?,
    //     })
    // }

    // fn to_json(&self) -> PyResult<String> {
    //     serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    // }
}

#[gen_stub_pymethods]
#[pymethods]
impl Message {
    #[new]
    fn __new__(role: &Role) -> Self {
        Self::with_role(role.clone())
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }

    #[getter]
    fn role(&self) -> Option<Role> {
        self.role.clone()
    }

    #[getter]
    fn content(&self) -> Vec<Part> {
        self.contents.clone()
    }

    #[setter]
    fn set_content(&mut self, contents: Vec<Part>) {
        self.contents = contents;
    }

    fn append_content(&mut self, part: Part) {
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

    // #[staticmethod]
    // fn from_json(s: &str) -> PyResult<Self> {
    //     Ok(PyMessage {
    //         inner: serde_json::from_str::<Message>(s)
    //             .map_err(|e| PyValueError::new_err(e.to_string()))?,
    //     })
    // }

    // fn to_json(&self) -> PyResult<String> {
    //     serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    // }
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

    // #[staticmethod]
    // fn from_json(s: &str) -> PyResult<Self> {
    //     Ok(PyMessageOutput {
    //         inner: serde_json::from_str::<MessageOutput>(s)
    //             .map_err(|e| PyValueError::new_err(e.to_string()))?,
    //     })
    // }

    // fn to_json(&self) -> PyResult<String> {
    //     serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    // }
}

#[gen_stub_pymethods]
#[pymethods]
impl MessageAggregator {
    #[new]
    fn __new__() -> Self {
        Self::new()
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
        let parameters =
            serde_json::from_value::<ToolDescArg>(pydict_to_json(py, parameters.bind(py)).unwrap())
                .expect("parameters is not a valid JSON schema");
        let returns = if let Some(returns) = returns {
            Some(
                serde_json::from_value::<ToolDescArg>(
                    pydict_to_json(py, returns.bind(py)).unwrap(),
                )
                .expect("returns is not a valid JSON schema"),
            )
        } else {
            None
        };
        Ok(Self::new(name, description, parameters, returns))
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
        json_to_pydict(py, &json_value).unwrap()
    }

    #[getter]
    fn returns<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyDict>> {
        match &self.r#return {
            Some(r#return) => {
                let json_value = serde_json::to_value(r#return).unwrap();
                Some(json_to_pydict(py, &json_value).unwrap())
            }
            None => None,
        }
    }
}
