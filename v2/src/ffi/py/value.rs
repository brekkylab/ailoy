use std::str::FromStr;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyList, PyString},
};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::PyWrapper,
    value::{Message, MessageAggregator, MessageOutput, Part, Role},
};

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "Part")]
pub struct PyPart {
    inner: Part,
}

impl PyWrapper for PyPart {
    type Inner = Part;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPart {
    /// Part(part_type, *, id=None, text=None, url=None, data=None, function=None)
    ///
    /// Examples:
    /// - Part(part_type="text", text="hello")
    /// - Part(part_type="image", url="https://example.com/cat.png")
    /// - Part(part_type="image", data="<base64>", mime_type="image/jpeg")  # 'base64=' alias also accepted
    /// - Part(part_type="function", function='{"name":"foo","arguments":"{}"}')
    #[new]
    #[pyo3(signature = (part_type, *, text=None, url=None, data=None, mime_type=None, function=None, id=None, name=None, arguments=None))]
    fn new(
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
                    Part::ImageData(data, mime_type)
                } else {
                    return Err(PyTypeError::new_err(
                        "image needs url= or data= with mime_type=",
                    ));
                }
            }
            other => return Err(PyValueError::new_err(format!("unknown type: {other}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn part_type(&self) -> &'static str {
        match &self.inner {
            Part::Text(_) => "text",
            Part::FunctionString(_) => "function",
            Part::Function { .. } => "function",
            Part::ImageURL(_) | Part::ImageData(_, _) => "image",
            // Part::AudioURL(_) | Part::AudioData(_) => "audio",
            // Part::Audio { .. } => "audio",
        }
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        match &self.inner {
            Part::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn function(&self) -> Option<&str> {
        match &self.inner {
            Part::FunctionString(s) => Some(s.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn url(&self) -> Option<&str> {
        match &self.inner {
            Part::ImageURL(u) => Some(u.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn data(&self) -> Option<&str> {
        match &self.inner {
            Part::ImageData(b, _) => Some(b.as_str()),
            _ => None,
        }
    }

    #[getter]
    fn mime_type(&self) -> Option<&str> {
        match &self.inner {
            Part::ImageData(_, mime_type) => Some(mime_type.as_str()),
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

    // fn __repr__(&self) -> PyResult<String> {
    //     Ok(self.inner.to_string())
    // }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "Message")]
pub struct PyMessage {
    pub inner: Message,
}

impl PyWrapper for PyMessage {
    type Inner = Message;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMessage {
    /// Message(role: str)
    /// role is one of: "system" | "user" | "assistant" | "tool"
    #[new]
    fn new(role: &str) -> PyResult<Self> {
        let role = Role::from_str(role).map_err(|_| PyValueError::new_err(role.to_owned()))?;
        Ok(Self {
            inner: Message::with_role(role),
        })
    }

    #[getter]
    fn role(&self) -> PyResult<Option<String>> {
        Ok(self.inner.role.as_ref().map(|r| r.to_string()))
    }

    #[getter]
    fn content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner
                .contents
                .clone()
                .into_iter()
                .map(|inner| PyPart { inner }),
        )
    }

    #[setter]
    fn set_content(&mut self, contents: Vec<PyPart>) {
        self.inner.contents = contents.into_iter().map(|v| v.inner).collect();
    }

    fn append_content(&mut self, part: PyPart) {
        self.inner.contents.push(part.inner);
    }

    #[getter]
    fn reasoning<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(PyString::new(py, &self.inner.reasoning))
    }

    #[setter]
    fn set_reasoning(&mut self, reasoning: String) {
        self.inner.reasoning = reasoning;
    }

    #[getter]
    fn tool_calls<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner
                .tool_calls
                .clone()
                .into_iter()
                .map(|inner| PyPart { inner }),
        )
    }

    #[setter]
    fn set_tool_calls(&mut self, tool_calls: Vec<PyPart>) {
        self.inner.tool_calls = tool_calls.into_iter().map(|v| v.inner).collect();
    }

    fn append_tool_call(&mut self, part: PyPart) {
        self.inner.tool_calls.push(part.inner);
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

    // fn __repr__(&self) -> PyResult<String> {
    //     Ok(self.inner.to_string())
    // }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "MessageOutput")]
pub struct PyMessageOutput {
    inner: MessageOutput,
}

impl PyWrapper for PyMessageOutput {
    type Inner = MessageOutput;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMessageOutput {
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "MessageAggregator")]
pub struct PyMessageAggregator {
    inner: MessageAggregator,
}

impl PyWrapper for PyMessageAggregator {
    type Inner = MessageAggregator;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMessageAggregator {}
