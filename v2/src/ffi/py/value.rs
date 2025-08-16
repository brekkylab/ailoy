use std::str::FromStr;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyList,
};
use pyo3_stub_gen::derive::*;

use crate::{
    ffi::py::base::PyWrapper,
    value::{Message, MessageAggregator, MessageDelta, Part, Role},
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
    /// - Part(part_type="image", data="<base64>")  # 'base64=' alias also accepted
    /// - Part(part_type="function", function='{"name":"foo","arguments":"{}"}', id="call_1")
    #[new]
    #[pyo3(signature = (part_type, *, id=None, text=None, url=None, data=None, function=None))]
    fn new(
        part_type: &str,
        id: Option<String>,
        text: Option<String>,
        url: Option<String>,
        data: Option<String>,
        function: Option<String>,
    ) -> PyResult<Self> {
        let inner = match part_type {
            "text" => Part::Text(text.ok_or_else(|| PyTypeError::new_err("text= required"))?),
            "function" => Part::Function {
                id,
                function: function.ok_or_else(|| PyTypeError::new_err("function= required"))?,
            },
            "image" => {
                if let Some(u) = url {
                    Part::ImageURL(
                        url::Url::parse(&u).map_err(|e| PyValueError::new_err(e.to_string()))?,
                    )
                } else if let Some(b) = data {
                    Part::ImageData(b)
                } else {
                    return Err(PyTypeError::new_err("image needs url= or data=/base64="));
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
            Part::Function { .. } => "function",
            Part::ImageURL(_) | Part::ImageData(_) => "image",
            Part::Audio { .. } => "audio",
        }
    }

    #[getter]
    fn text(&self) -> Option<&str> {
        self.inner.get_text()
    }

    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.get_function_id()
    }

    #[getter]
    fn function(&self) -> Option<&String> {
        self.inner.get_function()
    }

    #[getter]
    fn url(&self) -> Option<String> {
        match &self.inner {
            Part::ImageURL(u) => Some(u.as_str().to_string()),
            _ => None,
        }
    }

    #[getter]
    fn data(&self) -> Option<&str> {
        match &self.inner {
            Part::ImageData(b) => Some(b.as_str()),
            _ => None,
        }
    }

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(PyPart {
            inner: serde_json::from_str::<Part>(s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
}

#[derive(Clone)]
#[gen_stub_pyclass]
#[pyclass(name = "MessageDelta")]
pub struct PyMessageDelta {
    inner: MessageDelta,
}

impl PyWrapper for PyMessageDelta {
    type Inner = MessageDelta;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyMessageDelta {
    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(PyMessageDelta {
            inner: serde_json::from_str::<MessageDelta>(s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(self.inner.to_string())
    }
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
            inner: Message::new(role),
        })
    }

    #[getter]
    fn role(&self) -> PyResult<String> {
        Ok(self.inner.role.to_string())
    }

    #[getter]
    fn content<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner
                .content
                .clone()
                .into_iter()
                .map(|inner| PyPart { inner }),
        )
    }

    #[setter]
    fn set_content(&mut self, content: Vec<PyPart>) {
        self.inner.content = content.into_iter().map(|v| v.inner).collect();
    }

    fn append_content(&mut self, part: PyPart) {
        self.inner.content.push(part.inner);
    }

    #[getter]
    fn reasoning<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            self.inner
                .reasoning
                .clone()
                .into_iter()
                .map(|inner| PyPart { inner }),
        )
    }

    #[setter]
    fn set_reasoning(&mut self, reasoning: Vec<PyPart>) {
        self.inner.reasoning = reasoning.into_iter().map(|v| v.inner).collect();
    }

    fn append_reasoning(&mut self, part: PyPart) {
        self.inner.reasoning.push(part.inner);
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

    #[staticmethod]
    fn from_json(s: &str) -> PyResult<Self> {
        Ok(PyMessage {
            inner: serde_json::from_str::<Message>(s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyValueError::new_err(e.to_string()))
    }

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
