use std::str::FromStr;

use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};

use crate::value::{Message, Part, Role};

#[pyclass(name = "Part")]
#[derive(Clone)]
pub struct PyPart {
    inner: Part,
}

#[pymethods]
impl PyPart {
    /// Part(type, *, id=None, text=None, url=None, data=None, base64=None, function=None)
    ///
    /// Examples:
    /// - Part(type="text", text="hello")
    /// - Part(type="image", url="https://example.com/cat.png")
    /// - Part(type="image", data="<base64>")  # 'base64=' alias also accepted
    /// - Part(type="function", function=r#"{"name":"foo","arguments":"{}"}"#, id="call_1")
    #[new]
    #[pyo3(signature = (r#type, *, id=None, text=None, url=None, data=None, base64=None, function=None))]
    fn new(
        r#type: &str,
        id: Option<String>,
        text: Option<String>,
        url: Option<String>,
        data: Option<String>,
        base64: Option<String>,
        function: Option<String>,
    ) -> PyResult<Self> {
        let inner = match r#type {
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
                } else if let Some(b) = data.or(base64) {
                    Part::ImageBase64(b)
                } else {
                    return Err(PyTypeError::new_err("image needs url= or data=/base64="));
                }
            }
            other => return Err(PyValueError::new_err(format!("unknown type: {other}"))),
        };
        Ok(Self { inner })
    }

    #[getter]
    fn r#type(&self) -> &'static str {
        match &self.inner {
            Part::Text(_) => "text",
            Part::Function { .. } => "function",
            Part::ImageURL(_) | Part::ImageBase64(_) => "image",
        }
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    // Accessors return Option[str] (None if wrong variant)
    fn text(&self) -> Option<&str> {
        self.inner.get_text()
    }

    fn function_id(&self) -> Option<String> {
        self.inner.get_function_id()
    }

    fn function_raw(&self) -> Option<&String> {
        self.inner.get_function()
    }

    fn url(&self) -> Option<String> {
        match &self.inner {
            Part::ImageURL(u) => Some(u.as_str().to_string()),
            _ => None,
        }
    }

    fn base64(&self) -> Option<&str> {
        match &self.inner {
            Part::ImageBase64(b) => Some(b.as_str()),
            _ => None,
        }
    }

    /// Serialize using your Rust `Serialize` (OpenAI-shaped dict).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Part(type={})", self.r#type()))
    }
}

#[pyclass(name = "Message")]
pub struct PyMessage {
    inner: Message,
}

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

    /// Create from a JSON string that matches your Rust `Message` serde format.
    #[staticmethod]
    fn from_json(js: &str) -> PyResult<Self> {
        let msg: Message = serde_json::from_str(js).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid Message JSON: {e}"))
        })?;
        Ok(Self { inner: msg })
    }

    /// Convert the message to a JSON string using Rust serde.
    fn to_json(&self) -> PyResult<String> {
        let msg = &self.inner;
        serde_json::to_string(msg)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // --- role ---

    #[getter]
    fn role(&self) -> PyResult<String> {
        Ok(self.inner.role().to_string())
    }

    fn set_role(&mut self, role: &str) -> PyResult<()> {
        let role = Role::from_str(role).map_err(|_| PyValueError::new_err(role.to_owned()))?;
        let m = &self.inner;
        // rebuild to preserve existing vectors
        let mut newm = Message::new(role);
        for p in m.content().iter().cloned() {
            newm.push_content(p);
        }
        for p in m.reasoning().iter().cloned() {
            newm.push_reasoning(p);
        }
        for p in m.tool_calls().iter().cloned() {
            newm.push_tool_call(p);
        }
        self.inner = newm;
        Ok(())
    }

    fn push_content(&mut self, part: PyPart) {
        self.inner.push_content(part.inner);
    }

    fn push_reasoning(&mut self, part: PyPart) {
        self.inner.push_reasoning(part.inner);
    }

    fn push_tool_call(&mut self, part: PyPart) {
        self.inner.push_tool_call(part.inner);
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", serde_json::to_string(&self.inner).unwrap(),))
    }
}

#[pymodule(name = "_value")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyMessage>()?;
    Ok(())
}
