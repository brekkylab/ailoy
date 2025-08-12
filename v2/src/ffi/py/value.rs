use std::str::FromStr;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::value::{Message, Role};

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
            newm.push_tool_calls(p);
        }
        self.inner = newm;
        Ok(())
    }

    // fn push_text(&self, text: &str) {
    //     self.inner.borrow_mut().push_content(Part::new_text(text));
    // }

    // /// `function_json` is stored **as-is** (raw JSON string). You may pass incomplete JSON while streaming.
    // fn push_function(&self, function_json: &str) {
    //     self.inner
    //         .borrow_mut()
    //         .push_tool_calls(Part::new_function(function_json));
    // }

    // fn push_function_with_id(&self, id: &str, function_json: &str) {
    //     self.inner
    //         .borrow_mut()
    //         .push_tool_calls(Part::new_function_with_id(id, function_json));
    // }

    // fn push_image_url(&self, url: &str) -> PyResult<()> {
    //     let url =
    //         Url::parse(url).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    //     self.inner.borrow_mut().push_content(Part::ImageURL(url));
    //     Ok(())
    // }

    // fn push_image_base64(&self, data: &str) {
    //     self.inner
    //         .borrow_mut()
    //         .push_content(Part::ImageBase64(data.to_string()));
    // }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "Message(role={}, content_len={}, reasoning_len={}, tool_calls_len={})",
            self.inner.role().to_string(),
            self.inner.content().len(),
            self.inner.reasoning().len(),
            self.inner.tool_calls().len(),
        ))
    }
}

#[pymodule(name = "_value")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyMessage>()?;
    Ok(())
}
