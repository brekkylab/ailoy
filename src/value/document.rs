use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core", eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(from_wasm_abi, into_wasm_abi))]
pub struct Document {
    pub id: String,
    pub title: Option<String>,
    pub text: String,
}

impl Document {
    pub fn new(id: String, text: String) -> Self {
        Self {
            id,
            title: None,
            text,
        }
    }

    pub fn with_title(mut self, title: String) -> Self {
        self.title = Some(title);
        self
    }
}

impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = serde_json::to_string(self).map_err(|_| fmt::Error)?;
        write!(f, "Document {}", s)
    }
}
