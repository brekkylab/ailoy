use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
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
