use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub struct Document {
    pub title: String,
    pub text: String,
}
