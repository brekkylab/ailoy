//! Crossing as dicts: ailoy's types to Python through their serde form, and back.
//!
//! A [`Message`] crosses both ways as the JSON ailoy writes, `{"role": "assistant", ...}`,
//! except that bytes (an embedded image) are `bytes` rather than base64.

use ailoy::message::{Message, Part, Role};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Serialize, de::DeserializeOwned};

pub fn to_py<'py>(py: Python<'py>, value: &impl Serialize) -> PyResult<Bound<'py, PyAny>> {
    Ok(pythonize::pythonize(py, value)?)
}

/// `ValueError` naming what was expected, since serde's error names only the field it
/// tripped on.
pub fn from_py<T: DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    pythonize::depythonize(obj).map_err(|e| PyValueError::new_err(format!("not {what}: {e}")))
}

/// A query as a caller may spell it: a message, or a string as the user's text.
pub struct Query(pub Message);

impl<'a, 'py> FromPyObject<'a, 'py> for Query {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(text) = obj.extract::<String>() {
            return Ok(Query(
                Message::new(Role::User).with_contents([Part::text(text)]),
            ));
        }
        Ok(Query(from_py(&obj, "a message")?))
    }
}
