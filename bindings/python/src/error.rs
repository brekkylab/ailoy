//! How ailoy's failures arrive in Python.
//!
//! `AiloyError` for everything ailoy itself reports (an unserved model, an unregistered tool,
//! an API error). Console failures keep virtx's classes, so a refusal inside a turn is the
//! same `ConsoleRefused`, with the same `code`, as one from `ConsoleClient.exec`.

use pyo3::{create_exception, exceptions::PyException, prelude::*};
use virtx::protocol::Failure;

create_exception!(ailoy, AiloyError, PyException);

pub fn anyhow(error: anyhow::Error) -> PyErr {
    match error.downcast::<Failure>() {
        Ok(failure) => _virtx::error::failure(failure),
        Err(error) => AiloyError::new_err(format!("{error:#}")),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AiloyError", m.py().get_type::<AiloyError>())?;
    Ok(())
}
