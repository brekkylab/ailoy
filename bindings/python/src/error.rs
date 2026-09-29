//! How ailoy's failures arrive in Python.
//!
//! One exception, `AiloyError`, for everything ailoy itself reports — a model that no
//! provider serves, a tool that is not registered, an API that answered with an error. What
//! a console reports keeps cortex's own classes: a refusal inside an agent's turn is the
//! same `ConsoleRefused`, with the same `code`, as one from `ConsoleClient.exec`, so a caller
//! handles it once.

use cortex::protocol::Failure;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(ailoy, AiloyError, PyException);

pub fn anyhow(error: anyhow::Error) -> PyErr {
    match error.downcast::<Failure>() {
        Ok(failure) => _cortex::error::failure(failure),
        Err(error) => AiloyError::new_err(format!("{error:#}")),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AiloyError", m.py().get_type::<AiloyError>())?;
    Ok(())
}
