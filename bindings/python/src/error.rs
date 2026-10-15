//! How ailoy's failures arrive in Python.
//!
//! `AiloyError` for everything ailoy itself reports (an unserved model, an unregistered tool,
//! an API error). Console failures are raised as the `virtx` package's own classes, so a
//! refusal inside a turn is the same `virtx.ConsoleRefused`, with the same `code`, as one from
//! `ConsoleClient.exec`.

use pyo3::{create_exception, exceptions::PyException, prelude::*};
use virtx::protocol::Failure;

create_exception!(ailoy, AiloyError, PyException);

pub fn anyhow(error: anyhow::Error) -> PyErr {
    match error.downcast::<Failure>() {
        Ok(failure) => Python::attach(|py| self::failure(py, failure).unwrap_or_else(|e| e)),
        Err(error) => AiloyError::new_err(format!("{error:#}")),
    }
}

/// As virtx raises it: `ConsoleRefused` with the server's message and its `code`, or
/// `ConsoleBroken` for a channel that is gone.
fn failure(py: Python<'_>, failure: Failure) -> PyResult<PyErr> {
    let virtx = py.import("virtx")?;
    let error = match failure {
        Failure::Refused(error) => {
            let raised = virtx.getattr("ConsoleRefused")?.call1((error.message,))?;
            // An attribute, not an argument, so `str(err)` stays the server's message.
            raised.setattr("code", error.code)?;
            raised
        }
        Failure::Broken(error) => virtx
            .getattr("ConsoleBroken")?
            .call1((format!("{error:#}"),))?,
    };
    Ok(PyErr::from_value(error))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AiloyError", m.py().get_type::<AiloyError>())?;
    Ok(())
}
