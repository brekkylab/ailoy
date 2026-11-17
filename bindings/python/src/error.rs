//! How ailoy's failures arrive in Python.
//!
//! `AiloyError` for everything ailoy itself reports (an unserved model, an unregistered tool,
//! an API error). A sandbox's failures are raised as the `virtx` package's own `VirtxError`,
//! with the same `status`, as one from `Sandbox.exec` or `Client.read_file` is.

use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(ailoy, AiloyError, PyException);

pub fn anyhow(error: anyhow::Error) -> PyErr {
    match error.downcast::<virtx::Error>() {
        Ok(error) => virtx(error),
        Err(error) => AiloyError::new_err(format!("{error:#}")),
    }
}

/// As the `virtx` package raises it: `VirtxError` with the message and the server's `status`.
pub fn virtx(error: virtx::Error) -> PyErr {
    Python::attach(|py| {
        let raised = py
            .import("virtx")?
            .getattr("VirtxError")?
            .call1((error.to_string(), error.status()))?;
        Ok::<_, PyErr>(PyErr::from_value(raised))
    })
    .unwrap_or_else(|e| e)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("AiloyError", m.py().get_type::<AiloyError>())?;
    Ok(())
}
