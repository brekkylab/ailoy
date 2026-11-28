//! A `virtx.Client` from the `virtx` package, as the client this process's provider makes each
//! agent's sandbox with.
//!
//! Both packages reach the server through one `libvirtx_core` in the process: this module opens
//! the one `virtx` opened, and takes a hold of the client's `VirtxClient*` of its own, so the
//! sandboxes are made over the caller's session.

use std::{ffi::c_void, mem::ManuallyDrop};

use ailoy::agent::default_provider;
use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::error;

/// The client `client`, a `virtx.Client`, holds, with a hold of its own.
fn client(client: &Bound<'_, PyAny>) -> PyResult<virtx::Client> {
    let py = client.py();
    let virtx = py.import("virtx")?;
    if !client.is_instance(&virtx.getattr("Client")?)? {
        return Err(PyTypeError::new_err(format!(
            "expected a virtx.Client, got {}",
            client.get_type().name()?
        )));
    }
    let path: Option<String> = virtx.getattr("library_path")?.call0()?.extract()?;
    let path = path.ok_or_else(|| PyTypeError::new_err("virtx has no library open"))?;
    virtx::load(&path).map_err(error::virtx)?;

    let raw: usize = client.call_method0("as_raw")?.extract()?;
    // Safety: `virtx` made it with the library just opened, and holds it for as long as this
    // borrows it; the clone is a hold of its own.
    let borrowed = ManuallyDrop::new(
        unsafe { virtx::Client::from_raw(raw as *mut c_void) }.map_err(error::virtx)?,
    );
    Ok((*borrowed).clone())
}

/// Make each agent's sandbox with `client`, a `virtx.Client`, from now on; with `None`, make
/// none, and an agent whose spec uses a sandbox is refused.
#[pyfunction]
#[pyo3(signature = (client))]
fn set_virtx(client: Option<Bound<'_, PyAny>>) -> PyResult<()> {
    let client = client.map(|given| self::client(&given)).transpose()?;
    let before = std::mem::replace(&mut default_provider().write().unwrap().virtx, client);
    // Let go of the one before off this thread: a last hold waits there for its session to end.
    std::thread::spawn(move || drop(before));
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_virtx, m)?)?;
    Ok(())
}
