//! A `virtx.Sandbox` from the `virtx` package, as the sandbox an agent runs in.
//!
//! Both packages reach the server through one `libvirtx_core` in the process: this module opens
//! the one `virtx` opened, and takes a hold of its client's `VirtxClient*` of its own, so the
//! sandbox the agent holds is the caller's, over the same session.

use std::{ffi::c_void, mem::ManuallyDrop};

use pyo3::{exceptions::PyTypeError, prelude::*};

use crate::error;

/// The sandbox `sandbox`, a `virtx.Sandbox`, names, over its own client.
pub fn sandbox(sandbox: &Bound<'_, PyAny>) -> PyResult<virtx::Sandbox> {
    let py = sandbox.py();
    let virtx = py.import("virtx")?;
    if !sandbox.is_instance(&virtx.getattr("Sandbox")?)? {
        return Err(PyTypeError::new_err(format!(
            "expected a virtx.Sandbox, got {}",
            sandbox.get_type().name()?
        )));
    }
    let path: Option<String> = virtx.getattr("library_path")?.call0()?.extract()?;
    let path = path.ok_or_else(|| PyTypeError::new_err("virtx has no library open"))?;
    virtx::load(&path).map_err(error::virtx)?;

    let raw: usize = sandbox
        .getattr("client")?
        .call_method0("as_raw")?
        .extract()?;
    // Safety: `virtx` made it with the library just opened, and holds it for as long as this
    // borrows it; the sandbox takes a hold of its own.
    let borrowed = ManuallyDrop::new(
        unsafe { virtx::Client::from_raw(raw as *mut c_void) }.map_err(error::virtx)?,
    );
    let info = virtx::SandboxInfo {
        id: sandbox.getattr("id")?.extract()?,
        image: sandbox.getattr("image")?.extract()?,
        snapshot: sandbox.getattr("snapshot")?.extract()?,
        running: sandbox.getattr("running")?.extract()?,
        ports: Vec::new(),
    };
    Ok((*borrowed).sandbox(info))
}
