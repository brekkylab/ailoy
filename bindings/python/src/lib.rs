//! Python bindings for ailoy, imported as `ailoy._ailoy`.
//!
//! The shape is ailoy's own, spelled in Python: an [`Agent`](ailoy::agent::Agent) is built by
//! an [`AgentBuilder`](ailoy::agent::AgentBuilder) and awaited, a turn is iterated with
//! `async for`, and the registries are the process-wide ones the crate keeps. Nothing adds a
//! layer over that, so ailoy's Rust documentation applies as is.
//!
//! Data (messages, specs, tool descriptions, a turn's outputs) crosses as dicts in its serde
//! form rather than a class per type, since that form is what ailoy stores and sends.
//!
//! virtx's classes are registered in from `virtx-python` rather than imported from virtx's
//! own extension, because an agent takes a `ConsoleClient` apart to share its session and two
//! extension modules cannot see into each other's.

use pyo3::prelude::*;

mod agent;
mod convert;
mod error;
mod registry;
mod tool;

#[pymodule]
fn _ailoy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    _virtx::register(m)?;
    error::register(m)?;
    registry::register(m)?;
    agent::register(m)?;
    Ok(())
}
