//! Python bindings for ailoy, imported as `ailoy._ailoy`.
//!
//! The shape is ailoy's own, spelled in Python: an [`Agent`](ailoy::agent::Agent) is built by
//! an [`AgentBuilder`](ailoy::agent::AgentBuilder) and awaited, a turn is a stream iterated
//! with `async for`, and the registries an agent resolves its model and tools from are the
//! process-wide ones the crate keeps. Nothing here adds a layer of its own over that — a
//! Python caller reading ailoy's Rust documentation should find the same names doing the same
//! things.
//!
//! What crosses the boundary as data — messages, specs, tool descriptions, a turn's outputs —
//! crosses as the dicts its serde form already is, rather than as a class per type: that form
//! is what ailoy stores and sends, so it is the one a caller already has.
//!
//! cortex comes along whole. Its classes are registered into this module from
//! `cortex-python` rather than imported from cortex's own extension, because an agent has to
//! take a `ConsoleClient` apart to share its session, and two extension modules cannot see into
//! each other's — see [`agent`] for the sharing.

use pyo3::prelude::*;

mod agent;
mod convert;
mod error;
mod registry;
mod tool;

#[pymodule]
fn _ailoy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    _cortex::register(m)?;
    error::register(m)?;
    registry::register(m)?;
    agent::register(m)?;
    Ok(())
}
