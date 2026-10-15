//! The process-wide registry an agent builds its tools from: a provider per tool name, seeded
//! with every built-in tool, as models have one per org (see [`crate::model`]).
//!
//! The registering functions return the tool descriptions they registered, to pass to
//! `AgentBuilder.tool`/`tools`.

use ailoy::tool::{
    ToolDesc, register_a2a as register_a2a_rs, register_mcp_stdio as register_mcp_stdio_rs,
    register_mcp_streamable_http as register_mcp_streamable_http_rs,
    remove_tool_provider as remove_tool_provider_rs, set_tool_provider as set_tool_provider_rs,
    unregister_mcp as unregister_mcp_rs,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_async_runtimes::tokio::future_into_py;

use crate::{
    convert::{from_py, to_py},
    error,
    tool::tool_func,
};

fn parse_url(url: &str) -> PyResult<url::Url> {
    url::Url::parse(url).map_err(|e| PyValueError::new_err(format!("{url:?}: {e}")))
}

/// Descs as the list of dicts the builder takes back.
fn descs_to_py(py: Python<'_>, descs: &[ToolDesc]) -> PyResult<Py<PyAny>> {
    Ok(to_py(py, &descs)?.unbind())
}

fn ensure_callable(py: Python<'_>, func: &Py<PyAny>) -> PyResult<()> {
    if func.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyValueError::new_err("a tool's func has to be callable"))
    }
}

/// Build the tool `name` by calling `func`, replacing the provider it had. See
/// [`crate::tool`] for how `func` is called.
#[pyfunction]
fn set_tool_provider(py: Python<'_>, name: String, func: Py<PyAny>) -> PyResult<()> {
    ensure_callable(py, &func)?;
    set_tool_provider_rs(name, tool_func(func));
    Ok(())
}

/// Stop building the tool `name`. Answers whether it had a provider.
#[pyfunction]
fn remove_tool_provider(name: String) -> bool {
    remove_tool_provider_rs(&name).is_some()
}

/// Build the tool `desc` describes by calling `func`, and hand `desc` back.
///
/// `desc` is `{"name": ..., "description": ..., "parameters": <JSON schema>}`; the name is
/// what the model calls and what the provider is set under.
#[pyfunction]
fn register_tool<'py>(desc: Bound<'py, PyAny>, func: Py<PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = desc.py();
    ensure_callable(py, &func)?;
    let parsed: ToolDesc = from_py(&desc, "a tool description")?;
    set_tool_provider_rs(parsed.name.clone(), tool_func(func));
    to_py(py, &parsed)
}

/// Start the MCP server `command args…` on this host and register each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their descriptions.
///
/// The server runs outside any console, with this process's access — register only servers
/// you trust.
#[pyfunction]
#[pyo3(signature = (prefix, command, args = Vec::new()))]
fn register_mcp_stdio(
    py: Python<'_>,
    prefix: String,
    command: String,
    args: Vec<String>,
) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async move {
        let descs = register_mcp_stdio_rs(&prefix, &command, &args)
            .await
            .map_err(error::anyhow)?;
        Python::attach(|py| descs_to_py(py, &descs))
    })
}

/// Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their descriptions.
#[pyfunction]
fn register_mcp_streamable_http(
    py: Python<'_>,
    prefix: String,
    url: String,
) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async move {
        let descs = register_mcp_streamable_http_rs(&prefix, &url)
            .await
            .map_err(error::anyhow)?;
        Python::attach(|py| descs_to_py(py, &descs))
    })
}

/// Drop the tools registered under `prefix` and end that server's session. Answers with how
/// many were removed.
#[pyfunction]
fn unregister_mcp(prefix: String) -> usize {
    // Off the runtime is fine for the registry, but ending the session cancels a task on it.
    let _entered = pyo3_async_runtimes::tokio::get_runtime().enter();
    unregister_mcp_rs(&prefix)
}

/// Fetch the A2A agent card at `url` and register the agent as the tool `name`. Awaitable;
/// answers with its description.
#[pyfunction]
fn register_a2a(py: Python<'_>, name: String, url: String) -> PyResult<Bound<'_, PyAny>> {
    let url = parse_url(&url)?;
    future_into_py(py, async move {
        let desc = register_a2a_rs(&name, url).await.map_err(error::anyhow)?;
        Python::attach(|py| Ok(to_py(py, &desc)?.unbind()))
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_tool_provider, m)?)?;
    m.add_function(wrap_pyfunction!(remove_tool_provider, m)?)?;
    m.add_function(wrap_pyfunction!(register_tool, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_stdio, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_streamable_http, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_mcp, m)?)?;
    m.add_function(wrap_pyfunction!(register_a2a, m)?)?;
    Ok(())
}
