//! The tools of this process's provider, which an agent makes the tools it names through: every
//! built-in tool from the start, as models have a provider per org (see [`crate::model`]).
//!
//! The registering functions return the names of the tools they provide, to pass to
//! `AgentBuilder.tool`/`tools`.

use ailoy_runner::tool::{
    ToolDesc, WebSearchProvider, get_web_search_provider as get_web_search_provider_rs,
    register_mcp_stdio as register_mcp_stdio_rs,
    register_mcp_streamable_http as register_mcp_streamable_http_rs,
    remove_tool_provider as remove_tool_provider_rs, set_tool_provider as set_tool_provider_rs,
    set_web_search_provider as set_web_search_provider_rs, unregister_mcp as unregister_mcp_rs,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_async_runtimes::tokio::future_into_py;

use crate::{convert::from_py, error, tool::tool_func};

fn ensure_callable(py: Python<'_>, func: &Py<PyAny>) -> PyResult<()> {
    if func.bind(py).is_callable() {
        Ok(())
    } else {
        Err(PyValueError::new_err("a tool's func has to be callable"))
    }
}

/// Stop providing the tool `name`. Answers whether it was provided.
#[pyfunction]
fn remove_tool_provider(name: String) -> bool {
    remove_tool_provider_rs(&name)
}

/// Provide the tool `desc` describes by calling `func`, in place of one of its name, and answer
/// its name. See [`crate::tool`] for how `func` is called.
///
/// `desc` is `{"name": ..., "description": ..., "parameters": <JSON schema>}`; the name is
/// what the model calls and what a spec lists.
#[pyfunction]
fn register_tool(desc: Bound<'_, PyAny>, func: Py<PyAny>) -> PyResult<String> {
    ensure_callable(desc.py(), &func)?;
    let parsed: ToolDesc = from_py(&desc, "a tool description")?;
    let name = parsed.name.clone();
    set_tool_provider_rs((parsed, tool_func(func)));
    Ok(name)
}

/// Start the MCP server `command args…` on this host and provide each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their names.
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
        register_mcp_stdio_rs(&prefix, &command, &args)
            .await
            .map_err(error::anyhow)
    })
}

/// Connect to the streamable-HTTP MCP server at `url` and provide each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their names.
#[pyfunction]
fn register_mcp_streamable_http(
    py: Python<'_>,
    prefix: String,
    url: String,
) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async move {
        register_mcp_streamable_http_rs(&prefix, &url)
            .await
            .map_err(error::anyhow)
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

/// Search with `provider` — `"web"`, `"brave"`, `"tavily"`, `"exa"` or `"naver"` — in every
/// `web_search` tool built from now on.
#[pyfunction]
fn set_web_search_provider(provider: Bound<'_, PyAny>) -> PyResult<()> {
    set_web_search_provider_rs(from_py::<WebSearchProvider>(
        &provider,
        "a web search provider",
    )?);
    Ok(())
}

/// The provider `web_search` tools are built with.
#[pyfunction]
fn get_web_search_provider() -> String {
    get_web_search_provider_rs().to_string()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_web_search_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_web_search_provider, m)?)?;
    m.add_function(wrap_pyfunction!(remove_tool_provider, m)?)?;
    m.add_function(wrap_pyfunction!(register_tool, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_stdio, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_streamable_http, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_mcp, m)?)?;
    Ok(())
}
