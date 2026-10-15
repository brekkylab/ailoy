//! The process-wide registries an agent resolves its tools from: tool providers (every built-in
//! tool) keyed by name, and agent providers, each naming the tool provider an agent is built
//! against ([`AgentBuilder.agent_provider`](crate::agent), `"default"` if unset). Models have
//! their own registry, keyed by org: see [`crate::model`].
//!
//! Every tool function takes the entry it works on as `provider=`, defaulting to `"default"`.
//! Registering into a missing entry raises rather than creating it, so a misspelt name is an
//! error; `add_*_provider` creates one.
//!
//! The registering functions return the tool descriptions they registered, to pass to
//! `AgentBuilder.tool`/`tools`.

use ailoy::{
    agent::{AgentProvider, get_agent_providers_mut},
    tool::{
        ToolDesc, ToolProvider, get_tool_providers_mut, register_a2a as register_a2a_rs,
        register_mcp_stdio as register_mcp_stdio_rs,
        register_mcp_streamable_http as register_mcp_streamable_http_rs,
        unregister_mcp as unregister_mcp_rs,
    },
};
use pyo3::{exceptions::PyValueError, prelude::*};
use pyo3_async_runtimes::tokio::future_into_py;

use crate::{
    convert::{from_py, to_py},
    error::{self, AiloyError},
    tool::tool_func,
};

fn not_registered(kind: &str, name: &str) -> PyErr {
    AiloyError::new_err(format!("{kind} '{name}' not registered"))
}

fn parse_url(url: &str) -> PyResult<url::Url> {
    url::Url::parse(url).map_err(|e| PyValueError::new_err(format!("{url:?}: {e}")))
}

/// Descs as the list of dicts the builder takes back.
fn descs_to_py(py: Python<'_>, descs: &[ToolDesc]) -> PyResult<Py<PyAny>> {
    Ok(to_py(py, &descs)?.unbind())
}

/// Add a tool provider under `name`: every built-in tool, or none of them when not
/// `builtins`. Replaces an entry of the same name.
#[pyfunction]
#[pyo3(signature = (name, *, builtins = true))]
fn add_tool_provider(name: String, builtins: bool) {
    let provider = if builtins {
        ToolProvider::new()
    } else {
        ToolProvider::empty()
    };
    get_tool_providers_mut().insert(name, provider);
}

/// Register `func` as the tool `desc` describes, and hand `desc` back.
///
/// `desc` is `{"name": ..., "description": ..., "parameters": <JSON schema>}`; the name is
/// what the model calls and what the entry is keyed by. See [`crate::tool`] for how `func`
/// is called.
#[pyfunction]
#[pyo3(signature = (desc, func, *, provider = "default".to_string()))]
fn register_tool<'py>(
    desc: Bound<'py, PyAny>,
    func: Py<PyAny>,
    provider: String,
) -> PyResult<Bound<'py, PyAny>> {
    let py = desc.py();
    if !func.bind(py).is_callable() {
        return Err(PyValueError::new_err("a tool's func has to be callable"));
    }
    let parsed: ToolDesc = from_py(&desc, "a tool description")?;
    let mut registry = get_tool_providers_mut();
    let tp = registry
        .get_mut(&provider)
        .ok_or_else(|| not_registered("tool_provider", &provider))?;
    tp.insert_func(parsed.name.clone(), tool_func(func));
    to_py(py, &parsed)
}

/// Start the MCP server `command args…` on this host and register each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their descriptions.
///
/// The server runs outside any console, with this process's access — register only servers
/// you trust.
#[pyfunction]
#[pyo3(signature = (prefix, command, args = Vec::new(), *, provider = "default".to_string()))]
fn register_mcp_stdio(
    py: Python<'_>,
    prefix: String,
    command: String,
    args: Vec<String>,
    provider: String,
) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async move {
        let descs = register_mcp_stdio_rs(&provider, &prefix, &command, &args)
            .await
            .map_err(error::anyhow)?;
        Python::attach(|py| descs_to_py(py, &descs))
    })
}

/// Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
/// `{prefix}__{name}`. Awaitable; answers with their descriptions.
#[pyfunction]
#[pyo3(signature = (prefix, url, *, provider = "default".to_string()))]
fn register_mcp_streamable_http(
    py: Python<'_>,
    prefix: String,
    url: String,
    provider: String,
) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async move {
        let descs = register_mcp_streamable_http_rs(&provider, &prefix, &url)
            .await
            .map_err(error::anyhow)?;
        Python::attach(|py| descs_to_py(py, &descs))
    })
}

/// Drop the tools registered under `prefix` and end that server's session. Answers with how
/// many were removed.
#[pyfunction]
#[pyo3(signature = (prefix, *, provider = "default".to_string()))]
fn unregister_mcp(prefix: String, provider: String) -> PyResult<usize> {
    // Off the runtime is fine for the registry, but ending the session cancels a task on it.
    let _entered = pyo3_async_runtimes::tokio::get_runtime().enter();
    unregister_mcp_rs(&provider, &prefix).map_err(error::anyhow)
}

/// Fetch the A2A agent card at `url` and register the agent as the tool `name`. Awaitable;
/// answers with its description.
#[pyfunction]
#[pyo3(signature = (name, url, *, provider = "default".to_string()))]
fn register_a2a(
    py: Python<'_>,
    name: String,
    url: String,
    provider: String,
) -> PyResult<Bound<'_, PyAny>> {
    let url = parse_url(&url)?;
    future_into_py(py, async move {
        let desc = register_a2a_rs(&provider, &name, url)
            .await
            .map_err(error::anyhow)?;
        Python::attach(|py| Ok(to_py(py, &desc)?.unbind()))
    })
}

/// Name a tool provider as the agent provider `name`. It need not exist yet; it has to when
/// an agent is built against it.
#[pyfunction]
#[pyo3(signature = (name, *, tool_provider = "default".to_string()))]
fn add_agent_provider(name: String, tool_provider: String) {
    get_agent_providers_mut().insert(name, AgentProvider::new(tool_provider));
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_tool_provider, m)?)?;
    m.add_function(wrap_pyfunction!(register_tool, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_stdio, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_streamable_http, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_mcp, m)?)?;
    m.add_function(wrap_pyfunction!(register_a2a, m)?)?;
    m.add_function(wrap_pyfunction!(add_agent_provider, m)?)?;
    Ok(())
}
