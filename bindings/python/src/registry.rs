//! The process-wide registries an agent resolves its model and tools from.
//!
//! ailoy keeps three, each keyed by name and each with a `"default"` entry from the start:
//! language-model providers (seeded from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … on first
//! use), tool providers (every built-in tool), and agent providers, which pair one of each
//! by name. An agent names the agent provider it is built against — `"default"` unless
//! [`AgentBuilder.agent_provider`](crate::agent) says otherwise.
//!
//! Every function here takes the registry entry it works on as `provider=`, defaulting to
//! `"default"`, and one that registers into an entry that does not exist raises rather than
//! making it: `add_*_provider` makes one, so a misspelt name is an error and not a second,
//! empty registry.
//!
//! The registering functions hand back the tool descriptions they registered, as the Rust
//! ones do — pass them to `AgentBuilder.tool`/`tools` to put them in front of the model.

use ailoy::{
    agent::{AgentProvider, get_agent_providers_mut},
    lang_model::{
        LangModelAPISchema, LangModelProvider, LangModelProviderElem, get_lm_providers_mut,
    },
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

/// Add an empty language-model provider under `name`, or one seeded from the environment as
/// `"default"` is when `from_env`. Replaces an entry of the same name.
#[pyfunction]
#[pyo3(signature = (name, *, from_env = false))]
fn add_lang_model_provider(name: String, from_env: bool) {
    let provider = if from_env {
        LangModelProvider::default()
    } else {
        LangModelProvider::new()
    };
    get_lm_providers_mut().insert(name, provider);
}

/// Serve the models `pattern` matches — an exact name, or a glob with `*` and `?` — from the
/// API at `url`, spoken in `schema`: `"chat_completion"`, `"openai"`, `"anthropic"`,
/// `"gemini"` or `"bedrock"`.
#[pyfunction]
#[pyo3(signature = (pattern, schema, url, api_key = None, *, provider = "default".to_string()))]
fn register_lang_model(
    pattern: String,
    schema: Bound<'_, PyAny>,
    url: String,
    api_key: Option<String>,
    provider: String,
) -> PyResult<()> {
    let schema: LangModelAPISchema = from_py(&schema, "an API schema")?;
    let url = parse_url(&url)?;
    let mut registry = get_lm_providers_mut();
    let lmp = registry
        .get_mut(&provider)
        .ok_or_else(|| not_registered("lang_model_provider", &provider))?;
    lmp.insert(
        pattern,
        LangModelProviderElem::API {
            schema,
            url,
            api_key,
        },
    );
    Ok(())
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

/// Pair a language-model provider and a tool provider, by name, as the agent provider
/// `name`. Neither has to exist yet; both have to when an agent is built against it.
#[pyfunction]
#[pyo3(signature = (name, *, lang_model_provider = "default".to_string(), tool_provider = "default".to_string()))]
fn add_agent_provider(name: String, lang_model_provider: String, tool_provider: String) {
    get_agent_providers_mut().insert(name, AgentProvider::new(lang_model_provider, tool_provider));
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add_lang_model_provider, m)?)?;
    m.add_function(wrap_pyfunction!(register_lang_model, m)?)?;
    m.add_function(wrap_pyfunction!(add_tool_provider, m)?)?;
    m.add_function(wrap_pyfunction!(register_tool, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_stdio, m)?)?;
    m.add_function(wrap_pyfunction!(register_mcp_streamable_http, m)?)?;
    m.add_function(wrap_pyfunction!(unregister_mcp, m)?)?;
    m.add_function(wrap_pyfunction!(register_a2a, m)?)?;
    m.add_function(wrap_pyfunction!(add_agent_provider, m)?)?;
    Ok(())
}
