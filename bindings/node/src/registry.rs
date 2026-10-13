//! The process-wide registries an agent resolves its tools from: tool providers (every built-in
//! tool) keyed by name, and agent providers, each naming the tool provider an agent is built
//! against ([`AgentBuilder.agentProvider`](crate::agent), `'default'` if unset). Models have
//! their own registry, keyed by org: see [`crate::model`].
//!
//! Every tool function takes the entry it works on as `{ provider }` in its last argument,
//! defaulting to `'default'`. Registering into a missing entry throws rather than creating it, so a
//! misspelt name is an error; `add*Provider` creates one.
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
use napi::{
    Env,
    bindgen_prelude::{PromiseRaw, Unknown, within_runtime_if_available},
};
use napi_derive::napi;
use virtx_node::console::{promise, thrown};

use crate::{
    convert::{Json, from_js},
    error::{self, Result, invalid},
    tool::{Callback, tool_func},
};

fn not_registered(kind: &str, name: &str) -> napi::Error<String> {
    error::ailoy(format!("{kind} '{name}' not registered"))
}

fn parse_url(url: &str) -> Result<url::Url> {
    url::Url::parse(url).map_err(|e| invalid(format!("{url:?}: {e}")))
}

fn provider(options: Option<ProviderOptions>) -> String {
    options
        .and_then(|o| o.provider)
        .unwrap_or_else(|| "default".to_string())
}

#[napi(object)]
pub struct ProviderOptions {
    pub provider: Option<String>,
}

#[napi(object)]
pub struct ToolProviderOptions {
    pub builtins: Option<bool>,
}

/// Add a tool provider under `name`: every built-in tool, or none of them when `builtins` is
/// `false`. Replaces an entry of the same name.
#[napi]
pub fn add_tool_provider(name: String, options: Option<ToolProviderOptions>) {
    let provider = if options.and_then(|o| o.builtins).unwrap_or(true) {
        ToolProvider::new()
    } else {
        ToolProvider::empty()
    };
    get_tool_providers_mut().insert(name, provider);
}

/// Register `func` as the tool `desc` describes, and hand `desc` back.
///
/// `desc.name` is what the model calls and what the entry is keyed by. See [`crate::tool`] for
/// how `func` is called.
#[napi(ts_return_type = "ToolDesc")]
pub fn register_tool(
    env: &Env,
    #[napi(ts_arg_type = "ToolDesc")] desc: Unknown<'_>,
    #[napi(ts_arg_type = "(args: any) => unknown")] func: Callback,
    options: Option<ProviderOptions>,
) -> Result<Json<ToolDesc>> {
    let parsed: ToolDesc = from_js(env, desc, "a tool description")?;
    let provider = provider(options);
    let mut registry = get_tool_providers_mut();
    let tp = registry
        .get_mut(&provider)
        .ok_or_else(|| not_registered("tool_provider", &provider))?;
    tp.insert_func(parsed.name.clone(), tool_func(func));
    Ok(Json(parsed))
}

/// Start the MCP server `command args…` on this host and register each of its tools as
/// `{prefix}__{name}`, settling with their descriptions.
///
/// The server runs outside any console, with this process's access — register only servers
/// you trust.
#[napi(ts_return_type = "Promise<Array<ToolDesc>>")]
pub fn register_mcp_stdio<'env>(
    env: &'env Env,
    prefix: String,
    command: String,
    args: Option<Vec<String>>,
    options: Option<ProviderOptions>,
) -> napi::Result<PromiseRaw<'env, Json<Vec<ToolDesc>>>> {
    let provider = provider(options);
    let args = args.unwrap_or_default();
    promise(env, async move {
        register_mcp_stdio_rs(&provider, &prefix, &command, &args)
            .await
            .map(Json)
            .map_err(error::anyhow)
    })
}

/// Connect to the streamable-HTTP MCP server at `url` and register each of its tools as
/// `{prefix}__{name}`, settling with their descriptions.
#[napi(ts_return_type = "Promise<Array<ToolDesc>>")]
pub fn register_mcp_streamable_http<'env>(
    env: &'env Env,
    prefix: String,
    url: String,
    options: Option<ProviderOptions>,
) -> napi::Result<PromiseRaw<'env, Json<Vec<ToolDesc>>>> {
    let provider = provider(options);
    promise(env, async move {
        register_mcp_streamable_http_rs(&provider, &prefix, &url)
            .await
            .map(Json)
            .map_err(error::anyhow)
    })
}

/// Drop the tools registered under `prefix` and end that server's session. Answers with how
/// many were removed.
#[napi]
pub fn unregister_mcp(prefix: String, options: Option<ProviderOptions>) -> Result<u32> {
    let provider = provider(options);
    // Off the runtime is fine for the registry, but ending the session cancels a task on it.
    within_runtime_if_available(|| unregister_mcp_rs(&provider, &prefix))
        .map(|n| n as u32)
        .map_err(error::anyhow)
}

/// Fetch the A2A agent card at `url` and register the agent as the tool `name`, settling with
/// its description.
#[napi(ts_return_type = "Promise<ToolDesc>")]
pub fn register_a2a<'env>(
    env: &'env Env,
    name: String,
    url: String,
    options: Option<ProviderOptions>,
) -> napi::Result<PromiseRaw<'env, Json<ToolDesc>>> {
    let url = parse_url(&url).map_err(|e| thrown(env, e))?;
    let provider = provider(options);
    promise(env, async move {
        register_a2a_rs(&provider, &name, url)
            .await
            .map(Json)
            .map_err(error::anyhow)
    })
}

#[napi(object)]
pub struct AgentProviderOptions {
    pub tool_provider: Option<String>,
}

/// Name a tool provider as the agent provider `name`. It need not exist yet; it has to when an
/// agent is built against it.
#[napi]
pub fn add_agent_provider(name: String, options: Option<AgentProviderOptions>) {
    let tool_provider = options
        .and_then(|o| o.tool_provider)
        .unwrap_or_else(|| "default".to_string());
    get_agent_providers_mut().insert(name, AgentProvider::new(tool_provider));
}
