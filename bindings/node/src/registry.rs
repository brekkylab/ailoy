//! The process-wide registry an agent builds its tools from: a provider per tool name, seeded
//! with every built-in tool, as models have one per org (see [`crate::model`]).
//!
//! The registering functions return the tool descriptions they registered, to pass to
//! `AgentBuilder.tool`/`tools`.

use ailoy::tool::{
    ToolDesc, register_mcp_stdio as register_mcp_stdio_rs,
    register_mcp_streamable_http as register_mcp_streamable_http_rs,
    remove_tool_provider as remove_tool_provider_rs, set_tool_provider as set_tool_provider_rs,
    unregister_mcp as unregister_mcp_rs,
};
use napi::{
    Env,
    bindgen_prelude::{PromiseRaw, Unknown, within_runtime_if_available},
};
use napi_derive::napi;

use crate::{
    convert::{Json, from_js},
    error::{self, Result, promise},
    tool::{Callback, tool_func},
};

/// Build the tool `name` by calling `func`, replacing the provider it had. See [`crate::tool`]
/// for how `func` is called.
#[napi]
pub fn set_tool_provider(
    name: String,
    #[napi(ts_arg_type = "(args: any) => unknown")] func: Callback,
) {
    set_tool_provider_rs(name, tool_func(func));
}

/// Stop building the tool `name`. Answers whether it had a provider.
#[napi]
pub fn remove_tool_provider(name: String) -> bool {
    remove_tool_provider_rs(&name).is_some()
}

/// Build the tool `desc` describes by calling `func`, and hand `desc` back.
///
/// `desc.name` is what the model calls and what the provider is set under.
#[napi(ts_return_type = "ToolDesc")]
pub fn register_tool(
    env: &Env,
    #[napi(ts_arg_type = "ToolDesc")] desc: Unknown<'_>,
    #[napi(ts_arg_type = "(args: any) => unknown")] func: Callback,
) -> Result<Json<ToolDesc>> {
    let parsed: ToolDesc = from_js(env, desc, "a tool description")?;
    set_tool_provider_rs(parsed.name.clone(), tool_func(func));
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
) -> napi::Result<PromiseRaw<'env, Json<Vec<ToolDesc>>>> {
    let args = args.unwrap_or_default();
    promise(env, async move {
        register_mcp_stdio_rs(&prefix, &command, &args)
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
) -> napi::Result<PromiseRaw<'env, Json<Vec<ToolDesc>>>> {
    promise(env, async move {
        register_mcp_streamable_http_rs(&prefix, &url)
            .await
            .map(Json)
            .map_err(error::anyhow)
    })
}

/// Drop the tools registered under `prefix` and end that server's session. Answers with how
/// many were removed.
#[napi]
pub fn unregister_mcp(prefix: String) -> u32 {
    // Off the runtime is fine for the registry, but ending the session cancels a task on it.
    within_runtime_if_available(|| unregister_mcp_rs(&prefix)) as u32
}
