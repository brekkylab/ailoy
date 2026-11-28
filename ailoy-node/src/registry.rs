//! The tools of this process's provider, which an agent makes the tools it names through: every
//! built-in tool from the start, as models have a provider per org (see [`crate::model`]).
//!
//! The registering functions return the names of the tools they provide, to list in a spec's
//! `tools`.

use ailoy_runner::{
    agent::default_provider,
    tool::{
        MCPServer, MCPTransport, ToolDesc, WebSearchProvider,
        get_web_search_provider as get_web_search_provider_rs,
        set_web_search_provider as set_web_search_provider_rs,
    },
};
use napi::{
    Env,
    bindgen_prelude::{PromiseRaw, Unknown, within_runtime_if_available},
};
use napi_derive::napi;

use crate::{
    convert::from_js,
    error::{self, Result, promise},
    tool::{Callback, tool_func},
};

/// Stop providing the tool `name`. Answers whether it was provided.
#[napi]
pub fn remove_tool_provider(name: String) -> bool {
    default_provider().write().unwrap().tools.remove(&name)
}

/// Provide the tool `desc` describes by calling `func`, in place of one of its name, and answer
/// its name. See [`crate::tool`] for how `func` is called.
///
/// `desc.name` is what the model calls and what a spec lists.
#[napi]
pub fn register_tool(
    env: &Env,
    #[napi(ts_arg_type = "ToolDesc")] desc: Unknown<'_>,
    #[napi(ts_arg_type = "(args: any) => unknown")] func: Callback,
) -> Result<String> {
    let parsed: ToolDesc = from_js(env, desc, "a tool description")?;
    let name = parsed.name.clone();
    default_provider()
        .write()
        .unwrap()
        .tools
        .set((parsed, tool_func(func)));
    Ok(name)
}

/// Start the MCP server `command args…` on this host and provide each of its tools as
/// `{prefix}__{name}`, in place of a server there under it, settling with their names.
///
/// The server runs outside any console, with this process's access — register only servers
/// you trust.
#[napi(ts_return_type = "Promise<Array<string>>")]
pub fn register_mcp_stdio<'env>(
    env: &'env Env,
    prefix: String,
    command: String,
    args: Option<Vec<String>>,
) -> napi::Result<PromiseRaw<'env, Vec<String>>> {
    let args = args.unwrap_or_default();
    promise(env, async move {
        let transport = MCPTransport::Stdio { command, args };
        let server = MCPServer::connect(&prefix, transport)
            .await
            .map_err(error::anyhow)?;
        let mut provider = default_provider().write().unwrap();
        provider
            .tools
            .remove_by_prefix(&MCPServer::name_prefix(&prefix));
        Ok(provider.tools.set(server))
    })
}

/// Connect to the streamable-HTTP MCP server at `url` and provide each of its tools as
/// `{prefix}__{name}`, in place of a server there under it, settling with their names.
#[napi(ts_return_type = "Promise<Array<string>>")]
pub fn register_mcp_streamable_http<'env>(
    env: &'env Env,
    prefix: String,
    url: String,
) -> napi::Result<PromiseRaw<'env, Vec<String>>> {
    promise(env, async move {
        let transport = MCPTransport::StreamableHTTP { url };
        let server = MCPServer::connect(&prefix, transport)
            .await
            .map_err(error::anyhow)?;
        let mut provider = default_provider().write().unwrap();
        provider
            .tools
            .remove_by_prefix(&MCPServer::name_prefix(&prefix));
        Ok(provider.tools.set(server))
    })
}

/// Drop the tools registered under `prefix` and end that server's session. Answers with how
/// many were removed.
#[napi]
pub fn unregister_mcp(prefix: String) -> u32 {
    // Off the runtime is fine for the provider, but ending the session cancels a task on it.
    let name_prefix = MCPServer::name_prefix(&prefix);
    within_runtime_if_available(|| {
        default_provider()
            .write()
            .unwrap()
            .tools
            .remove_by_prefix(&name_prefix)
    }) as u32
}

/// Search with `provider` — `'web'`, `'brave'`, `'tavily'`, `'exa'` or `'naver'` — in every
/// `web_search` tool built from now on.
#[napi]
pub fn set_web_search_provider(
    env: &Env,
    #[napi(ts_arg_type = "WebSearchProvider")] provider: Unknown<'_>,
) -> Result<()> {
    set_web_search_provider_rs(from_js::<WebSearchProvider>(
        env,
        provider,
        "a web search provider",
    )?);
    Ok(())
}

/// The provider `web_search` tools are built with.
#[napi(ts_return_type = "WebSearchProvider")]
pub fn get_web_search_provider() -> String {
    get_web_search_provider_rs().to_string()
}
