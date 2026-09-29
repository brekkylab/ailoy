//! The process-wide registries an agent resolves its model and tools from.
//!
//! ailoy keeps three, each keyed by name and each with a `'default'` entry from the start:
//! language-model providers (seeded from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, … on first
//! use), tool providers (every built-in tool), and agent providers, which pair one of each
//! by name. An agent names the agent provider it is built against — `'default'` unless
//! [`AgentBuilder.agentProvider`](crate::agent) says otherwise.
//!
//! Every function here takes the registry entry it works on as `{ provider }` in its last
//! argument, defaulting to `'default'`, and one that registers into an entry that does not
//! exist throws rather than making it: `add*Provider` makes one, so a misspelt name is an
//! error and not a second, empty registry.
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
use cortex_node::console::{promise, thrown};
use napi::{
    Env,
    bindgen_prelude::{PromiseRaw, Unknown, within_runtime_if_available},
};
use napi_derive::napi;

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
pub struct LangModelProviderOptions {
    pub from_env: Option<bool>,
}

/// Add an empty language-model provider under `name`, or one seeded from the environment as
/// `'default'` is when `fromEnv`. Replaces an entry of the same name.
#[napi]
pub fn add_lang_model_provider(name: String, options: Option<LangModelProviderOptions>) {
    let provider = if options.and_then(|o| o.from_env).unwrap_or(false) {
        LangModelProvider::default()
    } else {
        LangModelProvider::new()
    };
    get_lm_providers_mut().insert(name, provider);
}

#[napi(object)]
pub struct RegisterLangModelOptions {
    pub api_key: Option<String>,
    pub provider: Option<String>,
}

/// Serve the models `pattern` matches — an exact name, or a glob with `*` and `?` — from the
/// API at `url`, spoken in `schema`: `'chat_completion'`, `'openai'`, `'anthropic'`,
/// `'gemini'` or `'bedrock'`.
#[napi]
pub fn register_lang_model(
    env: &Env,
    pattern: String,
    #[napi(ts_arg_type = "LangModelAPISchema")] schema: Unknown<'_>,
    url: String,
    options: Option<RegisterLangModelOptions>,
) -> Result<()> {
    let schema: LangModelAPISchema = from_js(env, schema, "an API schema")?;
    let url = parse_url(&url)?;
    let (api_key, provider) = match options {
        Some(o) => (
            o.api_key,
            o.provider.unwrap_or_else(|| "default".to_string()),
        ),
        None => (None, "default".to_string()),
    };
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
/// `desc` is `{ name, description, parameters: <JSON schema> }`; the name is what the model
/// calls and what the entry is keyed by. See [`crate::tool`] for how `func` is called.
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
    pub lang_model_provider: Option<String>,
    pub tool_provider: Option<String>,
}

/// Pair a language-model provider and a tool provider, by name, as the agent provider
/// `name`. Neither has to exist yet; both have to when an agent is built against it.
#[napi]
pub fn add_agent_provider(name: String, options: Option<AgentProviderOptions>) {
    let (lm, tp) = match options {
        Some(o) => (o.lang_model_provider, o.tool_provider),
        None => (None, None),
    };
    get_agent_providers_mut().insert(
        name,
        AgentProvider::new(
            lm.unwrap_or_else(|| "default".to_string()),
            tp.unwrap_or_else(|| "default".to_string()),
        ),
    );
}
