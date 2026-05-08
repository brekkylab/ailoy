//! Convenience constructors for building [`Tool`] instances directly,
//! without going through the [`ToolFactory`] / [`ToolProvider`] lifecycle.
//!
//! Use these when you know the target configuration up-front and don't need
//! the factory lifecycle — for example, in tests or in one-shot scripts.

use url::Url;

use crate::{
    agent::AgentSpec,
    tool::{BuiltinToolProviderElem, Tool},
};

/// Build a [`Tool`] for the given [`BuiltinToolProviderElem`].
pub async fn make_builtin_tool(provider: &BuiltinToolProviderElem) -> anyhow::Result<Tool> {
    let factory = crate::tool_impl::make_builtin_tool_factory(provider).await?;
    Ok(factory.make(&AgentSpec::new("")))
}

/// Build a [`Tool`] that delegates to a remote A2A agent.
///
/// Eagerly fetches the agent card from `{url}/.well-known/agent-card.json` so
/// that the tool name and description are known at construction time.
pub async fn make_a2a_tool(url: Url) -> anyhow::Result<Tool> {
    let factory = crate::tool_impl::make_a2a_tool_factory(url).await?;
    Ok(factory.make(&AgentSpec::new("")))
}
