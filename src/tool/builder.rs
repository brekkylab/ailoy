//! Convenience constructors for building [`Tool`] instances directly,
//! without going through the [`ToolFactory`] / [`crate::tool::ToolSet`] lifecycle.
//!
//! Use these when you know the target configuration up-front and don't need
//! the sandbox-aware factory indirection — for example, in tests or in
//! one-shot scripts.

use url::Url;

use crate::tool::{BuiltinToolProvider, Tool};

/// Build a [`Tool`] for the given [`BuiltinToolProvider`].
///
/// Chooses the sandbox-aware variant automatically when the `sandbox` feature
/// is enabled.
pub fn make_builtin_tool(provider: &BuiltinToolProvider) -> Tool {
    crate::tool_impl::make_builtin_tool(provider)
}

/// Build a [`Tool`] that delegates to a remote A2A agent.
///
/// Eagerly fetches the agent card from `{url}/.well-known/agent-card.json` so
/// that the tool name and description are known at construction time.
pub async fn make_a2a_tool(url: Url) -> anyhow::Result<Tool> {
    crate::tool_impl::make_a2a_tool(url).await
}
