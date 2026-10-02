//! The console a tool runs commands in.

/// `virtx`'s console type, re-exported unwrapped.
///
/// [`tool_func!`](crate::tool_func) names it through `$crate`, so tool crates need not
/// depend on `virtx` under that exact name. It lives outside [`tool`](crate::tool)
/// because [`AgentBuilder::console`](crate::agent::AgentBuilder::console) and
/// [`AgentState`](crate::agent::AgentState) use it too.
///
/// Tools call [`ConsoleClient::exec`], [`ConsoleClient::read`] and [`ConsoleClient::write`]
/// directly (argv in, bytes out, milliseconds, timeouts as errors); there is deliberately
/// no helper layer to keep in step with virtx.
pub use virtx::console::ConsoleClient;
