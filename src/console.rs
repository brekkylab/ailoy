//! The console a tool runs commands in.

/// The console a tool runs commands in — `cortex`'s own type, re-exported rather
/// than wrapped.
///
/// Re-exported here for two reasons. It is the path [`tool_func!`](crate::tool_func)
/// writes into its expansion, and a `#[macro_export]`ed macro can only name types
/// through `$crate` — spelling `::cortex::..` there would make every crate that
/// writes a tool depend on `cortex` under that exact name. And it is where a caller
/// looks: the console is not a tool-only concern — it is also what
/// [`AgentBuilder::console`](crate::agent::AgentBuilder::console) lends out and what
/// [`AgentState`](crate::agent::AgentState) holds — so it sits in a module of its own
/// rather than under [`tool`](crate::tool).
///
/// Nothing is added to it. A tool calls [`Console::exec`], [`Console::read`] and
/// [`Console::write`] as cortex defines them — argv in, bytes out, milliseconds, and
/// a timeout that arrives as an error rather than a flag. ailoy ships no convenience
/// layer over that on purpose: a helper here would become a second interface that
/// tool authors have to learn and that has to be kept in step with cortex's.
pub use cortex::console::Console;
