mod a2a;
pub mod builtins;
mod subagent;

// Public API — accessible to external callers via `ailoy::tool_impl`
pub use a2a::make_a2a_tool;
pub use subagent::make_subagent_tool;

// Crate-internal API — used by ToolSet and tests
pub(crate) use a2a::make_a2a_tool_factory;
pub(crate) use builtins::make_builtin_tool;
pub(crate) use subagent::make_subagent_tool_factory;
