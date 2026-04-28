mod a2a;
mod builtins;
mod subagent;

// Crate-internal API — used by ToolSet and tests
pub(crate) use a2a::{make_a2a_tool, make_a2a_tool_factory};
pub(crate) use builtins::make_builtin_tool;
pub(crate) use subagent::{make_subagent_tool, make_subagent_tool_factory};
