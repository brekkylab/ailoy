mod a2a;
mod builtins;
mod subagent;

// Crate-internal API — used by ToolProvider and tests
pub(crate) use a2a::{get_a2a_tool_desc, get_a2a_tool_func};
pub(crate) use builtins::*;
pub(crate) use subagent::{get_subagent_tool_desc, get_subagent_tool_func};
