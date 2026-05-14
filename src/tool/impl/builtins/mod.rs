mod apply_patch;
mod bash;
mod edit;
mod glob;
mod grep;
mod python_repl;
mod read;
mod web_search;
mod write;

use std::sync::Arc;

pub use apply_patch::*;
pub use bash::*;
pub use edit::*;
pub use glob::*;
pub use grep::*;
pub use python_repl::*;
pub use read::*;
pub use web_search::*;
pub use write::*;

use crate::tool::{ToolDesc, ToolFunc};

type BuiltinFactory = Arc<dyn Fn(&ToolDesc) -> ToolFunc + Send + Sync + 'static>;

/// Build `(name, factory)` pairs for every built-in tool. The factory takes
/// the [`ToolDesc`] requested by an agent spec and returns the [`ToolFunc`]
/// to bind to it.
pub fn get_builtin_tool_factories() -> Vec<(&'static str, BuiltinFactory)> {
    let bash = get_bash_tool_func();
    let read = get_read_tool_func();
    let write = get_write_tool_func();
    let edit = get_edit_tool_func();
    let apply_patch = get_apply_patch_tool_func();
    let glob = get_glob_tool_func();
    let grep = get_grep_tool_func();

    vec![
        ("bash", Arc::new(move |_| bash.clone())),
        ("python_repl", Arc::new(get_python_repl_tool_factory())),
        ("read", Arc::new(move |_| read.clone())),
        ("write", Arc::new(move |_| write.clone())),
        ("edit", Arc::new(move |_| edit.clone())),
        ("apply_patch", Arc::new(move |_| apply_patch.clone())),
        ("glob", Arc::new(move |_| glob.clone())),
        ("grep", Arc::new(move |_| grep.clone())),
        ("web_search", Arc::new(get_web_search_tool_factory())),
    ]
}
