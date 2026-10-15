mod apply_patch;
mod edit;
mod imgread;
mod read;
mod read_file;
mod shell;
mod web_fetch;
mod web_search;
mod write;

use std::sync::Arc;

pub use apply_patch::*;
pub use edit::*;
pub use imgread::*;
pub use read::*;
pub use read_file::*;
pub use shell::*;
pub use web_fetch::*;
pub use web_search::*;
pub use write::*;

use crate::tool::ToolFactory;

/// Build `(name, factory)` pairs for every built-in tool. The factory takes the
/// [`ToolDesc`](crate::tool::ToolDesc) requested by an agent spec and returns the
/// [`ToolFunc`](crate::tool::ToolFunc) to bind to it.
pub fn get_builtin_tool_factories() -> Vec<(&'static str, ToolFactory)> {
    let shell = get_shell_tool_func();
    let read = get_read_tool_func();
    let read_file = get_read_file_tool_func();
    let imgread = get_imgread_tool_func();
    let write = get_write_tool_func();
    let edit = get_edit_tool_func();
    let apply_patch = get_apply_patch_tool_func();

    vec![
        ("shell", Arc::new(move |_| Ok(shell.clone()))),
        ("read", Arc::new(move |_| Ok(read.clone()))),
        ("read_file", Arc::new(move |_| Ok(read_file.clone()))),
        ("imgread", Arc::new(move |_| Ok(imgread.clone()))),
        ("write", Arc::new(move |_| Ok(write.clone()))),
        ("edit", Arc::new(move |_| Ok(edit.clone()))),
        ("apply_patch", Arc::new(move |_| Ok(apply_patch.clone()))),
        ("web_search", Arc::new(get_web_search_tool_factory(vec![]))),
        ("web_fetch", Arc::new(get_web_fetch_tool_factory())),
    ]
}
