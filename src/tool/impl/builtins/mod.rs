mod apply_patch;
mod edit;
mod imgread;
mod read;
mod shell;
mod web_fetch;
mod web_search;
mod write;

use std::sync::Arc;

pub use apply_patch::*;
pub use edit::*;
pub use imgread::*;
pub use read::*;
pub use shell::*;
pub use web_fetch::*;
pub use web_search::*;
pub use write::*;

use crate::tool::{ToolDesc, ToolFunc};

type BuiltinFactory = Arc<dyn Fn(&ToolDesc) -> ToolFunc + Send + Sync + 'static>;

/// Build `(name, factory)` pairs for every built-in tool. The factory takes
/// the [`ToolDesc`] requested by an agent spec and returns the [`ToolFunc`]
/// to bind to it.
pub fn get_builtin_tool_factories() -> Vec<(&'static str, BuiltinFactory)> {
    let shell = get_shell_tool_func();
    let read_claude = get_read_tool_func(ReadStyle::Claude);
    let read_gemini = get_read_tool_func(ReadStyle::Gemini);
    let imgread = get_imgread_tool_func();
    let write = get_write_tool_func();
    let edit = get_edit_tool_func();
    let apply_patch = get_apply_patch_tool_func();

    vec![
        ("shell", Arc::new(move |_| shell.clone())),
        (
            "read",
            Arc::new(move |desc| match read_style_of(desc) {
                ReadStyle::Claude => read_claude.clone(),
                ReadStyle::Gemini => read_gemini.clone(),
            }),
        ),
        ("imgread", Arc::new(move |_| imgread.clone())),
        ("write", Arc::new(move |_| write.clone())),
        ("edit", Arc::new(move |_| edit.clone())),
        ("apply_patch", Arc::new(move |_| apply_patch.clone())),
        ("web_search", Arc::new(get_web_search_tool_factory(vec![]))),
        ("web_fetch", Arc::new(get_web_fetch_tool_factory())),
    ]
}
