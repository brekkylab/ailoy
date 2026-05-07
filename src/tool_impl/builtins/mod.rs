mod apply_patch;
mod bash;
mod edit;
mod python_repl;
mod read;
mod web_search;
mod write;

use apply_patch::build_apply_patch_tool;
use bash::build_bash_tool;
use edit::build_edit_tool;
use python_repl::build_python_repl_tool;
use read::build_read_tool;
use web_search::build_web_search_tool;
use write::build_write_tool;

use crate::tool::{BuiltinToolProviderElem, ToolFactory};

pub async fn make_builtin_tool_factory(
    provider: &BuiltinToolProviderElem,
) -> anyhow::Result<ToolFactory> {
    match provider {
        BuiltinToolProviderElem::Bash {} => build_bash_tool().await,
        BuiltinToolProviderElem::PythonRepl {} => build_python_repl_tool().await,
        BuiltinToolProviderElem::WebSearch {} => build_web_search_tool().await,
        BuiltinToolProviderElem::Read {} => build_read_tool().await,
        BuiltinToolProviderElem::Write {} => build_write_tool().await,
        BuiltinToolProviderElem::Edit {} => build_edit_tool().await,
        BuiltinToolProviderElem::ApplyPatch {} => build_apply_patch_tool().await,
    }
}
