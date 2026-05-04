mod bash;
mod python_repl;
mod web_search;

use bash::build_bash_tool;
use python_repl::build_python_repl_tool;
use web_search::build_web_search_tool;

use crate::tool::{BuiltinToolProviderElem, ToolFactory};

pub async fn make_builtin_tool_factory(
    provider: &BuiltinToolProviderElem,
) -> anyhow::Result<ToolFactory> {
    match provider {
        BuiltinToolProviderElem::Bash {} => build_bash_tool().await,
        BuiltinToolProviderElem::PythonRepl {} => build_python_repl_tool().await,
        BuiltinToolProviderElem::WebSearch {} => build_web_search_tool().await,
    }
}
