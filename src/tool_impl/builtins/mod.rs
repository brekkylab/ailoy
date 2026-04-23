mod bash;
mod python_repl;
mod web_search;

use bash::build_bash_tool;
use python_repl::build_python_repl_tool;
use web_search::build_web_search_tool;

use crate::tool::{BuiltinToolProvider, ToolFactory};

pub async fn make_builtin_tool(provider: &BuiltinToolProvider) -> anyhow::Result<ToolFactory> {
    match provider {
        BuiltinToolProvider::Bash {} => build_bash_tool().await,
        BuiltinToolProvider::PythonRepl {} => build_python_repl_tool().await,
        BuiltinToolProvider::WebSearch {} => build_web_search_tool().await,
    }
}
