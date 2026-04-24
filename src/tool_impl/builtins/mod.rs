mod bash;
mod python_repl;
mod web_search;

use bash::{build_bash_tool, build_bash_tool_factory};
use python_repl::{build_python_repl_tool, build_python_repl_tool_factory};
use web_search::{build_web_search_tool, build_web_search_tool_factory};

use crate::tool::{BuiltinToolProvider, Tool, ToolFactory};

pub async fn make_builtin_tool_factory(
    provider: &BuiltinToolProvider,
) -> anyhow::Result<ToolFactory> {
    match provider {
        BuiltinToolProvider::Bash {} => build_bash_tool_factory().await,
        BuiltinToolProvider::PythonRepl {} => build_python_repl_tool_factory().await,
        BuiltinToolProvider::WebSearch {} => build_web_search_tool_factory().await,
    }
}

pub(crate) fn make_builtin_tool(provider: &BuiltinToolProvider) -> Tool {
    let with_sandbox = if cfg!(feature = "sandbox") {
        true
    } else {
        false
    };

    match provider {
        BuiltinToolProvider::Bash {} => build_bash_tool(with_sandbox),
        BuiltinToolProvider::PythonRepl {} => build_python_repl_tool(with_sandbox),
        BuiltinToolProvider::WebSearch {} => build_web_search_tool(),
    }
}
