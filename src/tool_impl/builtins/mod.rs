mod convert_pdf_to_md;
mod python_repl;
mod web_search;

use std::sync::Arc;

use convert_pdf_to_md::*;
#[cfg(feature = "sandbox")]
use python_repl::*;
use web_search::*;

use crate::tool::{BuiltinToolProvider, Tool};

#[cfg(not(feature = "sandbox"))]
pub async fn make_builtin_tool(provider: &BuiltinToolProvider) -> anyhow::Result<Tool> {
    match provider {
        BuiltinToolProvider::WebSearch {} => Ok(Tool::new(
            make_web_search_tool_desc(),
            Arc::new(make_web_search_func()),
        )),
        BuiltinToolProvider::PythonRepl {} => {
            anyhow::bail!("sandbox feature required for PythonRepl")
        }
        BuiltinToolProvider::ConvertPdfToMd {} => build_convert_pdf_to_md_tool().await,
    }
}

#[cfg(feature = "sandbox")]
pub async fn make_builtin_tool(provider: &BuiltinToolProvider) -> anyhow::Result<Tool> {
    match provider {
        BuiltinToolProvider::WebSearch {} => Ok(Tool::new(
            make_web_search_tool_desc(),
            Arc::new(make_web_search_func()),
        )),
        BuiltinToolProvider::PythonRepl {} => build_python_repl_tool().await,
        BuiltinToolProvider::ConvertPdfToMd {} => build_convert_pdf_to_md_tool().await,
    }
}
