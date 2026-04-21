mod convert_pdf_to_md;
mod python_repl;
mod web_search;

use std::sync::Arc;

use convert_pdf_to_md::*;
#[cfg(feature = "sandbox-microvm")]
use python_repl::*;
use web_search::*;

use crate::tool::{BuiltinToolProvider, Tool};

#[cfg(not(feature = "sandbox-microvm"))]
pub async fn make_builtin_tool(provider: &BuiltinToolProvider) -> anyhow::Result<Tool> {
    match provider {
        BuiltinToolProvider::WebSearch {} => Ok(Tool::new(
            make_web_search_tool_desc(),
            Arc::new(make_web_search_func()),
        )),
        BuiltinToolProvider::PythonRepl {} => {
            anyhow::bail!("sandbox-microvm feature required for PythonRepl")
        }
        BuiltinToolProvider::ConvertPdfToMd {} => build_convert_pdf_to_md_tool().await,
    }
}

#[cfg(feature = "sandbox-microvm")]
pub async fn make_builtin_tool(
    provider: &BuiltinToolProvider,
    sandbox: Option<std::sync::Arc<crate::sandbox::Sandbox>>,
) -> anyhow::Result<Tool> {
    match provider {
        BuiltinToolProvider::WebSearch {} => Ok(Tool::new(
            make_web_search_tool_desc(),
            Arc::new(make_web_search_func()),
        )),
        BuiltinToolProvider::PythonRepl {} => {
            let sb = sandbox.ok_or_else(|| anyhow::anyhow!("sandbox required for python_repl"))?;
            build_python_repl_tool(sb).await
        }
        BuiltinToolProvider::ConvertPdfToMd {} => {
            let sb = sandbox.ok_or_else(|| anyhow::anyhow!("sandbox required for convert_pdf_to_md"))?;
            build_convert_pdf_to_md_tool(sb).await
        }
    }
}
