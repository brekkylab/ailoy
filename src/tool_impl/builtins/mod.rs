mod convert_pdf_to_md;
mod python_repl;
mod web_search;

use std::sync::Arc;

use convert_pdf_to_md::*;
use python_repl::*;
use web_search::*;

use crate::tool::{BuiltinToolProvider, Tool};

pub async fn make_builtin_tool(provider: &BuiltinToolProvider) -> anyhow::Result<Tool> {
    match provider {
        BuiltinToolProvider::WebSearch {} => Ok(Tool::new(
            make_web_search_tool_desc(),
            Arc::new(make_web_search_func()),
        )),
        BuiltinToolProvider::PythonRepl {
            python_version,
            venv_path,
            packages,
        } => {
            build_python_repl_tool(PythonReplConfig {
                python_version: python_version.clone(),
                venv_path: venv_path.clone(),
                packages: packages.clone(),
            })
            .await
        }
        BuiltinToolProvider::ConvertPdfToMd {} => build_convert_pdf_to_md_tool().await,
    }
}
