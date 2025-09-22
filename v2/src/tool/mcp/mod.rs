#[cfg(not(target_arch = "wasm32"))]
pub mod native;
#[cfg(target_arch = "wasm32")]
pub mod wasm32;

mod common;

use std::sync::Arc;

use crate::tool::Tool;

#[derive(PartialEq, Clone)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen_derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass(eq))]
pub enum MCPTransport {
    Stdio { command: String, args: Vec<String> },
    StreamableHttp { url: String },
}

impl MCPTransport {
    pub async fn get_tools(self, tool_name_prefix: &str) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
        match self {
            MCPTransport::Stdio {
                command: _command,
                args: _args,
            } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    return native::mcp_tools_from_stdio(_command, _args, tool_name_prefix)
                        .await?
                        .into_iter()
                        .map(|tool| Ok(Arc::new(tool) as Arc<dyn Tool>))
                        .collect::<anyhow::Result<Vec<Arc<dyn Tool>>>>();
                }

                #[cfg(target_arch = "wasm32")]
                {
                    Err(anyhow::anyhow!(
                        "stdio is not available on wasm32-unknown-unknown!"
                    ))
                }
            }
            MCPTransport::StreamableHttp { url } => {
                #[cfg(not(target_arch = "wasm32"))]
                return native::mcp_tools_from_streamable_http(url.as_str(), tool_name_prefix)
                    .await?
                    .into_iter()
                    .map(|tool| Ok(Arc::new(tool) as Arc<dyn Tool>))
                    .collect::<anyhow::Result<Vec<Arc<dyn Tool>>>>();

                #[cfg(target_arch = "wasm32")]
                return wasm32::mcp_tools_from_streamable_http(url.as_str(), tool_name_prefix)
                    .await?
                    .into_iter()
                    .map(|tool| Ok(Arc::new(tool) as Arc<dyn Tool>))
                    .collect::<anyhow::Result<Vec<Arc<dyn Tool>>>>();
            }
        }
    }
}
