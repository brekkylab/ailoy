#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm32;

mod common;

use std::sync::Arc;

use crate::tool::Tool;

pub enum MCPTransport {
    Stdio(&'static str, Vec<&'static str>),
    StreamableHttp(&'static str),
}

impl MCPTransport {
    pub async fn get_tools(self, tool_name_prefix: &str) -> anyhow::Result<Vec<Arc<dyn Tool>>> {
        match self {
            MCPTransport::Stdio(_command, _args) => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use rmcp::transport::child_process::ConfigureCommandExt;
                    use tokio::process::Command;

                    let command = Command::new(_command).configure(|cmd| {
                        for arg in _args.iter() {
                            cmd.arg(arg);
                        }
                    });
                    return native::mcp_tools_from_stdio(command, tool_name_prefix).await;
                }

                #[cfg(target_arch = "wasm32")]
                {
                    Err(anyhow::anyhow!(
                        "stdio is not available on wasm32-unknown-unknown!"
                    ))
                }
            }
            MCPTransport::StreamableHttp(url) => {
                #[cfg(not(target_arch = "wasm32"))]
                return native::mcp_tools_from_streamable_http(url, tool_name_prefix).await;

                #[cfg(target_arch = "wasm32")]
                return wasm32::mcp_tools_from_streamable_http(url, tool_name_prefix).await;
            }
        }
    }
}
