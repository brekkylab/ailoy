use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use rmcp::{
    model::CallToolRequestParam,
    service::{RoleClient, RunningService, ServiceExt},
    transport::{
        child_process::TokioChildProcess, streamable_http_client::StreamableHttpClientTransport,
    },
};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::{
    tool::{Tool, mcp::common::*},
    value::{Part, ToolDesc, Value},
};

#[derive(Clone, Debug)]
struct MCPClient {
    service: Arc<RunningService<RoleClient, ()>>,
}

impl MCPClient {
    pub async fn from_stdio(command: tokio::process::Command) -> anyhow::Result<Self> {
        let transport = TokioChildProcess::new(command)?;
        let service = ().serve(transport).await?;
        Ok(Self {
            service: Arc::new(service),
        })
    }

    pub async fn from_streamable_http(uri: impl Into<Arc<str>>) -> anyhow::Result<Self> {
        let transport = StreamableHttpClientTransport::from_uri(uri);
        let service = ().serve(transport).await?;
        Ok(Self {
            service: Arc::new(service),
        })
    }

    pub async fn list_tools(&self) -> anyhow::Result<Vec<MCPTool>> {
        let peer = self.service.peer();
        let tools = peer.list_all_tools().await?;
        Ok(tools
            .iter()
            .map(|t| MCPTool {
                client: Arc::new(self.clone()),
                name: t.name.to_string(),
                desc: map_mcp_tool_to_tool_description(t.clone()),
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    client: Arc<MCPClient>,
    name: String,
    desc: ToolDesc,
}

#[multi_platform_async_trait]
impl Tool for MCPTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    async fn run(&self, args: Value) -> Result<Vec<Part>, String> {
        let tool_name = self.name.clone();
        let peer = self.client.service.clone();

        // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
        let arguments: Option<JsonMap<String, JsonValue>> = serde_json::to_value(args)
            .map_err(|e| format!("serialize ToolCall arguments failed: {e}"))?
            .as_object()
            .cloned();

        // Invoke the MCP tool
        let result = peer
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments,
            })
            .await
            .map_err(|e| format!("mcp call_tool failed: {e}"))?;

        let parts = call_tool_result_to_parts(result)
            .map_err(|e| format!("call_tool_result_to_parts failed: {e}"))?;
        Ok(parts)
    }
}

pub async fn mcp_tools_from_stdio(
    // command: tokio::process::Command,
    command: String,
    args: Vec<String>,
    tool_name_prefix: &str,
) -> anyhow::Result<Vec<MCPTool>> {
    use rmcp::transport::child_process::ConfigureCommandExt;
    use tokio::process::Command;

    let command = Command::new(command).configure(|cmd| {
        for arg in args.iter() {
            cmd.arg(arg);
        }
    });
    let client = MCPClient::from_stdio(command).await?;
    Ok(client
        .list_tools()
        .await?
        .into_iter()
        .map(|mut t| {
            t.desc.name = format!("{}--{}", tool_name_prefix, t.desc.name);
            t
        })
        .collect())
}

pub async fn mcp_tools_from_streamable_http(
    url: &str,
    tool_name_prefix: &str,
) -> anyhow::Result<Vec<MCPTool>> {
    let client = MCPClient::from_streamable_http(url).await?;
    Ok(client
        .list_tools()
        .await?
        .into_iter()
        .map(|mut t| {
            t.desc.name = format!("{}--{}", tool_name_prefix, t.desc.name);
            t
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use crate::to_value;

    #[tokio::test]
    async fn run_stdio() -> anyhow::Result<()> {
        use onig::Regex;
        use rmcp::transport::ConfigureCommandExt;

        use super::*;

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let client = MCPClient::from_stdio(command).await?;

        let tools = client.list_tools().await?;
        assert_eq!(tools.len(), 2);

        let tool = tools[0].clone();
        let tool_name = tool.desc.name.clone();
        assert_eq!(tool_name, "get_current_time");

        let args = to_value!({
            "timezone": "Asia/Seoul"
        });
        let parts = tool.run(args).await.unwrap();
        assert_eq!(parts.len(), 1);

        let part = parts[0].clone();
        assert_eq!(part.is_text(), true);

        let parsed_part: serde_json::Value =
            serde_json::from_str(&part.as_text().unwrap()).unwrap();
        assert_eq!(parsed_part["timezone"].as_str(), Some("Asia/Seoul"));
        assert_eq!(parsed_part["is_dst"].as_bool(), Some(false));
        assert_eq!(
            Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$")?
                .is_match(parsed_part["datetime"].as_str().unwrap()),
            true
        );

        Ok(())
    }
}
