use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use anyhow::Context;
use rmcp::{
    RoleClient, ServiceExt as _,
    model::CallToolRequestParam,
    service::RunningService,
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};

use super::{super::ToolBehavior, common::handle_result};
use crate::value::{ToolDesc, Value};

#[derive(Debug)]
pub struct MCPClient {
    #[allow(unused)]
    service: Arc<RunningService<RoleClient, ()>>,
    tools: Vec<MCPTool>,
}

impl MCPClient {
    pub async fn new(service: RunningService<RoleClient, ()>) -> anyhow::Result<Self> {
        let service = Arc::new(service);
        let tools = service
            .peer()
            .list_all_tools()
            .await?
            .iter()
            .map(|t| MCPTool {
                service: service.clone(),
                inner: t.clone(),
            })
            .collect();
        Ok(Self { service, tools })
    }

    pub async fn from_stdio(command: tokio::process::Command) -> anyhow::Result<Self> {
        Self::new(().serve(TokioChildProcess::new(command)?).await?).await
    }

    pub async fn from_streamable_http(uri: impl Into<String>) -> anyhow::Result<Self> {
        Self::new(
            ().serve(StreamableHttpClientTransport::from_uri(uri.into()))
                .await?,
        )
        .await
    }

    pub fn get_tools(&self) -> &Vec<MCPTool> {
        &self.tools
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    service: Arc<RunningService<RoleClient, ()>>,
    inner: rmcp::model::Tool,
}

#[multi_platform_async_trait]
impl ToolBehavior for MCPTool {
    fn get_description(&self) -> ToolDesc {
        ToolDesc {
            name: self.inner.name.to_string(),
            description: self.inner.description.clone().map(|v| v.into()),
            parameters: self
                .inner
                .input_schema
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        <serde_json::Value as Into<Value>>::into(v.clone()),
                    )
                })
                .collect(),
            returns: self.inner.output_schema.clone().map(|map| {
                map.iter()
                    .map(|(k, v)| {
                        (
                            k.clone(),
                            <serde_json::Value as Into<Value>>::into(v.clone()),
                        )
                    })
                    .collect()
            }),
        }
    }

    async fn run(&self, args: Value) -> anyhow::Result<Value> {
        let tool_name = self.inner.name.clone();
        let peer = self.service.clone();

        let arguments: Option<serde_json::Map<String, serde_json::Value>> =
            serde_json::to_value(args)
                .context("serialize ToolCall arguments failed")?
                .as_object()
                .cloned();

        let result = peer
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments,
            })
            .await
            .context("mcp call_tool failed")?;

        let parts = handle_result(result).context("call_tool_result_to_parts failed")?;
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    #[tokio::test]
    async fn run_stdio() -> anyhow::Result<()> {
        use fancy_regex::Regex;
        use rmcp::transport::ConfigureCommandExt;

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let client = MCPClient::from_stdio(command).await?;

        let tools = client.tools;
        assert_eq!(tools.len(), 2);

        let tool = tools[0].clone();
        let tool_name = tool.get_description().name.clone();
        assert_eq!(tool_name, "get_current_time");

        let tool_call_args: Value = serde_json::json!({
            "timezone": "Asia/Seoul"
        })
        .into();

        let part = tool.run(tool_call_args).await.unwrap();
        assert_eq!(part.is_string(), true);

        let parsed_part: serde_json::Value =
            serde_json::from_str(&part.as_str().unwrap()).unwrap();
        assert_eq!(parsed_part["timezone"].as_str(), Some("Asia/Seoul"));
        assert_eq!(parsed_part["is_dst"].as_bool(), Some(false));
        assert_eq!(
            Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$")?
                .is_match(parsed_part["datetime"].as_str().unwrap())
                .unwrap(),
            true
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_streamable_http_client() -> anyhow::Result<()> {
        let client = MCPClient::from_streamable_http("http://localhost:8123/mcp").await?;
        let tools = client.tools;
        crate::debug!("list of tools: {:?}", tools);

        let tool = tools[1].clone();
        let tool_call_args: Value =
            serde_json::json!({"latitude": 32.7767, "longitude": -96.797}).into();
        let call_tool = tool.run(tool_call_args).await.unwrap();
        crate::debug!("call tool result: {:?}", call_tool);

        Ok(())
    }
}
