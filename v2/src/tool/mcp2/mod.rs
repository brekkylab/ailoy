use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use rmcp::{
    RoleClient, ServiceExt as _,
    model::CallToolRequestParam,
    service::RunningService,
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};

use crate::{
    to_value,
    tool::Tool,
    value::{ToolDesc, Value},
};

#[derive(Debug)]
#[cfg_attr(feature = "python", pyo3_stub_gen_derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct MCPClient {
    pub service: Arc<RunningService<RoleClient, ()>>,
    pub tools: Vec<MCPTool>,
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

    pub async fn from_streamable_http(uri: impl Into<Arc<str>>) -> anyhow::Result<Self> {
        Self::new(
            ().serve(StreamableHttpClientTransport::from_uri(uri))
                .await?,
        )
        .await
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    service: Arc<RunningService<RoleClient, ()>>,
    inner: rmcp::model::Tool,
}

#[multi_platform_async_trait]
impl Tool for MCPTool {
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

    async fn run(&self, args: Value) -> Result<Value, String> {
        let tool_name = self.inner.name.clone();
        let peer = self.service.clone();

        // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
        let arguments: Option<serde_json::Map<String, serde_json::Value>> =
            serde_json::to_value(args)
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

        let parts =
            handle_result(result).map_err(|e| format!("call_tool_result_to_parts failed: {e}"))?;
        Ok(parts)
    }
}

fn handle_result(value: rmcp::model::CallToolResult) -> Result<Value, String> {
    if let Some(result) = value.structured_content {
        Ok(result.into())
    } else if let Some(content) = value.content {
        let mut rv = Vec::with_capacity(content.len());
        for raw_content in content {
            let v = match raw_content.raw {
                rmcp::model::RawContent::Text(raw_text_content) => {
                    Value::String(raw_text_content.text)
                }
                rmcp::model::RawContent::Image(_) => {
                    // @jhlee: Currently, it ignores
                    Value::Null
                }
                rmcp::model::RawContent::Resource(_) => Value::Null,
                rmcp::model::RawContent::Audio(_) => Value::Null,
            };
            rv.push(v);
        }
        if rv.len() == 0 {
            Ok(Value::Null)
        } else if rv.len() == 1 {
            Ok(rv.pop().unwrap())
        } else {
            Ok(to_value!(rv))
        }
    } else {
        Ok(Value::Null)
    }
}
