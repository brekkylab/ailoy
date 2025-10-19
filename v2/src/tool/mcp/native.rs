use std::sync::Arc;

use ailoy_macros::multi_platform_async_trait;
use anyhow::Context;
use rmcp::{
    RoleClient, ServiceExt as _,
    model::CallToolRequestParam,
    service::RunningService,
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};

use crate::{
    tool::{ToolBehavior, mcp::common::handle_result},
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

        // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
        let arguments: Option<serde_json::Map<String, serde_json::Value>> =
            serde_json::to_value(args)
                .context("serialize ToolCall arguments failed: {e}")?
                .as_object()
                .cloned();

        // Invoke the MCP tool
        let result = peer
            .call_tool(CallToolRequestParam {
                name: tool_name.into(),
                arguments,
            })
            .await
            .context("mcp call_tool failed: {e}")?;

        let parts = handle_result(result).context("call_tool_result_to_parts failed: {e}")?;
        Ok(parts)
    }
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn run_stdio() -> anyhow::Result<()> {
        use onig::Regex;
        use rmcp::transport::ConfigureCommandExt;

        use super::*;
        use crate::value::Value;

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

        let parsed_part: serde_json::Value = serde_json::from_str(&part.as_str().unwrap()).unwrap();
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

#[cfg(feature = "python")]
mod py {
    use pyo3::{prelude::*, pymethods, types::PyType};
    use pyo3_stub_gen_derive::gen_stub_pymethods;
    use rmcp::transport::ConfigureCommandExt;

    use super::*;
    use crate::tool::{MCPClient, Tool};

    #[gen_stub_pymethods]
    #[pymethods]
    impl MCPClient {
        fn __repr__(&self) -> String {
            format!(
                "MCPClient(tools=[{}])",
                self.tools
                    .iter()
                    .map(|tool| format!("Tool(MCPTool(name={}))", tool.inner.name.as_ref()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }

        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MCPClient]"))]
        #[pyo3(name="from_stdio", signature = (command, args))]
        fn from_stdio_py<'py>(
            _cls: Bound<'py, PyType>,
            py: Python<'py>,
            command: String,
            args: Vec<String>,
        ) -> PyResult<Py<PyAny>> {
            let fut = async move {
                let command = tokio::process::Command::new(command).configure(|cmd| {
                    cmd.args(args);
                });
                MCPClient::from_stdio(command).await.map_err(Into::into)
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut.into())
        }

        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MCPClient]"))]
        #[pyo3(name="from_streamable_http", signature = (url))]
        fn from_streamable_http_py<'py>(
            _cls: Bound<'py, PyType>,
            py: Python<'py>,
            url: String,
        ) -> PyResult<Py<PyAny>> {
            let fut = async move {
                MCPClient::from_streamable_http(url)
                    .await
                    .map_err(Into::into)
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut.into())
        }

        #[pyo3(signature = ())]
        fn tools(&self) -> Vec<Tool> {
            self.tools
                .iter()
                .map(|t| Tool::new_mcp(t.clone()))
                .collect()
        }

        #[pyo3(signature = (name))]
        fn get_tool(&self, name: String) -> Option<Tool> {
            self.tools()
                .iter()
                .find(|tool| tool.get_description().name == name)
                .cloned()
        }
    }
}
