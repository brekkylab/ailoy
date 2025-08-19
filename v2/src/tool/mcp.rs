use std::sync::Arc;

use futures::future::BoxFuture;
use rmcp::{
    model::{CallToolRequestParam, RawContent, Tool as McpTool},
    service::{RoleClient, RunningService, ServiceExt},
    transport::{
        child_process::TokioChildProcess, sse_client::SseClientTransport,
        streamable_http_client::StreamableHttpClientTransport,
    },
};
use serde_json::{Map as JsonMap, Value as JsonValue};
use tokio::process::Command;

use crate::{
    tool::Tool,
    value::{Part, ToolCall, ToolDesc, ToolDescArg},
};

#[derive(Clone, Debug)]
pub struct MCPClient {
    service: Arc<RunningService<RoleClient, ()>>,
}

impl MCPClient {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    pub async fn from_stdio(command: Command) -> anyhow::Result<Self> {
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

    pub async fn from_sse(uri: impl Into<Arc<str>>) -> anyhow::Result<Self> {
        let transport = SseClientTransport::start(uri).await?;
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
                service: self.service.clone(),
                name: t.name.to_string(),
                desc: map_mcp_tool_to_tool_description(&t),
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    service: Arc<RunningService<RoleClient, ()>>,
    name: String,
    pub desc: ToolDesc,
}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    fn run(self: Arc<Self>, toll_call: ToolCall) -> BoxFuture<'static, Result<Vec<Part>, String>> {
        let peer = self.service.peer().clone();

        Box::pin(async move {
            // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
            let arguments: Option<JsonMap<String, JsonValue>> =
                serde_json::to_value(toll_call.arguments)
                    .map_err(|e| format!("serialize ToolCall arguments failed: {e}"))?
                    .as_object()
                    .cloned();

            // Invoke the MCP tool
            let result = peer
                .call_tool(CallToolRequestParam {
                    name: self.name.clone().into(),
                    arguments,
                })
                .await
                .map_err(|e| format!("mcp call_tool failed: {e}"))?; // rmcp Peer::call_tool

            // Prefer structured_content; else fall back to unstructured content.
            if let Some(v) = result.structured_content {
                return Ok(vec![Part::Text(v.to_string())]);
            }
            // Convert raw contents into corresponding Part types.
            // TODO: Handling resources
            if let Some(content) = result.content {
                let parts = content
                    .iter()
                    .map(|c| match c.raw.clone() {
                        RawContent::Text(text) => Part::Text(text.text),
                        RawContent::Image(image) => Part::ImageData(image.data),
                        RawContent::Audio(audio) => {
                            todo!();
                            // Part::Audio {
                            //     data: audio.data.clone(),
                            //     format: audio.mime_type.replace(r"^audio\/", ""),
                            // }
                        }
                        RawContent::Resource(_) => {
                            panic!("Not Implemented")
                        }
                    })
                    .collect();
                return Ok(parts);
            }

            // Nothing returned (valid per spec); return JSON null to keep contract stable.
            Ok(vec![Part::Text("null".to_string())])
        })
    }
}

/* ---------- helpers: map MCP Tool schema → [`ToolDesc`] ---------- */

// map McpTool → ToolDesc without moving out of Arc
fn map_mcp_tool_to_tool_description(tool: &McpTool) -> ToolDesc {
    let name = tool.name.clone();
    let desc = tool.description.clone().unwrap_or_default();

    let params_schema = json_obj_to_tool_desc_arg(tool.input_schema.as_ref());

    let ret_schema = tool
        .output_schema
        .as_ref()
        .map(|o| json_obj_to_tool_desc_arg(o.as_ref()));

    ToolDesc::new(name, desc, params_schema, ret_schema)
}

// Parse an object-shaped JSON schema by reference
fn json_obj_to_tool_desc_arg(obj: &JsonMap<String, JsonValue>) -> ToolDescArg {
    let ty = obj.get("type").and_then(|t| t.as_str()).unwrap_or("object");
    let desc = obj.get("description").and_then(|d| d.as_str());

    match ty {
        "string" => {
            let mut a = ToolDescArg::new_string();
            if let Some(e) = obj.get("enum").and_then(|e| e.as_array()) {
                let items = e
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect::<Vec<_>>();
                if !items.is_empty() {
                    a = a.with_enum(items);
                }
            }
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "number" | "integer" => {
            let mut a = ToolDescArg::new_number();
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "boolean" => {
            let mut a = ToolDescArg::new_boolean();
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "array" => {
            let mut a = ToolDescArg::new_array();
            if let Some(items) = obj.get("items") {
                a = a.with_items(json_to_tool_desc_arg(items));
            }
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "null" => ToolDescArg::new_null(),
        _ => {
            // object (or unknown) case
            let mut props = Vec::<(String, ToolDescArg)>::new();
            if let Some(p) = obj.get("properties").and_then(|p| p.as_object()) {
                for (k, v) in p.iter() {
                    props.push((k.clone(), json_to_tool_desc_arg(v)));
                }
            }
            let required = obj
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_owned))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();

            let mut o = ToolDescArg::new_object().with_properties(props.into_iter(), required);
            if let Some(d) = desc {
                o = o.with_desc(d);
            }
            o
        }
    }
}

// Fallback: handle either object or leaf JSON values by reference
fn json_to_tool_desc_arg(v: &JsonValue) -> ToolDescArg {
    if let Some(obj) = v.as_object() {
        return json_obj_to_tool_desc_arg(obj);
    }
    // If a server hands back a non-object where an object is expected, be defensive:
    ToolDescArg::new_object()
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_stdio() -> anyhow::Result<()> {
        use super::*;
        use crate::tool::Tool;
        use crate::value::{ToolCall, ToolCallArg};
        use indexmap::IndexMap;
        use onig::Regex;
        use rmcp::transport::ConfigureCommandExt;

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let client = MCPClient::from_stdio(command).await?;

        let tools = client.list_tools().await?;
        assert_eq!(tools.len(), 2);

        let tool = tools[0].clone();
        let tool_name = tool.desc.name.clone();
        assert_eq!(tool_name, "get_current_time");

        let mut tool_call_args: IndexMap<String, Box<ToolCallArg>> = IndexMap::new();
        tool_call_args.insert(
            "timezone".into(),
            Box::new(ToolCallArg::String("Asia/Seoul".into())),
        );

        let parts = Arc::new(tool)
            .run(ToolCall::new(
                tool_name,
                ToolCallArg::Object(tool_call_args),
            ))
            .await
            .unwrap();
        assert_eq!(parts.len(), 1);

        let part = parts[0].clone();

        let parsed_part: serde_json::Value = serde_json::from_str(part.as_str().unwrap()).unwrap();
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
