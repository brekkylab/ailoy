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
    value::{Part, ToolCall, ToolDescription, ToolDescriptionArgument},
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

    pub async fn list_tools(&self) -> anyhow::Result<Vec<Arc<MCPTool>>> {
        let peer = self.service.peer();
        let tools = peer.list_all_tools().await?;
        Ok(tools
            .iter()
            .map(|t| {
                Arc::new(MCPTool {
                    service: self.service.clone(),
                    name: t.name.to_string(),
                    desc: map_mcp_tool_to_tool_description(&t),
                })
            })
            .collect())
    }
}

#[derive(Clone, Debug)]
pub struct MCPTool {
    service: Arc<RunningService<RoleClient, ()>>,
    name: String,
    desc: ToolDescription,
}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDescription {
        self.desc.clone()
    }

    fn run(self: Arc<Self>, toll_call: ToolCall) -> BoxFuture<'static, Result<Vec<Part>, String>> {
        let peer = self.service.peer().clone();
        let name = self.name.clone();

        Box::pin(async move {
            // Convert your ToolCall arguments → serde_json::Map (MCP expects JSON object)
            let arguments: Option<JsonMap<String, JsonValue>> =
                serde_json::to_value(toll_call.get_argument())
                    .map_err(|e| format!("serialize ToolCall arguments failed: {e}"))?
                    .as_object()
                    .cloned();

            // Invoke the MCP tool
            let result = peer
                .call_tool(CallToolRequestParam {
                    name: name.into(),
                    arguments,
                })
                .await
                .map_err(|e| format!("mcp call_tool failed: {e}"))?; // rmcp Peer::call_tool

            // Prefer structured_content; else fall back to unstructured content.
            if let Some(v) = result.structured_content {
                println!("structured_content: {:?}", v);
                return Ok(vec![Part::Text(v.to_string())]);
            }
            if let Some(content) = result.content {
                let parts = content
                    .iter()
                    .map(|c| match c.raw.clone() {
                        RawContent::Text(text) => Part::Text(text.text),
                        RawContent::Image(image) => Part::ImageData(image.data),
                        RawContent::Audio(audio) => Part::Audio {
                            data: audio.data.clone(),
                            format: audio.mime_type.replace(r"^audio\/", ""),
                        },
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

/* ---------- helpers: map MCP Tool schema → [`ToolDescription`] ---------- */

// map McpTool → ToolDescription without moving out of Arc
fn map_mcp_tool_to_tool_description(tool: &McpTool) -> ToolDescription {
    let name = tool.name.clone();
    let desc = tool.description.clone().unwrap_or_default();

    let params_schema = json_obj_to_tool_desc_arg(tool.input_schema.as_ref());

    let ret_schema = tool
        .output_schema
        .as_ref()
        .map(|o| json_obj_to_tool_desc_arg(o.as_ref()));

    ToolDescription::new(name, desc, params_schema, ret_schema)
}

// Parse an object-shaped JSON schema by reference
fn json_obj_to_tool_desc_arg(obj: &JsonMap<String, JsonValue>) -> ToolDescriptionArgument {
    let ty = obj.get("type").and_then(|t| t.as_str()).unwrap_or("object");
    let desc = obj.get("description").and_then(|d| d.as_str());

    match ty {
        "string" => {
            let mut a = ToolDescriptionArgument::new_string();
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
            let mut a = ToolDescriptionArgument::new_number();
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "boolean" => {
            let mut a = ToolDescriptionArgument::new_boolean();
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "array" => {
            let mut a = ToolDescriptionArgument::new_array();
            if let Some(items) = obj.get("items") {
                a = a.with_items(json_to_tool_desc_arg(items));
            }
            if let Some(d) = desc {
                a = a.with_desc(d);
            }
            a
        }
        "null" => ToolDescriptionArgument::new_null(),
        _ => {
            // object (or unknown) case
            let mut props = Vec::<(String, ToolDescriptionArgument)>::new();
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

            let mut o =
                ToolDescriptionArgument::new_object().with_properties(props.into_iter(), required);
            if let Some(d) = desc {
                o = o.with_desc(d);
            }
            o
        }
    }
}

// Fallback: handle either object or leaf JSON values by reference
fn json_to_tool_desc_arg(v: &JsonValue) -> ToolDescriptionArgument {
    if let Some(obj) = v.as_object() {
        return json_obj_to_tool_desc_arg(obj);
    }
    // If a server hands back a non-object where an object is expected, be defensive:
    ToolDescriptionArgument::new_object()
}

#[cfg(test)]
mod tests {
    #[cfg(any(target_family = "unix", target_family = "windows"))]
    #[tokio::test]
    async fn run_stdio() -> anyhow::Result<()> {
        use std::collections::HashMap;

        use super::*;
        use crate::tool::Tool;
        use crate::value::{ToolCall, ToolCallArgument};
        use regex::Regex;
        use rmcp::transport::ConfigureCommandExt;

        let command = tokio::process::Command::new("uvx").configure(|cmd| {
            cmd.arg("mcp-server-time");
        });
        let client = MCPClient::from_stdio(command).await?;

        let tools = client.list_tools().await?;
        assert_eq!(tools.len(), 2);

        let tool = tools[0].clone();
        let tool_name = tool.name.clone();
        assert_eq!(tool_name, "get_current_time");

        let mut tool_call_args: HashMap<String, Box<ToolCallArgument>> = HashMap::new();
        tool_call_args.insert(
            "timezone".into(),
            Box::new(ToolCallArgument::String("Asia/Seoul".into())),
        );

        let parts = tool
            .run(ToolCall::new(
                tool_name,
                ToolCallArgument::Object(tool_call_args),
            ))
            .await
            .unwrap();
        assert_eq!(parts.len(), 1);

        let part = parts[0].clone();
        assert_eq!(part.is_text(), true);

        let parsed_part: serde_json::Value =
            serde_json::from_str(part.get_text().unwrap()).unwrap();
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
