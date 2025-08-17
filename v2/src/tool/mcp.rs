use std::sync::Arc;

use futures::future::BoxFuture;
use rmcp::{
    model::{CallToolRequestParam, Tool as McpTool},
    service::{Peer, RoleClient},
};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::{
    tool::Tool,
    value::{Part, ToolCall, ToolDesc, ToolDescArg},
};

/// Bridge a single MCP tool (on a connected MCP server) into your `Tool` trait.
#[derive(Clone, Debug)]
pub struct MCPTool {
    peer: Arc<Peer<RoleClient>>,
    name: String,
    desc: ToolDesc,
}

impl MCPTool {
    /// Build from a discovered MCP `Tool` object.
    pub fn from_known(peer: Arc<Peer<RoleClient>>, tool: McpTool) -> Self {
        let name = tool.name.to_string();
        let desc = map_mcp_tool_to_tool_description(&tool);
        Self { peer, name, desc }
    }

    /// Look up a tool by name from the server and build an `MCPTool`.
    pub async fn discover(peer: Arc<Peer<RoleClient>>, tool_name: &str) -> Result<Self, String> {
        let tools = peer
            .list_all_tools()
            .await
            .map_err(|e| format!("list_all_tools failed: {e}"))?; // rmcp Peer API
        let tool = tools
            .into_iter()
            .find(|t| t.name == tool_name)
            .ok_or_else(|| format!("MCP tool '{tool_name}' not found"))?;
        Ok(Self::from_known(peer, tool))
    }
}

impl Tool for MCPTool {
    fn get_description(&self) -> ToolDesc {
        self.desc.clone()
    }

    fn run(self: Arc<Self>, toll_call: ToolCall) -> BoxFuture<'static, Result<Part, String>> {
        let peer = self.peer.clone();
        let name = self.name.clone();

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
                    name: name.into(),
                    arguments,
                })
                .await
                .map_err(|e| format!("mcp call_tool failed: {e}"))?; // rmcp Peer::call_tool

            // Prefer structured_content; else fall back to unstructured content.
            if let Some(v) = result.structured_content {
                return Ok(Part::Text(v.to_string()));
            }
            if let Some(content) = result.content {
                // Content is an MCP vector of parts; serialize to JSON for consistency
                let s = serde_json::to_string(&content).unwrap_or_else(|_| "[]".to_string());
                return Ok(Part::Text(s));
            }

            // Nothing returned (valid per spec); return JSON null to keep contract stable.
            Ok(Part::Text("null".to_string()))
        })
    }
}

/* ---------- helpers: map MCP Tool schema → [`ToolDesc`] ---------- */

// map MCP tool → your ToolDesc without moving out of Arc
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
