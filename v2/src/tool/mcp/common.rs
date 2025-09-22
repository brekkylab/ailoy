use rmcp::model::{CallToolResult, RawContent, Tool as McpTool};
use serde_json::Value as JsonValue;

use crate::value::{Part, ToolDesc};

/* ---------- helpers: map MCP Tool schema → [`ToolDescription`] ---------- */
// map McpTool → ToolDescription without moving out of Arc
pub fn map_mcp_tool_to_tool_description(tool: &McpTool) -> ToolDesc {
    let name = tool.name.clone();
    let desc = tool.description.clone().unwrap_or_default();
    let parameters = JsonValue::Object(tool.input_schema.as_ref().clone());
    let ret_schema = tool
        .output_schema
        .as_ref()
        .map(|o| JsonValue::Object(o.as_ref().clone()));

    ToolDesc::new(name.to_string(), desc.to_string(), parameters, ret_schema).unwrap()
}

pub fn call_tool_result_to_parts(result: &CallToolResult) -> anyhow::Result<Vec<Part>> {
    // Prefer structured_content; else fall back to unstructured content.
    if let Some(v) = &result.structured_content {
        return Ok(vec![Part::Text(v.to_string())]);
    }
    // Convert raw contents into corresponding Part types.
    // TODO: Handling resources
    if let Some(content) = &result.content {
        let parts = content
            .iter()
            .map(|c| match c.raw.clone() {
                RawContent::Text(text) => Part::Text(text.text),
                RawContent::Image(image) => Part::ImageData {
                    data: image.data,
                    mime_type: image.mime_type,
                },
                // RawContent::Audio(audio) => Part::Audio {
                //     data: audio.data.clone(),
                //     format: audio.mime_type.replace(r"^audio\/", ""),
                // },
                RawContent::Resource(_) => {
                    panic!("Not Implemented")
                }
                _ => todo!(),
            })
            .collect();
        return Ok(parts);
    }

    // Nothing returned (valid per spec); return JSON null to keep contract stable.
    Ok(vec![Part::Text("null".to_string())])
}
