use rmcp::model::{CallToolResult, RawContent, Tool as McpTool};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::value::{Part, ToolDesc, ToolDescArg};

/* ---------- helpers: map MCP Tool schema → [`ToolDescription`] ---------- */
// map McpTool → ToolDescription without moving out of Arc
pub fn map_mcp_tool_to_tool_description(tool: &McpTool) -> ToolDesc {
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
pub fn json_obj_to_tool_desc_arg(obj: &JsonMap<String, JsonValue>) -> ToolDescArg {
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
pub fn json_to_tool_desc_arg(v: &JsonValue) -> ToolDescArg {
    if let Some(obj) = v.as_object() {
        return json_obj_to_tool_desc_arg(obj);
    }
    // If a server hands back a non-object where an object is expected, be defensive:
    ToolDescArg::new_object()
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
                RawContent::Image(image) => Part::ImageData(image.data, image.mime_type),
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
