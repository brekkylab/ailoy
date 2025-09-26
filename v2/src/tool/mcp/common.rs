use crate::{
    to_value,
    value::{ToolDesc, Value},
};

/// Convert MCP tool description to ToolDesc
pub(super) fn map_mcp_tool_to_tool_description(value: rmcp::model::Tool) -> ToolDesc {
    ToolDesc {
        name: value.name.into(),
        description: value.description.map(|v| v.into()),
        parameters: value
            .input_schema
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    <serde_json::Value as Into<Value>>::into(v.clone()),
                )
            })
            .collect(),
        returns: value.output_schema.map(|map| {
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

/// Convert MCP result to parts
pub(super) fn call_tool_result_to_parts(
    value: rmcp::model::CallToolResult,
) -> Result<Value, String> {
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
