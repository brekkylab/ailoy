use crate::{to_value, value::Value};

pub fn handle_result(value: rmcp::model::CallToolResult) -> Result<Value, String> {
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
