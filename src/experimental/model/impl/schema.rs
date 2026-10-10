//! Helpers for the JSON schemas handed to model APIs.

use crate::datatype::Value;

/// Adds `"additionalProperties": false` to every object sub-schema that omits it, as the
/// APIs require of a response schema in strict mode.
pub(super) fn close_objects(schema: &Value) -> Value {
    let Value::Object(obj) = schema else {
        return schema.clone();
    };
    let map_values = |v: &Value| match v {
        Value::Object(inner) => Value::Object(
            inner
                .iter()
                .map(|(k, v)| (k.clone(), close_objects(v)))
                .collect(),
        ),
        other => other.clone(),
    };
    let map_items = |v: &Value| match v {
        Value::Array(items) => Value::Array(items.iter().map(close_objects).collect()),
        other => other.clone(),
    };
    let mut out: indexmap::IndexMap<String, Value> = obj
        .iter()
        .map(|(k, v)| {
            let v = match k.as_str() {
                "properties" | "$defs" | "definitions" => map_values(v),
                "items" | "not" => close_objects(v),
                "prefixItems" | "anyOf" | "oneOf" | "allOf" => map_items(v),
                _ => v.clone(),
            };
            (k.clone(), v)
        })
        .collect();
    let is_object = out.get("type").and_then(|t| t.as_str()) == Some("object");
    if (is_object || out.contains_key("properties")) && !out.contains_key("additionalProperties") {
        out.insert("additionalProperties".into(), Value::Bool(false));
    }
    Value::Object(out)
}
