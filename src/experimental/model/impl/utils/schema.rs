//! Adapting a JSON schema to a vendor's structured outputs.

/// Adds `"additionalProperties": false` to every object sub-schema that leaves it out, as
/// strict structured outputs require; an explicit value is kept.
pub(crate) fn close_objects(schema: &serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(obj) = schema else {
        return schema.clone();
    };
    let mut out: serde_json::Map<_, _> = obj
        .iter()
        .map(|(k, v)| {
            let v = match (k.as_str(), v) {
                ("properties" | "$defs" | "definitions", serde_json::Value::Object(inner)) => {
                    serde_json::Value::Object(
                        inner
                            .iter()
                            .map(|(ik, iv)| (ik.clone(), close_objects(iv)))
                            .collect(),
                    )
                }
                ("items" | "not", _) => close_objects(v),
                ("prefixItems" | "anyOf" | "oneOf" | "allOf", serde_json::Value::Array(arr)) => {
                    serde_json::Value::Array(arr.iter().map(close_objects).collect())
                }
                _ => v.clone(),
            };
            (k.clone(), v)
        })
        .collect();
    let is_object = out.get("type").and_then(|t| t.as_str()) == Some("object");
    if (is_object || out.contains_key("properties")) && !out.contains_key("additionalProperties") {
        out.insert("additionalProperties".into(), false.into());
    }
    serde_json::Value::Object(out)
}
