use crate::datatype::Value;
use indexmap::IndexMap;

/// Recursively add `"additionalProperties": false` to object sub-schemas that
/// omit it.  Required by Anthropic and OpenAI strict mode.  Returns a new owned
/// Value; the input is never mutated.
///
/// Traversal covers: `properties` values, `items`, `prefixItems` entries,
/// `anyOf`/`oneOf`/`allOf` entries, and `$defs`/`definitions` values.
pub fn for_strict_providers(schema: &Value) -> Value {
    let Value::Object(obj) = schema else {
        return schema.clone();
    };

    let mut out: IndexMap<String, Value> = obj
        .iter()
        .map(|(k, v)| {
            let transformed = match k.as_str() {
                "properties" | "$defs" | "definitions" => {
                    if let Value::Object(inner) = v {
                        Value::Object(
                            inner
                                .iter()
                                .map(|(ik, iv)| (ik.clone(), for_strict_providers(iv)))
                                .collect(),
                        )
                    } else {
                        v.clone()
                    }
                }
                "items" | "not" => for_strict_providers(v),
                "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                    if let Value::Array(arr) = v {
                        Value::Array(arr.iter().map(for_strict_providers).collect())
                    } else {
                        v.clone()
                    }
                }
                _ => v.clone(),
            };
            (k.clone(), transformed)
        })
        .collect();

    let is_object = out.get("type").and_then(|t| t.as_str()) == Some("object");
    let has_properties = out.contains_key("properties");
    if (is_object || has_properties) && !out.contains_key("additionalProperties") {
        out.insert("additionalProperties".into(), Value::Bool(false));
    }

    Value::Object(out)
}

/// Strip JSON Schema keywords that Gemini's `responseSchema` field does not
/// support.  Gemini uses a restricted OpenAPI 3.0 subset; `additionalProperties`,
/// `$schema`, `$defs`, and `definitions` cause a 400 INVALID_ARGUMENT.
/// Returns a new owned Value; the input is never mutated.
pub fn for_gemini(schema: &Value) -> Value {
    const STRIP: &[&str] = &["additionalProperties", "$schema", "$defs", "definitions"];
    match schema {
        Value::Object(obj) => Value::Object(
            obj.iter()
                .filter(|(k, _)| !STRIP.contains(&k.as_str()))
                .map(|(k, v)| (k.clone(), for_gemini(v)))
                .collect(),
        ),
        Value::Array(arr) => Value::Array(arr.iter().map(for_gemini).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_value;

    // ── for_strict_providers ─────────────────────────────────────────────

    #[test]
    fn strict_adds_additional_properties_to_flat_object() {
        let schema = to_value!({"type": "object", "properties": {"x": {"type": "string"}}});
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_adds_additional_properties_when_only_properties_present() {
        let schema = to_value!({"properties": {"x": {"type": "string"}}});
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/additionalProperties").and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_does_not_overwrite_existing_additional_properties_true() {
        let schema = to_value!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": true
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/additionalProperties").and_then(|v| v.as_bool()),
            Some(true),
            "must not overwrite caller's explicit value"
        );
    }

    #[test]
    fn strict_recurses_into_nested_object() {
        let schema = to_value!({
            "type": "object",
            "properties": {
                "addr": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/additionalProperties").and_then(|v| v.as_bool()),
            Some(false),
            "top level"
        );
        assert_eq!(
            out.pointer("/properties/addr/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false),
            "nested object"
        );
    }

    #[test]
    fn strict_recurses_into_array_items() {
        let schema = to_value!({
            "type": "array",
            "items": {"type": "object", "properties": {"name": {"type": "string"}}}
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/items/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_recurses_into_prefix_items() {
        let schema = to_value!({
            "type": "array",
            "prefixItems": [
                {"type": "object", "properties": {"id": {"type": "integer"}}},
                {"type": "string"}
            ]
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/prefixItems/0/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false),
            "object in prefixItems must be normalised"
        );
        assert!(
            out.pointer("/prefixItems/1/additionalProperties").is_none(),
            "non-object in prefixItems must not be touched"
        );
    }

    #[test]
    fn strict_recurses_into_anyof() {
        let schema = to_value!({
            "anyOf": [
                {"type": "object", "properties": {"a": {"type": "string"}}},
                {"type": "string"}
            ]
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/anyOf/0/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false),
            "object in anyOf must be normalised"
        );
        assert!(
            out.pointer("/anyOf/1/additionalProperties").is_none(),
            "non-object must not be touched"
        );
    }

    #[test]
    fn strict_recurses_into_defs() {
        let schema = to_value!({
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}},
            "$defs": {
                "Item": {"type": "object", "properties": {"id": {"type": "integer"}}}
            }
        });
        let out = for_strict_providers(&schema);
        assert_eq!(
            out.pointer("/$defs/Item/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_leaves_non_object_schema_unchanged() {
        let schema = to_value!({"type": "string"});
        let out = for_strict_providers(&schema);
        assert!(out.pointer("/additionalProperties").is_none());
    }

    #[test]
    fn strict_does_not_mutate_original() {
        let schema = to_value!({"type": "object", "properties": {"x": {"type": "string"}}});
        let _ = for_strict_providers(&schema);
        assert!(
            schema.pointer("/additionalProperties").is_none(),
            "original must not be mutated"
        );
    }

    // ── for_gemini ───────────────────────────────────────────────────────

    #[test]
    fn gemini_strips_additional_properties() {
        let schema = to_value!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": false
        });
        let out = for_gemini(&schema);
        assert!(out.pointer("/additionalProperties").is_none());
    }

    #[test]
    fn gemini_strips_schema_keyword() {
        let schema =
            to_value!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"});
        let out = for_gemini(&schema);
        assert!(out.pointer("/$schema").is_none());
    }

    #[test]
    fn gemini_strips_defs_and_definitions() {
        let schema = to_value!({
            "type": "object",
            "$defs": {"Foo": {"type": "string"}},
            "definitions": {"Bar": {"type": "integer"}}
        });
        let out = for_gemini(&schema);
        assert!(out.pointer("/$defs").is_none());
        assert!(out.pointer("/definitions").is_none());
    }

    #[test]
    fn gemini_preserves_other_keys() {
        let schema = to_value!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let out = for_gemini(&schema);
        assert!(out.pointer("/properties/name").is_some());
        assert!(out.pointer("/required").is_some());
    }

    #[test]
    fn gemini_strips_recursively_in_nested_object() {
        let schema = to_value!({
            "type": "object",
            "properties": {
                "addr": {
                    "type": "object",
                    "additionalProperties": false,
                    "properties": {"city": {"type": "string"}}
                }
            }
        });
        let out = for_gemini(&schema);
        assert!(
            out.pointer("/properties/addr/additionalProperties").is_none(),
            "must strip additionalProperties in nested object"
        );
    }
}
