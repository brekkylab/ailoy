use indexmap::IndexMap;

use crate::datatype::Value;

/// Adapts a JSON schema for a specific provider's wire format requirements.
///
/// The default implementation adds `"additionalProperties": false` to every
/// object sub-schema, satisfying Anthropic and OpenAI strict mode.  Marshal
/// types that require different transformations (e.g. stripping unsupported
/// keywords) override [`marshal_response_schema`].
pub trait ResponseSchemaMarshal {
    /// Recursively add `"additionalProperties": false` to object sub-schemas that
    /// omit it.  Returns a new owned Value; the input is never mutated.
    ///
    /// Traversal covers: `properties` values, `items`, `prefixItems` entries,
    /// `anyOf`/`oneOf`/`allOf` entries, and `$defs`/`definitions` values.
    fn marshal_response_schema(&self, schema: &Value) -> Value {
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
                                    .map(|(ik, iv)| (ik.clone(), self.marshal_response_schema(iv)))
                                    .collect(),
                            )
                        } else {
                            v.clone()
                        }
                    }
                    "items" | "not" => self.marshal_response_schema(v),
                    "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                        if let Value::Array(arr) = v {
                            Value::Array(
                                arr.iter()
                                    .map(|iv| self.marshal_response_schema(iv))
                                    .collect(),
                            )
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_value;

    struct TestMarshal {}
    impl ResponseSchemaMarshal for TestMarshal {}

    static MARSHAL: TestMarshal = TestMarshal {};

    #[test]
    fn strict_adds_additional_properties_to_flat_object() {
        let schema = to_value!({"type": "object", "properties": {"x": {"type": "string"}}});
        let out = MARSHAL.marshal_response_schema(&schema);
        assert_eq!(
            out.pointer("/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_adds_additional_properties_when_only_properties_present() {
        let schema = to_value!({"properties": {"x": {"type": "string"}}});
        let out = MARSHAL.marshal_response_schema(&schema);
        assert_eq!(
            out.pointer("/additionalProperties")
                .and_then(|v| v.as_bool()),
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
        let out = MARSHAL.marshal_response_schema(&schema);
        assert_eq!(
            out.pointer("/additionalProperties")
                .and_then(|v| v.as_bool()),
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
        let out = MARSHAL.marshal_response_schema(&schema);
        assert_eq!(
            out.pointer("/additionalProperties")
                .and_then(|v| v.as_bool()),
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
        let out = MARSHAL.marshal_response_schema(&schema);
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
        let out = MARSHAL.marshal_response_schema(&schema);
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
        let out = MARSHAL.marshal_response_schema(&schema);
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
        let out = MARSHAL.marshal_response_schema(&schema);
        assert_eq!(
            out.pointer("/$defs/Item/additionalProperties")
                .and_then(|v| v.as_bool()),
            Some(false)
        );
    }

    #[test]
    fn strict_leaves_non_object_schema_unchanged() {
        let schema = to_value!({"type": "string"});
        let out = MARSHAL.marshal_response_schema(&schema);
        assert!(out.pointer("/additionalProperties").is_none());
    }

    #[test]
    fn strict_does_not_mutate_original() {
        let schema = to_value!({"type": "object", "properties": {"x": {"type": "string"}}});
        let _ = MARSHAL.marshal_response_schema(&schema);
        assert!(
            schema.pointer("/additionalProperties").is_none(),
            "original must not be mutated"
        );
    }
}
