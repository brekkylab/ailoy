mod gemini_api;
mod gpt_api;
mod openrouter_api;

pub use gemini_api::*;
pub use gpt_api::*;
pub use openrouter_api::*;

/// Deep-merges a section's `extra` into the request `body`: objects merge key by key, any
/// other value replaces. `extra` must be an object and may not set a `reserved` path
/// (dotted, as in `generationConfig.responseModalities`).
fn merge_extra(
    body: &mut serde_json::Value,
    extra: Option<&crate::datatype::Value>,
    section: &str,
    reserved: &[&str],
) -> anyhow::Result<()> {
    let Some(extra) = extra else {
        return Ok(());
    };
    let extra = serde_json::Value::from(extra.clone());
    if !extra.is_object() {
        anyhow::bail!("`{section}.extra` must be a JSON object");
    }
    for path in reserved {
        let pointer = format!("/{}", path.replace('.', "/"));
        if extra.pointer(&pointer).is_some() {
            anyhow::bail!("`{section}.extra` may not set `{path}`");
        }
    }
    merge(body, extra);
    Ok(())
}

fn merge(target: &mut serde_json::Value, patch: serde_json::Value) {
    match (target, patch) {
        (serde_json::Value::Object(target), serde_json::Value::Object(patch)) => {
            for (key, value) in patch {
                match target.get_mut(&key) {
                    Some(slot) => merge(slot, value),
                    None => {
                        target.insert(key, value);
                    }
                }
            }
        }
        (target, patch) => *target = patch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_extra_merges_objects_and_replaces_the_rest() {
        let mut body = serde_json::json!({"a": {"b": 1, "c": 2}, "d": [1]});
        let extra = crate::to_value!({"a": {"c": 3, "e": 4}, "d": [2], "f": true});
        merge_extra(&mut body, Some(&extra), "test", &[]).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"a": {"b": 1, "c": 3, "e": 4}, "d": [2], "f": true})
        );
    }

    #[test]
    fn merge_extra_refuses_reserved_paths_and_non_objects() {
        let mut body = serde_json::json!({});
        let extra = crate::to_value!({"a": {"b": 1}});
        let err = merge_extra(&mut body, Some(&extra), "test", &["a.b"]).unwrap_err();
        assert_eq!(err.to_string(), "`test.extra` may not set `a.b`");
        // A sibling of a reserved path is fine.
        merge_extra(&mut body, Some(&extra), "test", &["a.c"]).unwrap();

        let err = merge_extra(&mut body, Some(&crate::to_value!([1])), "test", &[]).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"), "{err}");
    }
}
