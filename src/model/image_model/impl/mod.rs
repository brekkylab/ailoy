mod gemini_api;
mod gpt_api;
mod openrouter_api;

pub use gemini_api::*;
pub use gpt_api::*;
pub use openrouter_api::*;

use crate::model::ImageOptions;

/// `options` as the parameters `T` of `model`'s API, whichever API they were built for; no
/// options read as `T`'s defaults. A field `T` does not have is an error naming it.
fn read_options<T: serde::de::DeserializeOwned + Default>(
    options: Option<&dyn ImageOptions>,
    model: &str,
) -> anyhow::Result<T> {
    let Some(options) = options else {
        return Ok(T::default());
    };
    options
        .to_json()
        .and_then(serde_json::from_value)
        .map_err(|e| anyhow::anyhow!("invalid image options for '{model}': {e}"))
}

/// Deep-merges `extra` into the request `body`: objects merge key by key, any other value
/// replaces. `extra` must be an object and may not set a `reserved` path (dotted, as in
/// `generationConfig.responseModalities`).
fn merge_extra(
    body: &mut serde_json::Value,
    extra: Option<&crate::datatype::Value>,
    reserved: &[&str],
) -> anyhow::Result<()> {
    let Some(extra) = extra else {
        return Ok(());
    };
    let extra = serde_json::Value::from(extra.clone());
    if !extra.is_object() {
        anyhow::bail!("`extra` must be a JSON object");
    }
    for path in reserved {
        let pointer = format!("/{}", path.replace('.', "/"));
        if extra.pointer(&pointer).is_some() {
            anyhow::bail!("`extra` may not set `{path}`");
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
    use crate::model::{
        GeminiImageOptions, ImageModelOptions, OpenAIImageOptions, OpenRouterImageOptions,
    };

    #[test]
    fn read_options_takes_options_built_for_another_api_when_the_fields_fit() {
        let options = ImageModelOptions::OpenAI(OpenAIImageOptions {
            n: Some(2),
            size: Some("1536x1024".to_owned()),
            ..Default::default()
        });
        let read: OpenRouterImageOptions = read_options(Some(&options), "m").unwrap();
        assert_eq!(read.n, Some(2));
        assert_eq!(read.size.as_deref(), Some("1536x1024"));

        let none: GeminiImageOptions = read_options(None, "m").unwrap();
        assert!(none.aspect_ratio.is_none() && none.extra.is_none());
    }

    #[test]
    fn read_options_names_a_field_the_api_does_not_have() {
        let options = ImageModelOptions::OpenRouter(OpenRouterImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        });
        let err = read_options::<OpenAIImageOptions>(Some(&options), "gpt-image-1")
            .unwrap_err()
            .to_string();
        assert!(err.contains("'gpt-image-1'"), "{err}");
        assert!(err.contains("unknown field `aspect_ratio`"), "{err}");
        assert!(
            err.contains("`size`"),
            "the fields it has are listed: {err}"
        );
    }

    #[test]
    fn merge_extra_merges_objects_and_replaces_the_rest() {
        let mut body = serde_json::json!({"a": {"b": 1, "c": 2}, "d": [1]});
        let extra = crate::to_value!({"a": {"c": 3, "e": 4}, "d": [2], "f": true});
        merge_extra(&mut body, Some(&extra), &[]).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"a": {"b": 1, "c": 3, "e": 4}, "d": [2], "f": true})
        );
    }

    #[test]
    fn merge_extra_refuses_reserved_paths_and_non_objects() {
        let mut body = serde_json::json!({});
        let extra = crate::to_value!({"a": {"b": 1}});
        let err = merge_extra(&mut body, Some(&extra), &["a.b"]).unwrap_err();
        assert_eq!(err.to_string(), "`extra` may not set `a.b`");
        // A sibling of a reserved path is fine.
        merge_extra(&mut body, Some(&extra), &["a.c"]).unwrap();

        let err = merge_extra(&mut body, Some(&crate::to_value!([1])), &[]).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"), "{err}");
    }
}
