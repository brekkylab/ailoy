mod gemini_api;
mod gpt_api;
mod openrouter_api;

pub use gemini_api::*;
pub use gpt_api::*;
pub use openrouter_api::*;

use crate::model::ImageOptions;

/// `options` as the parameters `T` of `model`'s API, or `T`'s defaults without any; an error
/// names the model.
fn read_options<T: serde::de::DeserializeOwned + Default>(
    options: Option<&dyn ImageOptions>,
    model: &str,
) -> anyhow::Result<T> {
    let Some(options) = options else {
        return Ok(T::default());
    };
    options
        .read()
        .map_err(|e| anyhow::anyhow!("invalid image options for '{model}': {e}"))
}

/// Deep-merges `extra` into the request `body`: objects merge key by key, any other value
/// replaces. `extra` must be an object and may not set a `reserved` path (dotted, as in
/// `generationConfig.responseModalities`), nor replace an object on the way to one.
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
        let keys: Vec<&str> = path.split('.').collect();
        let mut value = &extra;
        for (depth, key) in keys.iter().enumerate() {
            let Some(next) = value.get(key) else {
                break;
            };
            if depth + 1 == keys.len() {
                anyhow::bail!("`extra` may not set `{path}`");
            }
            if !next.is_object() {
                let prefix = keys[..=depth].join(".");
                anyhow::bail!("`extra` may not replace `{prefix}`, which holds `{path}`");
            }
            value = next;
        }
    }
    merge(body, extra);
    Ok(())
}

/// Seconds to wait before retry `attempt`, counted from 0: what `retry-after` asks, else 1, 2
/// then 4, and never more than 10.
fn retry_wait_secs(headers: &reqwest::header::HeaderMap, attempt: u32) -> u64 {
    const MAX_WAIT_SECS: u64 = 10;
    headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(1u64 << attempt)
        .min(MAX_WAIT_SECS)
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
    use crate::model::{GeminiImageOptions, OpenAIImageOptions, OpenRouterImageOptions};

    #[test]
    fn read_options_reads_any_options_as_the_api_parameters() {
        // Another API's options type, where the fields fit.
        let options = OpenAIImageOptions {
            n: Some(2),
            size: Some("1536x1024".to_owned()),
            ..Default::default()
        };
        let read: OpenRouterImageOptions = read_options(Some(&options), "m").unwrap();
        assert_eq!(read.n, Some(2));
        assert_eq!(read.size.as_deref(), Some("1536x1024"));

        // Plain JSON, and none at all.
        let read: GeminiImageOptions =
            read_options(Some(&serde_json::json!({"aspect_ratio": "16:9"})), "m").unwrap();
        assert_eq!(read.aspect_ratio.as_deref(), Some("16:9"));
        let none: GeminiImageOptions = read_options(None, "m").unwrap();
        assert!(none.aspect_ratio.is_none() && none.extra.is_none());

        // Anything serializable compiles, and anything but an object fails when read.
        let err = read_options::<GeminiImageOptions>(Some(&"16:9"), "m")
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid type"), "{err}");

        // A field the API does not have is named, with the model and the fields it has.
        let options = OpenRouterImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        };
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
    fn retry_waits_as_asked_else_doubling_and_never_past_ten_seconds() {
        use reqwest::header::{HeaderMap, HeaderValue};

        let none = HeaderMap::new();
        let waits: Vec<_> = (0..3)
            .map(|attempt| retry_wait_secs(&none, attempt))
            .collect();
        assert_eq!(waits, [1, 2, 4]);

        let asking = |secs: &'static str| {
            let mut headers = HeaderMap::new();
            headers.insert("retry-after", HeaderValue::from_static(secs));
            headers
        };
        assert_eq!(retry_wait_secs(&asking("3"), 0), 3);
        assert_eq!(retry_wait_secs(&asking("0"), 2), 0);
        assert_eq!(retry_wait_secs(&asking("30"), 0), 10);
        // A date, which is not read, counts as no ask.
        assert_eq!(
            retry_wait_secs(&asking("Wed, 21 Oct 2026 07:28:00 GMT"), 1),
            2
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
        // Anything but an object on the way would wipe the reserved path out.
        for extra in [serde_json::json!({"a": null}), serde_json::json!({"a": 1})] {
            let extra = crate::datatype::Value::from(extra);
            let err = merge_extra(&mut body, Some(&extra), &["a.b"]).unwrap_err();
            assert_eq!(
                err.to_string(),
                "`extra` may not replace `a`, which holds `a.b`"
            );
        }

        let err = merge_extra(&mut body, Some(&crate::to_value!([1])), &[]).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"), "{err}");
    }
}
