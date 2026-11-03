//! An image model that calls the OpenRouter Images API (`/api/v1/images`) directly over
//! HTTP.

use anyhow::{Context as _, bail};
use futures::future::BoxFuture;

use crate::{
    datatype::{Bytes, Value},
    message::{PartImage, TokenUsage},
    model::{ImageOptions, ImageOutput, InferImageModel, OpenRouterImageOptions},
    to_value,
};

const API_URL: &str = "https://openrouter.ai/api/v1/images";

/// Generates images with any image model OpenRouter serves, named by its OpenRouter slug
/// (`<vendor>/<model>`).
#[derive(Clone)]
pub struct OpenRouterImageApiModel {
    api_key: String,
    model: String,
}

impl std::fmt::Debug for OpenRouterImageApiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterImageApiModel")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

impl OpenRouterImageApiModel {
    /// Runs `model`, an OpenRouter slug (e.g. "openai/gpt-image-1"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `OPENROUTER_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("OPENROUTER_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The request body for one call. A combination OpenRouter never accepts fails here,
    /// before anything is sent.
    fn body(
        &self,
        prompt: &str,
        options: &OpenRouterImageOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body = to_value!({
            "model": self.model.as_str(),
            "prompt": prompt,
        });
        let obj = body.as_object_mut().unwrap();

        if let Some(n) = options.n {
            obj.insert("n".into(), u64::from(n).into());
        }
        for (key, value) in [
            ("size", &options.size),
            ("resolution", &options.resolution),
            ("aspect_ratio", &options.aspect_ratio),
            ("quality", &options.quality),
            ("output_format", &options.output_format),
            ("background", &options.background),
        ] {
            if let Some(value) = value {
                obj.insert(key.into(), value.as_str().into());
            }
        }

        let mut body = body.into();
        // The response is read whole, so it may not be streamed.
        super::merge_extra(
            &mut body,
            options.extra.as_ref(),
            &["model", "prompt", "stream"],
        )?;

        // Checked as sent, so a value `extra` set is checked too.
        if let Some(n) = body.get("n")
            && !n.as_u64().is_some_and(|n| (1..=10).contains(&n))
        {
            bail!("OpenRouter accepts n between 1 and 10; requested n = {n}");
        }
        let is = |key: &str, want: &str| {
            body.get(key)
                .and_then(|v| v.as_str())
                .is_some_and(|v| v.eq_ignore_ascii_case(want))
        };
        if is("background", "transparent") && is("output_format", "jpeg") {
            bail!(
                "a transparent background needs an alpha-capable format; \
                 `output_format: jpeg` cannot carry one (use png or webp)"
            );
        }
        Ok(body)
    }
}

impl InferImageModel for OpenRouterImageApiModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let request = super::read_options::<OpenRouterImageOptions>(options, &self.model)
            .and_then(|options| self.body(prompt, &options));
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        Box::pin(async move {
            let body = request?;
            // Read from the body as sent, `extra` and all.
            let asked = body.get("n").and_then(|n| n.as_u64()).unwrap_or(1) as usize;
            let resolution = body.get("resolution").and_then(|r| r.as_str());
            let size_set = body.get("size").is_some();
            let response = send(API_URL, &api_key, &body, &model, resolution, size_set).await?;
            let val: Value = serde_json::from_str(&response.text().await?)
                .context("OpenRouter image response is not JSON")?;
            let out = parse_response(&val)?;
            if out.images.len() < asked {
                log::warn!(
                    "asked OpenRouter for {asked} images, got {}",
                    out.images.len()
                );
            }
            Ok(out)
        })
    }
}

/// One client for every call, so connections are reused. Generating an image can take
/// minutes, so the read timeout is long.
static CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .read_timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("the HTTP client builds")
});

/// POSTs `body` to `url`, retrying rate limits (429), failed generations (502, which
/// OpenRouter does not bill) and overloads (503, 529) with backoff, and returns the
/// successful response unread. Other errors are not retried: a paid image job may already
/// have run. A refusal of `resolution` is explained in front of the API's own message.
async fn send(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    model: &str,
    resolution: Option<&str>,
    size_set: bool,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    // Rate limits, unbilled failures, and overloads where the request was not taken on.
    const RETRY_STATUSES: [u16; 4] = [429, 502, 503, 529];

    for attempt in 0..=MAX_RETRIES {
        let response = CLIENT
            .post(url)
            .bearer_auth(api_key)
            .json(body)
            .send()
            .await?;

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if RETRY_STATUSES.contains(&status.as_u16()) && attempt < MAX_RETRIES {
            let wait_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1u64 << attempt)
                .min(MAX_WAIT_SECS);
            log::warn!(
                "Request failed with status {status}. Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        match explain_error(model, resolution, size_set, status.as_u16(), &text) {
            Some(hint) => {
                bail!("OpenRouter API request failed with status {status} ({hint}): {text}")
            }
            None => bail!("OpenRouter API request failed with status {status}: {text}"),
        }
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Names the option behind a refusal of `resolution`. With `size` also set, the refusal may
/// be about either, so nothing is added.
fn explain_error(
    model: &str,
    resolution: Option<&str>,
    size_set: bool,
    status: u16,
    body: &str,
) -> Option<String> {
    let resolution = resolution?;
    if size_set || status != 400 || !body.contains("resolution") {
        return None;
    }
    Some(format!(
        "`resolution: {}` is not accepted by '{model}'; try another resolution, \
         or leave it unset to use the model's default size",
        resolution,
    ))
}

/// Reads a successful response; one without an image is an error.
fn parse_response(val: &Value) -> anyhow::Result<ImageOutput> {
    if let Some(message) = val.pointer("/error/message").and_then(|v| v.as_str()) {
        bail!("OpenRouter image request failed: {message}");
    }
    let data = val
        .pointer("/data")
        .and_then(|v| v.as_array())
        .context("Missing data[] in OpenRouter image response")?;
    if data.is_empty() {
        bail!("OpenRouter returned no images (empty `data`)");
    }

    let mut images = Vec::with_capacity(data.len());
    let mut undecodable = 0usize;
    for entry in data {
        // A broken entry does not sink the response; another may still be good.
        let Some(data) = entry
            .pointer("/b64_json")
            .and_then(|v| v.as_str())
            .and_then(|b64| Bytes::from_base64(b64).ok())
        else {
            log::warn!("skipping an OpenRouter image entry with no decodable `b64_json`");
            undecodable += 1;
            continue;
        };
        // `media_type` is left out only when OpenRouter does not know it.
        let mime_type = entry
            .pointer("/media_type")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
            .or_else(|| infer::get(data.as_ref()).map(|t| t.mime_type().to_owned()))
            .unwrap_or_else(|| fallback_mime(data.as_ref()).to_owned());
        images.push(PartImage::Embedded { mime_type, data });
    }
    if images.is_empty() {
        bail!(
            "OpenRouter returned no usable image ({undecodable} entry(ies) had no decodable data)"
        );
    }

    Ok(ImageOutput {
        images,
        text: None,
        usage: val.pointer("/usage").and_then(parse_usage),
    })
}

/// The type of bytes the sniffer does not know: SVG, which is text, or else PNG.
fn fallback_mime(data: &[u8]) -> &'static str {
    let head = String::from_utf8_lossy(&data[..data.len().min(512)]);
    if head.contains("<svg") {
        "image/svg+xml"
    } else {
        "image/png"
    }
}

/// The token counts; the `cost` beside them has no place in [`TokenUsage`].
fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let u = usage.as_object()?;
    let count = |key: &str| u.get(key).and_then(|v| v.as_unsigned()).unwrap_or(0);
    Some(TokenUsage {
        input_tokens: count("prompt_tokens"),
        output_tokens: count("completion_tokens"),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GeminiImageOptions, OpenAIImageOptions};

    /// An 8-byte PNG header, enough for `infer` to recognise the format.
    const PNG_HEADER_B64: &str = "iVBORw0KGgo=";

    fn body(options: &OpenRouterImageOptions) -> anyhow::Result<serde_json::Value> {
        OpenRouterImageApiModel::new("google/gemini-3.1-flash-image", "sk-or-test")
            .body("a red cube on a white table", options)
    }

    #[test]
    fn minimal_body_sends_only_model_and_prompt() {
        assert_eq!(
            body(&OpenRouterImageOptions::default()).unwrap(),
            serde_json::json!({
                "model": "google/gemini-3.1-flash-image",
                "prompt": "a red cube on a white table",
            })
        );
    }

    #[tokio::test]
    async fn generate_takes_only_fields_the_openrouter_api_has() {
        let model = OpenRouterImageApiModel::new("google/gemini-3.1-flash-image", "sk-or-test");
        // Gemini's names for a size tier and for thinking are not OpenRouter's; both are
        // refused before anything is sent, which the fake key would fail otherwise.
        for (gemini, field) in [
            (
                GeminiImageOptions {
                    image_size: Some("2K".to_owned()),
                    ..Default::default()
                },
                "image_size",
            ),
            (
                GeminiImageOptions {
                    thinking_level: Some("high".to_owned()),
                    ..Default::default()
                },
                "thinking_level",
            ),
        ] {
            let err = model
                .generate("a red cube", Some(&gemini))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("unknown field `{field}`")), "{err}");
        }

        // Options built for OpenAI are read where OpenRouter has the same fields.
        let options = OpenAIImageOptions {
            n: Some(11),
            ..Default::default()
        };
        let err = model
            .generate("a red cube", Some(&options))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("OpenRouter accepts n between 1 and 10"),
            "{err}"
        );
    }

    #[test]
    fn body_maps_every_option_under_openrouter_names() {
        let options = OpenRouterImageOptions {
            n: Some(2),
            size: Some("1024x1024".to_owned()),
            quality: Some("auto".to_owned()),
            output_format: Some("webp".to_owned()),
            background: Some("transparent".to_owned()),
            aspect_ratio: Some("9:19.5".to_owned()),
            resolution: Some("1.5K".to_owned()),
            extra: None,
        };
        assert_eq!(
            body(&options).unwrap(),
            serde_json::json!({
                "model": "google/gemini-3.1-flash-image",
                "prompt": "a red cube on a white table",
                "n": 2,
                "size": "1024x1024",
                "resolution": "1.5K",
                "aspect_ratio": "9:19.5",
                "quality": "auto",
                "output_format": "webp",
                "background": "transparent",
            })
        );
    }

    #[test]
    fn body_sends_one_option_alone() {
        let cases = [
            (
                OpenRouterImageOptions {
                    resolution: Some("768".to_owned()),
                    ..Default::default()
                },
                "resolution",
                serde_json::json!("768"),
            ),
            (
                OpenRouterImageOptions {
                    aspect_ratio: Some("auto".to_owned()),
                    ..Default::default()
                },
                "aspect_ratio",
                serde_json::json!("auto"),
            ),
            (
                OpenRouterImageOptions {
                    output_format: Some("svg".to_owned()),
                    ..Default::default()
                },
                "output_format",
                serde_json::json!("svg"),
            ),
            (
                OpenRouterImageOptions {
                    background: Some("auto".to_owned()),
                    ..Default::default()
                },
                "background",
                serde_json::json!("auto"),
            ),
        ];
        for (options, key, value) in cases {
            let body = body(&options).unwrap();
            let obj = body.as_object().unwrap();
            assert_eq!(obj.len(), 3, "only {key} beside model and prompt: {body}");
            assert_eq!(obj[key], value);
        }
    }

    #[test]
    fn extra_adds_fields_and_wins_over_typed_ones() {
        let options = OpenRouterImageOptions {
            quality: Some("high".to_owned()),
            extra: Some(to_value!({"user": "u-1", "quality": "low"})),
            ..Default::default()
        };
        let body = body(&options).unwrap();
        assert_eq!(body["user"], "u-1");
        assert_eq!(body["quality"], "low", "extra wins on conflict");
    }

    #[test]
    fn extra_must_be_an_object_and_leave_reserved_keys_alone() {
        let with_extra = |extra| OpenRouterImageOptions {
            extra: Some(extra),
            ..Default::default()
        };
        let err = body(&with_extra(to_value!(["user"])))
            .unwrap_err()
            .to_string();
        assert!(err.contains("`extra` must be a JSON object"), "{err}");
        for key in ["model", "prompt", "stream"] {
            let extra = crate::datatype::Value::object([(key, true)]);
            let err = body(&with_extra(extra)).unwrap_err().to_string();
            assert_eq!(err, format!("`extra` may not set `{key}`"));
        }
    }

    #[test]
    fn body_refuses_what_openrouter_never_accepts() {
        for n in [0, 11] {
            let options = OpenRouterImageOptions {
                n: Some(n),
                ..Default::default()
            };
            let err = body(&options).unwrap_err().to_string();
            assert!(err.contains("between 1 and 10"), "{err}");
        }
        let options = OpenRouterImageOptions {
            output_format: Some("jpeg".to_owned()),
            background: Some("transparent".to_owned()),
            ..Default::default()
        };
        let err = body(&options).unwrap_err().to_string();
        assert!(err.contains("transparent background"), "{err}");

        // `extra` is checked as well, as the body is sent.
        let options = OpenRouterImageOptions {
            extra: Some(to_value!({"n": 50})),
            ..Default::default()
        };
        let err = body(&options).unwrap_err().to_string();
        assert!(err.contains("requested n = 50"), "{err}");
        let options = OpenRouterImageOptions {
            background: Some("transparent".to_owned()),
            extra: Some(to_value!({"output_format": "JPEG"})),
            ..Default::default()
        };
        let err = body(&options).unwrap_err().to_string();
        assert!(err.contains("transparent background"), "{err}");
    }

    #[test]
    fn parse_takes_the_media_type_and_usage() {
        let response = to_value!({
            "data": [{"b64_json": PNG_HEADER_B64, "media_type": "image/webp"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1290, "total_tokens": 1302, "cost": 0.04}
        });
        let out = parse_response(&response).unwrap();
        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("expected an embedded image, got {:?}", out.images[0]);
        };
        assert_eq!(
            mime_type, "image/webp",
            "the reported type wins over sniffing"
        );
        assert_eq!(&data.as_ref()[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert!(out.text.is_none());
        let usage = out.usage.expect("usage is parsed");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 1290);
    }

    #[test]
    fn parse_sniffs_a_missing_media_type() {
        let svg = Bytes::from(b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>".to_vec()).base64();
        let response = to_value!({
            "data": [{"b64_json": PNG_HEADER_B64}, {"b64_json": svg}]
        });
        let out = parse_response(&response).unwrap();
        let mimes: Vec<&str> = out
            .images
            .iter()
            .map(|image| match image {
                PartImage::Embedded { mime_type, .. } => mime_type.as_str(),
                other => panic!("expected an embedded image, got {other:?}"),
            })
            .collect();
        assert_eq!(mimes, ["image/png", "image/svg+xml"]);
        assert!(out.usage.is_none());
    }

    #[test]
    fn parse_skips_an_undecodable_entry() {
        let response = to_value!({
            "data": [{"b64_json": "not base64!"}, {"media_type": "image/png"}, {"b64_json": PNG_HEADER_B64}]
        });
        assert_eq!(parse_response(&response).unwrap().images.len(), 1);

        let err = parse_response(&to_value!({"data": [{"b64_json": "not base64!"}]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 entry(ies) had no decodable data"), "{err}");
    }

    #[test]
    fn an_error_in_a_successful_response_is_reported() {
        let response = to_value!({"error": {"code": 502, "message": "Provider returned error"}});
        let err = parse_response(&response).unwrap_err().to_string();
        assert_eq!(
            err,
            "OpenRouter image request failed: Provider returned error"
        );
    }

    #[test]
    fn endpoint_is_the_images_api() {
        assert_eq!(API_URL, "https://openrouter.ai/api/v1/images");
    }

    #[test]
    fn a_response_without_images_is_an_error() {
        let err = parse_response(&to_value!({"data": []}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no images"), "{err}");
        assert!(parse_response(&to_value!({})).is_err());
    }

    #[test]
    fn explain_error_names_the_resolution_option() {
        const REFUSAL: &str =
            r#"{"error":{"code":400,"message":"Unsupported resolution '4K' for this model"}}"#;
        let model = "bytedance-seed/seedream-5-0-pro";

        let hint = explain_error(model, Some("4K"), false, 400, REFUSAL)
            .expect("a resolution refusal is explained");
        assert!(hint.contains("`resolution: 4K`"), "{hint}");
        assert!(hint.contains(model), "{hint}");

        assert!(explain_error(model, None, false, 400, REFUSAL).is_none());
        assert!(explain_error(model, Some("4K"), false, 402, REFUSAL).is_none());
        // With `size` set too, the refusal may be about either.
        assert!(explain_error(model, Some("4K"), true, 400, REFUSAL).is_none());
        assert!(
            explain_error(
                model,
                Some("4K"),
                false,
                400,
                r#"{"error":{"code":400,"message":"bad aspect_ratio"}}"#
            )
            .is_none()
        );
    }

    /// Serves `respond(attempt)` for every POST, counting attempts from 1.
    async fn mock(
        respond: fn(u32) -> (u16, &'static str),
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicU32>) {
        use std::sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        };

        use axum::{Router, response::Response, routing::post};

        let calls = Arc::new(AtomicU32::new(0));
        let counter = calls.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
                async move {
                    let (status, body) = respond(attempt);
                    Response::builder()
                        .status(status)
                        .header("retry-after", "0")
                        .body(axum::body::Body::from(body))
                        .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });
        (format!("http://{addr}/"), calls)
    }

    async fn send_to(url: &str, resolution: Option<&str>) -> anyhow::Result<reqwest::Response> {
        send(
            url,
            "sk-or-test",
            &serde_json::json!({}),
            "openai/gpt-image-1",
            resolution,
            false,
        )
        .await
    }

    #[tokio::test]
    async fn send_authenticates_with_a_bearer_token() {
        use axum::{Router, http::HeaderMap, http::StatusCode, routing::post};

        let app = Router::new().route(
            "/",
            post(|headers: HeaderMap| async move {
                if headers.get("authorization").and_then(|v| v.to_str().ok())
                    == Some("Bearer sk-or-test")
                {
                    StatusCode::OK
                } else {
                    StatusCode::UNAUTHORIZED
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        send_to(&format!("http://{addr}/"), None).await.unwrap();
    }

    #[tokio::test]
    async fn send_retries_a_failed_generation() {
        let (url, calls) = mock(|attempt| match attempt {
            1 => (
                502,
                r#"{"error":{"code":502,"message":"generation failed"}}"#,
            ),
            _ => (200, r#"{"data":[{"b64_json":"iVBORw0KGgo="}]}"#),
        })
        .await;

        let response = send_to(&url, None).await.unwrap();
        let out = parse_response(&serde_json::from_str(&response.text().await.unwrap()).unwrap())
            .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
        assert_eq!(out.images.len(), 1);
    }

    #[tokio::test]
    async fn send_does_not_retry_a_server_error() {
        let (url, calls) = mock(|_| (500, r#"{"error":{"code":500,"message":"internal"}}"#)).await;

        let err = send_to(&url, None).await.unwrap_err();
        assert!(err.to_string().contains("500"), "{err}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn generate_refuses_a_bad_n_before_sending() {
        let options = OpenRouterImageOptions {
            n: Some(11),
            ..Default::default()
        };
        // The key is fake, so anything that reached the network would fail differently.
        let err = OpenRouterImageApiModel::new("openai/gpt-image-1", "sk-or-test")
            .generate("a red cube", Some(&options))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("between 1 and 10"), "{err}");
    }

    #[tokio::test]
    #[ignore = "spends OpenRouter credit and needs OPENROUTER_API_KEY; run with --ignored"]
    async fn live_generate() {
        dotenvy::dotenv().ok();
        let Ok(model) = OpenRouterImageApiModel::from_env("google/gemini-3.1-flash-image") else {
            eprintln!("OPENROUTER_API_KEY is not set; skipping");
            return;
        };
        let options = OpenRouterImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        };

        let out = model
            .generate(
                "A single red cube on a plain white background",
                Some(&options),
            )
            .await
            .unwrap_or_else(|e| panic!("OpenRouter image generation failed: {e:#}"));

        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("OpenRouter must return bytes, got {:?}", out.images[0]);
        };
        assert!(mime_type.starts_with("image/"), "{mime_type}");
        let decoded = image::load_from_memory(data.as_ref()).expect("the bytes are an image");
        let (width, height) = image::GenericImageView::dimensions(&decoded);
        assert!(
            width > height,
            "16:9 must be landscape, got {width}x{height}"
        );
        assert!(out.usage.is_some(), "OpenRouter reports usage");
    }
}
