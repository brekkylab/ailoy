//! An image model that calls the OpenAI Images API (`/v1/images/generations`) directly
//! over HTTP.

use anyhow::{Context as _, bail};
use futures::future::BoxFuture;

use crate::{
    datatype::{Bytes, Value},
    message::{PartImage, TokenUsage},
    model::{ImageOptions, ImageOutput, InferImageModel, OpenAIImageOptions},
    to_value,
};

const API_URL: &str = "https://api.openai.com/v1/images/generations";

/// Generates images with OpenAI's `gpt-image` models through the Images API.
///
/// Images come back as base64 bytes. `response_format` is never sent: these models answer
/// it with `unknown_parameter`.
#[derive(Clone)]
pub struct GptImageApiModel {
    api_key: String,
    model: String,
}

impl std::fmt::Debug for GptImageApiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GptImageApiModel")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

impl GptImageApiModel {
    /// Runs `model`, an API model id (e.g. "gpt-image-1"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `OPENAI_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("OPENAI_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The request body for one call. A combination the API never accepts fails here, before
    /// anything is sent.
    fn body(
        &self,
        prompt: &str,
        options: &OpenAIImageOptions,
    ) -> anyhow::Result<serde_json::Value> {
        let mut body = to_value!({
            "model": self.model.as_str(),
            "prompt": prompt,
        });
        let obj = body.as_object_mut().unwrap();

        if let Some(n) = options.n {
            if !(1..=10).contains(&n) {
                bail!("OpenAI accepts n between 1 and 10; requested n = {n}");
            }
            obj.insert("n".into(), u64::from(n).into());
        }
        let is = |value: &Option<String>, want: &str| {
            value
                .as_deref()
                .is_some_and(|v| v.eq_ignore_ascii_case(want))
        };
        if is(&options.background, "transparent") && is(&options.output_format, "jpeg") {
            bail!(
                "a transparent background needs an alpha-capable format; \
                 `output_format: jpeg` cannot carry one (use png or webp)"
            );
        }
        for (key, value) in [
            ("size", &options.size),
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
        Ok(body)
    }
}

impl InferImageModel for GptImageApiModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        let request =
            super::read_options::<OpenAIImageOptions>(options, &self.model).and_then(|options| {
                let asked = options.n.unwrap_or(1) as usize;
                Ok((self.body(prompt, &options)?, asked))
            });
        let api_key = self.api_key.clone();
        Box::pin(async move {
            let (body, asked) = request?;
            let response = send(API_URL, &api_key, &body).await?;
            let val: Value = serde_json::from_str(&response.text().await?)
                .context("OpenAI image response is not JSON")?;
            let out = parse_response(&val)?;
            if out.images.len() < asked {
                log::warn!("asked OpenAI for {asked} images, got {}", out.images.len());
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

/// POSTs `body` to `url`, retrying rate limits (429) and overloads (503, 529) with
/// backoff, and returns the successful response unread. Other errors are not retried: a
/// paid image job may already have run.
async fn send(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    // Rate limits, and overloads where the request was not taken on.
    const RETRY_STATUSES: [u16; 3] = [429, 503, 529];

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
            let text = response.text().await.unwrap_or_default();
            // Running out of quota or credit does not pass with time.
            if is_permanent_quota(&text) {
                bail!("OpenAI API request failed with status {status}: {text}");
            }
            log::warn!(
                "Request failed with status {status}. Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        bail!("OpenAI API request failed with status {status}: {text}");
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Whether an error body reports a used-up quota or credit rather than a passing limit.
fn is_permanent_quota(body: &str) -> bool {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let error = &json["error"];
    error["type"] == "insufficient_quota" || error["code"] == "insufficient_quota"
}

/// Reads a successful response; one without an image is an error.
fn parse_response(val: &Value) -> anyhow::Result<ImageOutput> {
    if let Some(message) = val.pointer("/error/message").and_then(|v| v.as_str()) {
        bail!("OpenAI image request failed: {message}");
    }
    let data = val
        .pointer("/data")
        .and_then(|v| v.as_array())
        .context("Missing data[] in OpenAI image response")?;
    if data.is_empty() {
        bail!("OpenAI returned no images (empty `data`)");
    }

    // The format produced is echoed at the top level; without it the bytes are sniffed,
    // and PNG is the API's default.
    let echoed_mime = val
        .pointer("/output_format")
        .and_then(|v| v.as_str())
        .and_then(format_mime);

    let mut images = Vec::with_capacity(data.len());
    let mut undecodable = 0usize;
    for entry in data {
        if let Some(b64) = entry.pointer("/b64_json").and_then(|v| v.as_str()) {
            // A broken entry does not sink the response; another may still be good.
            let Ok(data) = Bytes::from_base64(b64) else {
                log::warn!("skipping an OpenAI image entry whose `b64_json` does not decode");
                undecodable += 1;
                continue;
            };
            let mime_type = echoed_mime
                .map(str::to_owned)
                .or_else(|| infer::get(data.as_ref()).map(|t| t.mime_type().to_owned()))
                .unwrap_or_else(|| "image/png".to_owned());
            images.push(PartImage::Embedded { mime_type, data });
        } else if let Some(url) = entry.pointer("/url").and_then(|v| v.as_str()) {
            // gpt-image always answers with bytes; a link is taken in case one comes.
            images.push(PartImage::Url {
                url: url.to_owned(),
            });
        } else {
            log::warn!("skipping an OpenAI image entry with neither `b64_json` nor `url`");
            undecodable += 1;
        }
    }
    if images.is_empty() {
        bail!("OpenAI returned no usable image ({undecodable} entry(ies) had no decodable data)");
    }

    let text = val
        .pointer("/data/0/revised_prompt")
        .and_then(|v| v.as_str())
        .map(str::to_owned);

    Ok(ImageOutput {
        images,
        text,
        usage: val.pointer("/usage").and_then(parse_usage),
    })
}

/// The media type of an `output_format` value, if it is one this knows.
fn format_mime(format: &str) -> Option<&'static str> {
    Some(match format.to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        _ => return None,
    })
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let u = usage.as_object()?;
    let count = |key: &str| u.get(key).and_then(|v| v.as_unsigned()).unwrap_or(0);
    Some(TokenUsage {
        input_tokens: count("input_tokens"),
        output_tokens: count("output_tokens"),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GeminiImageOptions, ImageModelOptions, OpenRouterImageOptions};

    /// An 8-byte PNG header, enough for `infer` to recognise the format.
    const PNG_HEADER_B64: &str = "iVBORw0KGgo=";

    fn body(options: &OpenAIImageOptions) -> anyhow::Result<serde_json::Value> {
        GptImageApiModel::new("gpt-image-1", "sk-test").body("a red cube on a white table", options)
    }

    #[test]
    fn minimal_body_sends_only_model_and_prompt() {
        let body = body(&OpenAIImageOptions::default()).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"model": "gpt-image-1", "prompt": "a red cube on a white table"})
        );
    }

    #[tokio::test]
    async fn generate_takes_only_fields_the_openai_api_has() {
        let model = GptImageApiModel::new("gpt-image-1", "sk-test");
        // The key is fake, so anything that reached the network would fail differently.
        // One API's options type is passed as it is, as is `ImageModelOptions`.
        let gemini = GeminiImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        };
        for options in [
            &gemini as &dyn ImageOptions,
            &ImageModelOptions::from(gemini.clone()),
        ] {
            let err = model
                .generate("a red cube", Some(options))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("unknown field `aspect_ratio`"), "{err}");
        }

        // Fields OpenAI has too are read, whichever API the options were built for.
        let options = ImageModelOptions::OpenRouter(OpenRouterImageOptions {
            n: Some(0),
            ..Default::default()
        });
        let err = model
            .generate("a red cube", Some(&options))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("OpenAI accepts n between 1 and 10"),
            "{err}"
        );
    }

    #[test]
    fn body_maps_every_openai_option() {
        let options = OpenAIImageOptions {
            n: Some(3),
            size: Some("1536x1024".to_owned()),
            quality: Some("high".to_owned()),
            output_format: Some("webp".to_owned()),
            background: Some("transparent".to_owned()),
            extra: None,
        };
        assert_eq!(
            body(&options).unwrap(),
            serde_json::json!({
                "model": "gpt-image-1",
                "prompt": "a red cube on a white table",
                "n": 3,
                "size": "1536x1024",
                "quality": "high",
                "output_format": "webp",
                "background": "transparent",
            }),
            "never response_format"
        );
    }

    #[test]
    fn body_rejects_an_out_of_range_n() {
        for n in [0, 11] {
            let options = OpenAIImageOptions {
                n: Some(n),
                ..Default::default()
            };
            let err = body(&options).unwrap_err().to_string();
            assert!(err.contains("between 1 and 10"), "{err}");
        }
        for n in [1, 10] {
            let options = OpenAIImageOptions {
                n: Some(n),
                ..Default::default()
            };
            assert_eq!(body(&options).unwrap()["n"], n);
        }
    }

    #[test]
    fn body_rejects_transparent_background_on_jpeg() {
        let options = OpenAIImageOptions {
            output_format: Some("JPEG".to_owned()),
            background: Some("Transparent".to_owned()),
            ..Default::default()
        };
        let err = body(&options).unwrap_err().to_string();
        assert!(
            err.contains("transparent background"),
            "case does not matter: {err}"
        );

        let options = OpenAIImageOptions {
            background: Some("opaque".to_owned()),
            ..options
        };
        assert!(body(&options).is_ok());
    }

    #[test]
    fn extra_adds_fields_and_wins_over_typed_ones() {
        let options = OpenAIImageOptions {
            quality: Some("high".to_owned()),
            extra: Some(to_value!({"safety_identifier": "u-1", "quality": "low"})),
            ..Default::default()
        };
        let body = body(&options).unwrap();
        assert_eq!(body["safety_identifier"], "u-1");
        assert_eq!(body["quality"], "low", "extra wins on conflict");
    }

    #[test]
    fn extra_must_be_an_object_and_leave_reserved_keys_alone() {
        let with_extra = |extra| OpenAIImageOptions {
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
    fn body_sends_size_verbatim() {
        for size in ["1024x1024", "1920x1088", "auto"] {
            let options = OpenAIImageOptions {
                size: Some(size.to_owned()),
                ..Default::default()
            };
            assert_eq!(body(&options).unwrap()["size"], size);
        }
    }

    #[test]
    fn parse_a_gpt_image_response() {
        let response = to_value!({
            "created": 1713833628,
            "background": "opaque",
            "output_format": "png",
            "size": "1024x1024",
            "quality": "high",
            "data": [{"b64_json": PNG_HEADER_B64}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 60,
                "total_tokens": 110,
                "input_tokens_details": {"text_tokens": 10, "image_tokens": 40}
            }
        });

        let out = parse_response(&response).unwrap();
        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("expected an embedded image, got {:?}", out.images[0]);
        };
        assert_eq!(mime_type, "image/png");
        assert_eq!(&data.as_ref()[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert!(out.text.is_none());
        let usage = out.usage.expect("gpt-image reports usage");
        assert_eq!(usage.input_tokens, 50);
        assert_eq!(usage.output_tokens, 60);
    }

    #[test]
    fn parse_sniffs_the_mime_type_when_the_format_is_not_echoed() {
        let response = to_value!({
            "created": 1713833628,
            "data": [{"b64_json": PNG_HEADER_B64}]
        });

        let out = parse_response(&response).unwrap();
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(mime_type, "image/png");
        assert!(out.usage.is_none());
    }

    #[test]
    fn parse_a_url_only_response() {
        let response = to_value!({
            "created": 1713833628,
            "data": [{
                "url": "https://images.example.com/private/img.png",
                "revised_prompt": "A vivid red cube resting on a white table"
            }]
        });

        let out = parse_response(&response).unwrap();
        assert_eq!(
            out.images,
            vec![PartImage::Url {
                url: "https://images.example.com/private/img.png".to_owned()
            }]
        );
        assert_eq!(
            out.text.as_deref(),
            Some("A vivid red cube resting on a white table")
        );
    }

    #[test]
    fn parse_skips_an_undecodable_entry() {
        let response = to_value!({
            "output_format": "webp",
            "data": [{"b64_json": "not base64!"}, {"b64_json": PNG_HEADER_B64}]
        });
        let out = parse_response(&response).unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "the good entry survives its broken sibling"
        );
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(
            mime_type, "image/webp",
            "the echoed format wins over sniffing"
        );

        let err = parse_response(&to_value!({"data": [{"b64_json": "not base64!"}]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 entry(ies) had no decodable data"), "{err}");
    }

    #[test]
    fn parse_falls_back_to_sniffing_an_unknown_echoed_format() {
        let response = to_value!({
            "output_format": "bmp",
            "data": [{"b64_json": PNG_HEADER_B64}]
        });
        let out = parse_response(&response).unwrap();
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(mime_type, "image/png");
    }

    #[test]
    fn an_error_in_a_successful_response_is_reported() {
        let response = to_value!({"error": {"message": "the image could not be made"}});
        let err = parse_response(&response).unwrap_err().to_string();
        assert_eq!(
            err,
            "OpenAI image request failed: the image could not be made"
        );
    }

    #[test]
    fn endpoint_is_the_images_api() {
        assert_eq!(API_URL, "https://api.openai.com/v1/images/generations");
    }

    #[test]
    fn a_response_without_images_is_an_error() {
        let err = parse_response(&to_value!({"created": 1, "data": []}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no images"), "{err}");
        assert!(parse_response(&to_value!({"created": 1})).is_err());
    }

    #[test]
    fn permanent_quota_is_recognised() {
        assert!(is_permanent_quota(
            r#"{"error":{"message":"quota","type":"insufficient_quota"}}"#
        ));
        assert!(is_permanent_quota(
            r#"{"error":{"code":"insufficient_quota"}}"#
        ));
        assert!(!is_permanent_quota(
            r#"{"error":{"type":"rate_limit_exceeded"}}"#
        ));
        assert!(!is_permanent_quota("not json"));
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

    #[tokio::test]
    async fn send_retries_a_passing_rate_limit() {
        let (url, calls) = mock(|attempt| match attempt {
            1 | 2 => (429, r#"{"error":{"type":"rate_limit_exceeded"}}"#),
            _ => (200, r#"{"created":1,"data":[{"b64_json":"iVBORw0KGgo="}]}"#),
        })
        .await;

        let response = send(&url, "sk-test", &serde_json::json!({})).await.unwrap();
        let out = parse_response(&serde_json::from_str(&response.text().await.unwrap()).unwrap())
            .unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 3);
        assert_eq!(out.images.len(), 1);
    }

    #[tokio::test]
    async fn send_retries_an_overload() {
        let (url, calls) = mock(|attempt| match attempt {
            1 => (503, r#"{"error":{"message":"overloaded"}}"#),
            _ => (200, "{}"),
        })
        .await;

        send(&url, "sk-test", &serde_json::json!({})).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn send_does_not_retry_a_server_error() {
        let (url, calls) = mock(|_| (500, r#"{"error":{"message":"internal"}}"#)).await;

        let err = send(&url, "sk-test", &serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("500"), "{err}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn generate_refuses_a_bad_n_before_sending() {
        let options = ImageModelOptions::OpenAI(OpenAIImageOptions {
            n: Some(0),
            ..Default::default()
        });
        // The key is fake, so anything that reached the network would fail differently.
        let err = GptImageApiModel::new("gpt-image-1", "sk-test")
            .generate("a red cube", Some(&options))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("between 1 and 10"), "{err}");
    }

    #[tokio::test]
    async fn send_authenticates_with_a_bearer_token() {
        use axum::{Router, http::HeaderMap, http::StatusCode, routing::post};

        let app = Router::new().route(
            "/",
            post(|headers: HeaderMap| async move {
                if headers.get("authorization").and_then(|v| v.to_str().ok())
                    == Some("Bearer sk-test")
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

        send(
            &format!("http://{addr}/"),
            "sk-test",
            &serde_json::json!({}),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn send_does_not_retry_a_used_up_quota() {
        let (url, calls) = mock(|_| {
            (
                429,
                r#"{"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"You exceeded your current quota"}}"#,
            )
        })
        .await;

        let err = send(&url, "sk-test", &serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("insufficient_quota"), "{err}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[ignore = "spends OpenAI credit and needs OPENAI_API_KEY; run with --ignored"]
    async fn live_generate() {
        dotenvy::dotenv().ok();
        let Ok(model) = GptImageApiModel::from_env("gpt-image-1-mini") else {
            eprintln!("OPENAI_API_KEY is not set; skipping");
            return;
        };
        let options = ImageModelOptions::OpenAI(OpenAIImageOptions {
            quality: Some("low".to_owned()),
            ..Default::default()
        });

        let out = model
            .generate(
                "A single red cube on a plain white background",
                Some(&options),
            )
            .await
            .unwrap_or_else(|e| panic!("OpenAI image generation failed: {e:#}"));

        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("gpt-image must return bytes, got {:?}", out.images[0]);
        };
        assert_eq!(mime_type, "image/png");
        assert_eq!(&data.as_ref()[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert!(out.usage.is_some(), "gpt-image reports token usage");
    }
}
