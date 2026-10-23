//! An image model that calls the Gemini API (`generativelanguage.googleapis.com`)
//! `:generateContent` directly over HTTP, asking for the `IMAGE` response modality.

use anyhow::{Context as _, bail};
use futures::future::BoxFuture;

use crate::{
    datatype::{Bytes, Value},
    message::{PartImage, TokenUsage},
    model::{AspectRatio, ImageModelOptions, ImageOutput, ImageSize, InferImageModel},
    to_value,
};

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Generates images with Gemini's image models through the Gemini API.
///
/// Only image models answer with an image; a text model refuses the `IMAGE` modality.
#[derive(Clone)]
pub struct GeminiImageApiModel {
    api_key: String,
    model: String,
}

impl std::fmt::Debug for GeminiImageApiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiImageApiModel")
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .finish()
    }
}

impl GeminiImageApiModel {
    /// Runs `model`, an API model id (e.g. "gemini-3.1-flash-image"), with `api_key`.
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
        }
    }

    /// Runs `model` with the key read from `GEMINI_API_KEY`.
    pub fn from_env(model: impl Into<String>) -> anyhow::Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .context("GEMINI_API_KEY is not set")?;
        Ok(Self::new(model, api_key))
    }

    /// The request body for one call.
    fn body(&self, prompt: &str, options: &ImageModelOptions) -> serde_json::Value {
        // `aspect_ratio` goes out as `imageConfig.aspectRatio` and `image_size` as
        // `imageConfig.imageSize`; the other options are not read. Both are sent as asked:
        // a model that does not take a value says so, and `explain_error` ties that to the
        // option.
        let mut image_config = Value::object_empty();
        let image_obj = image_config.as_object_mut().unwrap();
        if let Some(ratio) = options.aspect_ratio {
            image_obj.insert("aspectRatio".into(), <&str>::from(ratio).into());
        }
        if let Some(size) = options.image_size {
            image_obj.insert("imageSize".into(), <&str>::from(size).into());
        }

        let mut config = to_value!({"responseModalities": ["TEXT", "IMAGE"]});
        if !image_obj.is_empty() {
            config
                .as_object_mut()
                .unwrap()
                .insert("imageConfig".into(), image_config);
        }

        to_value!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": config,
        })
        .into()
    }
}

impl InferImageModel for GeminiImageApiModel {
    fn generate(
        &self,
        prompt: &str,
        options: &ImageModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        // The id goes in the URL path, where these would change what is called.
        if self.model.contains(['/', '?', '#']) {
            let e = anyhow::anyhow!("a Gemini model id has no '/', '?' or '#': {}", self.model);
            return Box::pin(async move { Err(e) });
        }
        let body = self.body(prompt, options);
        let url = endpoint(&self.model);
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let (aspect_ratio, image_size) = (options.aspect_ratio, options.image_size);
        Box::pin(async move {
            let response = send(&url, &api_key, &body, &model, aspect_ratio, image_size).await?;
            let val: Value = serde_json::from_str(&response.text().await?)
                .context("Gemini image response is not JSON")?;
            parse_response(&val)
        })
    }
}

/// The `generateContent` URL for `model`.
fn endpoint(model: &str) -> String {
    format!("{API_BASE}/{model}:generateContent")
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
/// paid image job may already have run. A refusal of `aspect_ratio` or `image_size` is
/// explained in front of the API's own message.
async fn send(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    model: &str,
    aspect_ratio: Option<AspectRatio>,
    image_size: Option<ImageSize>,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    // Rate limits, and overloads where the request was not taken on.
    const RETRY_STATUSES: [u16; 3] = [429, 503, 529];

    for attempt in 0..=MAX_RETRIES {
        let response = CLIENT
            .post(url)
            .header("x-goog-api-key", api_key)
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
            // A quota that is used up does not pass with time.
            if status.as_u16() == 429 && is_permanent_quota(&text) {
                bail!("Gemini API request failed with status {status}: {text}");
            }
            log::warn!(
                "Request failed with status {status}. Retrying after {wait_secs}s (attempt {}/{MAX_RETRIES})",
                attempt + 1
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        match explain_error(model, aspect_ratio, image_size, status.as_u16(), &text) {
            Some(hint) => bail!("Gemini API request failed with status {status} ({hint}): {text}"),
            None => bail!("Gemini API request failed with status {status}: {text}"),
        }
    }
    unreachable!("retry loop returns or bails on every path")
}

/// Whether a 429 body reports a used-up quota: `RESOURCE_EXHAUSTED` covers both kinds,
/// and only a passing limit carries a `RetryInfo`.
fn is_permanent_quota(body: &str) -> bool {
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return false;
    };
    let error = &json["error"];
    error["status"] == "RESOURCE_EXHAUSTED"
        && !error["details"].as_array().into_iter().flatten().any(|d| {
            d["@type"]
                .as_str()
                .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
        })
}

/// Names the option behind a refusal, which the API reports by its wire field. Runs only
/// once the model has said no, so no table of what each model takes is kept.
fn explain_error(
    model: &str,
    aspect_ratio: Option<AspectRatio>,
    image_size: Option<ImageSize>,
    status: u16,
    body: &str,
) -> Option<String> {
    if status != 400 {
        return None;
    }
    let mut hints = Vec::new();
    if let Some(ratio) = aspect_ratio
        && body.contains("aspect_ratio")
    {
        hints.push(format!(
            "`aspect_ratio: {}` is not accepted by '{model}'; try another ratio, or leave it \
             unset to let the model choose",
            <&str>::from(ratio),
        ));
    }
    if let Some(size) = image_size
        && (body.contains("image_size")
            || (body.contains("Image size") && body.contains("not supported")))
    {
        hints.push(format!(
            "`image_size: {}` is not accepted by '{model}'; try another size, or leave it \
             unset to use the model's default size",
            <&str>::from(size),
        ));
    }
    (!hints.is_empty()).then(|| hints.join("; "))
}

/// Reads a successful response; one without a final image is an error quoting why.
fn parse_response(val: &Value) -> anyhow::Result<ImageOutput> {
    let mut images = Vec::new();
    let mut texts: Vec<&str> = Vec::new();
    let mut undecodable = 0usize;
    let mut drafts = 0usize;
    for part in val
        .pointer("/candidates/0/content/parts")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        let inline = part
            .pointer("/inlineData")
            .or_else(|| part.pointer("/inline_data"));
        // Thoughts, both text and draft images, are not the answer. They are not asked
        // for, but are kept out if they come.
        if part.pointer("/thought").and_then(|v| v.as_bool()) == Some(true) {
            drafts += usize::from(inline.is_some());
            continue;
        }
        if let Some(inline) = inline {
            // A broken part does not sink the response; a later one may still be good.
            let Some(data) = inline
                .pointer("/data")
                .and_then(|v| v.as_str())
                .and_then(|b64| Bytes::from_base64(b64).ok())
            else {
                log::warn!("skipping a Gemini inlineData part with no decodable `data`");
                undecodable += 1;
                continue;
            };
            let mime_type = inline
                .pointer("/mimeType")
                .or_else(|| inline.pointer("/mime_type"))
                .and_then(|v| v.as_str())
                .map(str::to_owned)
                .or_else(|| infer::get(data.as_ref()).map(|t| t.mime_type().to_owned()))
                .unwrap_or_else(|| "image/png".to_owned());
            images.push(PartImage::Embedded { mime_type, data });
        } else if let Some(text) = part.pointer("/text").and_then(|v| v.as_str()) {
            texts.push(text);
        }
    }
    // Separate parts are separate lines.
    let text = texts.join("\n");

    if images.is_empty() {
        let mut reasons = Vec::new();
        if let Some(reason) = val
            .pointer("/candidates/0/finishReason")
            .and_then(|v| v.as_str())
        {
            reasons.push(format!("finishReason: {reason}"));
        }
        if let Some(reason) = val
            .pointer("/promptFeedback/blockReason")
            .and_then(|v| v.as_str())
        {
            reasons.push(format!("promptFeedback.blockReason: {reason}"));
        }
        if drafts > 0 {
            reasons.push(format!(
                "{drafts} draft image(s) marked `thought` and no final image"
            ));
        }
        if undecodable > 0 {
            reasons.push(format!(
                "{undecodable} inlineData part(s) had no decodable data"
            ));
        }
        if !text.is_empty() {
            reasons.push(format!("text: {text}"));
        }
        if reasons.is_empty() {
            reasons.push("no reason reported".to_owned());
        }
        bail!("Gemini returned no image ({})", reasons.join("; "));
    }

    Ok(ImageOutput {
        images,
        text: (!text.is_empty()).then_some(text),
        usage: val.pointer("/usageMetadata").and_then(parse_usage),
    })
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let u = usage.as_object()?;
    let count = |key: &str| u.get(key).and_then(|v| v.as_unsigned());
    let input_tokens = count("promptTokenCount").unwrap_or(0);
    // Everything past the prompt is billed as output, thinking included, which is
    // reported apart from the candidates.
    let output_tokens = match count("totalTokenCount") {
        Some(total) if total >= input_tokens => total - input_tokens,
        _ => count("candidatesTokenCount").unwrap_or(0),
    };
    Some(TokenUsage {
        input_tokens,
        output_tokens,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{AspectRatio, ImageBackground, ImageFormat, ImageQuality};

    /// An 8-byte PNG header, enough for `infer` to recognise the format.
    const PNG_HEADER_B64: &str = "iVBORw0KGgo=";

    /// The body a real `gemini-3.1-flash-lite-image` 400 carried for `imageSize: 512`.
    const SIZE_REFUSAL: &str = r#"{"error":{"code":400,"message":"Image size 512 is not supported for this model","status":"INVALID_ARGUMENT"}}"#;

    fn body(options: &ImageModelOptions) -> serde_json::Value {
        GeminiImageApiModel::new("gemini-3.1-flash-image", "AIza-test")
            .body("a red cube on a white table", options)
    }

    #[test]
    fn minimal_body_asks_for_text_and_image_only() {
        assert_eq!(
            body(&ImageModelOptions::default()),
            serde_json::json!({
                "contents": [{"parts": [{"text": "a red cube on a white table"}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            })
        );
    }

    #[test]
    fn body_sends_aspect_ratio_and_image_size() {
        let options = ImageModelOptions {
            aspect_ratio: Some(AspectRatio::Ratio16x9),
            image_size: Some(ImageSize::Size512),
            ..Default::default()
        };
        assert_eq!(
            body(&options)["generationConfig"]["imageConfig"],
            serde_json::json!({"aspectRatio": "16:9", "imageSize": "512"})
        );
    }

    #[test]
    fn body_ignores_fields_gemini_does_not_read() {
        // `quality` must not turn into a resolution, nor `n` into a candidate count.
        let options = ImageModelOptions {
            n: Some(2),
            size: Some("1536x1024".to_owned()),
            quality: Some(ImageQuality::High),
            output_format: Some(ImageFormat::Jpeg),
            background: Some(ImageBackground::Transparent),
            ..Default::default()
        };
        assert_eq!(body(&options), body(&ImageModelOptions::default()));
    }

    #[test]
    fn explain_error_names_the_image_size_option() {
        let with_size = Some(ImageSize::Size512);
        let model = "gemini-3.1-flash-lite-image";

        let hint = explain_error(model, None, with_size, 400, SIZE_REFUSAL)
            .expect("an imageSize refusal is explained");
        assert!(hint.contains("`image_size: 512`"), "{hint}");
        assert!(hint.contains(model), "{hint}");
        assert!(
            hint.contains("leave it unset"),
            "the advice must hold for any model, not name a size: {hint}"
        );
        assert!(!hint.contains("1K"), "{hint}");

        // Another status, another error, or no size set: nothing to explain.
        assert!(explain_error(model, None, with_size, 429, SIZE_REFUSAL).is_none());
        assert!(
            explain_error(
                model,
                None,
                with_size,
                400,
                r#"{"error":{"message":"API key not valid"}}"#
            )
            .is_none()
        );
        assert!(explain_error(model, None, None, 400, SIZE_REFUSAL).is_none());

        // The wording another model used for the same refusal.
        let hint = explain_error(
            model,
            None,
            Some(ImageSize::Size768),
            400,
            r#"{"error":{"code":400,"message":"Unsupported image_size '768'. Supported values are: 1K, 2K, 4K, 512"}}"#,
        )
        .expect("an image_size refusal is explained");
        assert!(hint.contains("`image_size: 768`"), "{hint}");
    }

    #[test]
    fn explain_error_names_the_aspect_ratio_option() {
        const RATIO_REFUSAL: &str = r#"{"error":{"code":400,"message":"aspect_ratio must be one of '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'","status":"INVALID_ARGUMENT"}}"#;
        let model = "gemini-3.1-flash-image";

        let hint = explain_error(model, Some(AspectRatio::Auto), None, 400, RATIO_REFUSAL)
            .expect("an aspect_ratio refusal is explained");
        assert!(hint.contains("`aspect_ratio: auto`"), "{hint}");
        assert!(hint.contains(model), "{hint}");
        assert!(hint.contains("leave it unset"), "{hint}");
        assert!(!hint.contains("image_size"), "{hint}");

        assert!(explain_error(model, None, None, 400, RATIO_REFUSAL).is_none());
        assert!(explain_error(model, Some(AspectRatio::Auto), None, 429, RATIO_REFUSAL).is_none());
    }

    #[test]
    fn endpoint_puts_the_model_in_the_path() {
        assert_eq!(
            endpoint("gemini-3.1-flash-image"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent"
        );
    }

    #[test]
    fn parse_text_and_inline_data() {
        let response = to_value!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Here is your generated image:"},
                        {"inlineData": {"mimeType": "image/png", "data": PNG_HEADER_B64}}
                    ]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 1290,
                "totalTokenCount": 1302
            }
        });

        let out = parse_response(&response).unwrap();
        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("expected an embedded image, got {:?}", out.images[0]);
        };
        assert_eq!(mime_type, "image/png");
        assert_eq!(&data.as_ref()[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(out.text.as_deref(), Some("Here is your generated image:"));
        let usage = out.usage.expect("usageMetadata is parsed");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 1290);
    }

    #[test]
    fn no_image_reports_the_finish_reason() {
        let response = to_value!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "I can't create that."}]},
                "finishReason": "IMAGE_SAFETY"
            }]
        });

        let err = parse_response(&response).unwrap_err().to_string();
        assert!(err.contains("IMAGE_SAFETY"), "{err}");
        assert!(
            err.contains("I can't create that."),
            "the model's own explanation survives: {err}"
        );
    }

    #[test]
    fn no_image_reports_a_prompt_block() {
        let response = to_value!({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}});
        let err = parse_response(&response).unwrap_err().to_string();
        assert!(err.contains("PROHIBITED_CONTENT"), "{err}");
    }

    #[test]
    fn thought_parts_are_left_out() {
        // The part order a real response had when thoughts were requested: thought text,
        // draft, thought text, final.
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "Defining the visual parameters.", "thought": true},
                    {"inlineData": {"mimeType": "image/jpeg", "data": PNG_HEADER_B64}, "thought": true},
                    {"text": "Checking the draft against the prompt.", "thought": true},
                    {"text": "A red cube."},
                    {"text": "Rendered at 16:9."},
                    {"inlineData": {"mimeType": "image/webp", "data": PNG_HEADER_B64}}
                ]}
            }]
        });

        let out = parse_response(&response).unwrap();
        assert_eq!(out.images.len(), 1, "images holds the final render only");
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        // PNG bytes labelled webp: the response's own mimeType wins over sniffing.
        assert_eq!(mime_type, "image/webp");
        assert_eq!(out.text.as_deref(), Some("A red cube.\nRendered at 16:9."));

        // A draft is counted, never decoded: a broken one is not reported as undecodable.
        let broken_draft = to_value!({
            "candidates": [{"content": {"parts": [
                {"inlineData": {"mimeType": "image/jpeg", "data": "not base64!"}, "thought": true},
                {"inlineData": {"mimeType": "image/png", "data": PNG_HEADER_B64}}
            ]}}]
        });
        assert_eq!(parse_response(&broken_draft).unwrap().images.len(), 1);

        // Drafts alone are not a result.
        let drafts_only = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": PNG_HEADER_B64}, "thought": true}
                ]},
                "finishReason": "STOP"
            }]
        });
        let err = parse_response(&drafts_only).unwrap_err().to_string();
        assert!(err.contains("1 draft image(s) marked `thought`"), "{err}");
        assert!(err.contains("finishReason: STOP"), "{err}");
        assert!(!err.contains("decodable"), "{err}");
    }

    #[test]
    fn an_undecodable_inline_part_is_skipped() {
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/png"}},
                    {"inlineData": {"mimeType": "image/png", "data": "not base64!"}},
                    {"inlineData": {"mimeType": "image/png", "data": PNG_HEADER_B64}}
                ]}
            }]
        });
        let out = parse_response(&response).unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "the good part survives its broken siblings"
        );

        let response = to_value!({
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png"}}]}}]
        });
        let err = parse_response(&response).unwrap_err().to_string();
        assert!(
            err.contains("1 inlineData part(s) had no decodable data"),
            "{err}"
        );
    }

    #[test]
    fn permanent_quota_is_recognised() {
        assert!(is_permanent_quota(
            r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}"#
        ));
        assert!(
            !is_permanent_quota(
                r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"9s"}]}}"#
            ),
            "a RetryInfo marks a passing limit"
        );
        assert!(!is_permanent_quota("not json"));
    }

    #[test]
    fn usage_counts_thinking_as_output() {
        // Thinking is reported apart from the candidates, and billed as output.
        let usage = parse_usage(&to_value!({
            "promptTokenCount": 14,
            "candidatesTokenCount": 1227,
            "thoughtsTokenCount": 135,
            "totalTokenCount": 1376
        }))
        .unwrap();
        assert_eq!(usage.input_tokens, 14);
        assert_eq!(usage.output_tokens, 1227 + 135);

        // No total, or one that makes no sense: the candidates.
        for usage in [
            to_value!({"promptTokenCount": 14, "candidatesTokenCount": 1227}),
            to_value!({"promptTokenCount": 14, "candidatesTokenCount": 1227, "totalTokenCount": 3}),
        ] {
            assert_eq!(parse_usage(&usage).unwrap().output_tokens, 1227);
        }
    }

    /// Serves `respond(attempt)` for every request, counting attempts from 1.
    async fn mock(
        respond: fn(u32) -> (u16, &'static str),
    ) -> (String, std::sync::Arc<std::sync::atomic::AtomicU32>) {
        use std::sync::{
            Arc,
            atomic::{AtomicU32, Ordering},
        };

        use axum::{Router, response::Response};

        let calls = Arc::new(AtomicU32::new(0));
        let counter = calls.clone();
        // The model id and `:generateContent` are in the path, so any path is answered.
        let app = Router::new().fallback(move || {
            let attempt = counter.fetch_add(1, Ordering::SeqCst) + 1;
            async move {
                let (status, body) = respond(attempt);
                Response::builder()
                    .status(status)
                    .header("retry-after", "0")
                    .body(axum::body::Body::from(body))
                    .unwrap()
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });
        (format!("http://{addr}/"), calls)
    }

    async fn send_to(
        url: &str,
        image_size: Option<ImageSize>,
    ) -> anyhow::Result<reqwest::Response> {
        send(
            url,
            "AIza-test",
            &serde_json::json!({}),
            "gemini-3.1-flash-lite-image",
            None,
            image_size,
        )
        .await
    }

    #[tokio::test]
    async fn send_authenticates_with_the_api_key_header() {
        use axum::{Router, http::HeaderMap, http::StatusCode};

        let app = Router::new().fallback(|headers: HeaderMap| async move {
            if headers.get("x-goog-api-key").and_then(|v| v.to_str().ok()) == Some("AIza-test")
                && headers.get("authorization").is_none()
            {
                StatusCode::OK
            } else {
                StatusCode::UNAUTHORIZED
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.ok() });

        send_to(&format!("http://{addr}/"), None).await.unwrap();
    }

    #[tokio::test]
    async fn send_retries_an_overload() {
        let (url, calls) = mock(|attempt| match attempt {
            1 => (503, r#"{"error":{"status":"UNAVAILABLE"}}"#),
            _ => (200, "{}"),
        })
        .await;

        send_to(&url, None).await.unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn send_does_not_retry_a_server_error() {
        let (url, calls) = mock(|_| (500, r#"{"error":{"status":"INTERNAL"}}"#)).await;

        let err = send_to(&url, None).await.unwrap_err();
        assert!(err.to_string().contains("500"), "{err}");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn generate_refuses_a_model_id_that_would_change_the_url() {
        for model in ["models/gemini", "gemini?alt=sse", "gemini#x"] {
            let err = GeminiImageApiModel::new(model, "AIza-test")
                .generate("a red cube", &ImageModelOptions::default())
                .await
                .unwrap_err();
            assert!(err.to_string().contains("model id"), "{err}");
        }
    }

    #[tokio::test]
    async fn send_explains_a_size_refusal() {
        let (url, calls) = mock(|_| (400, SIZE_REFUSAL)).await;

        let err = send_to(&url, Some(ImageSize::Size512))
            .await
            .unwrap_err()
            .to_string();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

        assert!(err.contains("400"), "{err}");
        assert!(err.contains("`image_size: 512`"), "{err}");
        assert!(
            err.contains("Image size 512 is not supported"),
            "the API's own message is kept: {err}"
        );
    }

    #[tokio::test]
    #[ignore = "spends Gemini credit and needs GEMINI_API_KEY; run with --ignored"]
    async fn live_generate() {
        dotenvy::dotenv().ok();
        let Ok(model) = GeminiImageApiModel::from_env("gemini-3.1-flash-lite-image") else {
            eprintln!("GEMINI_API_KEY is not set; skipping");
            return;
        };
        let options = ImageModelOptions {
            aspect_ratio: Some(AspectRatio::Ratio16x9),
            ..Default::default()
        };

        let out = model
            .generate("A single red cube on a plain white background", &options)
            .await
            .unwrap_or_else(|e| panic!("Gemini image generation failed: {e:#}"));

        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("Gemini must return inline bytes, got {:?}", out.images[0]);
        };
        assert!(mime_type.starts_with("image/"), "{mime_type}");
        // The `imageConfig` reached the model: a 16:9 request does not come back square.
        let decoded = image::load_from_memory(data.as_ref()).expect("the bytes are an image");
        let (width, height) = image::GenericImageView::dimensions(&decoded);
        assert!(
            width > height,
            "16:9 must be landscape, got {width}x{height}"
        );
    }
}
