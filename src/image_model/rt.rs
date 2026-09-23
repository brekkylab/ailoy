use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::ImageModelProviderElem;
use crate::{
    datatype::Value,
    image_model::{ImageModelOptions, get_im_providers, r#impl::api},
    message::{PartImage, TokenUsage},
};

/// Runtime handle for one text-to-image model at one provider endpoint.
pub struct ImageModel {
    model: String,
    provider: ImageModelProviderElem,
}

/// The marshal-side view of one generate call.
///
/// `pub(crate)` rather than `pub(super)` because it appears in
/// [`ImageProviderApi::marshal_request`](crate::image_model::r#impl::api::ImageProviderApi::marshal_request),
/// whose own reach is crate-wide; a narrower type there is a `private_interfaces`
/// warning. Nothing re-exports it, so it stays out of the public API either way.
pub(crate) struct ImageModelRequest<'a> {
    pub model: &'a str,
    pub prompt: &'a str,
    pub provider: &'a ImageModelProviderElem,
    pub options: &'a ImageModelOptions,
}

/// The result of one [`ImageModel::generate`] call.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOutput {
    /// The generated images, never empty — a response that produced none is an
    /// error, not an empty vector.  Normally
    /// [`PartImage::Embedded`](crate::message::PartImage::Embedded) with the
    /// mime type the provider reported. A provider that answers with a link
    /// instead of bytes yields
    /// [`PartImage::Url`](crate::message::PartImage::Url); those URLs expire
    /// after an hour.
    pub images: Vec<PartImage>,

    /// Text the model returned alongside the images, such as a caption or the
    /// prompt as the provider rewrote it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Tokens billed for the call, when the provider reports them. `None` when
    /// the response carries no usage block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl ImageModel {
    /// Resolve `model` against the `"default"` entry of
    /// [`get_im_providers`](crate::image_model::get_im_providers).  Convenience
    /// over [`try_from_provider`](Self::try_from_provider).
    ///
    /// Returns an error if the `"default"` provider is missing or has no entry
    /// matching `model`.
    pub fn try_new(model: String) -> anyhow::Result<Self> {
        Self::try_from_provider(model, "default")
    }

    /// Resolve `model` against the [`ImageModelProvider`](super::ImageModelProvider)
    /// registered under `provider` in
    /// [`get_im_providers`](crate::image_model::get_im_providers).
    ///
    /// `model` is the spec-side name (e.g. `"openai/gpt-image-1"`) used to look
    /// up the registered pattern; the stored API-side id has any `provider/`
    /// prefix stripped (e.g. `"gpt-image-1"`) so it matches what the upstream
    /// endpoint expects.
    ///
    /// Returns an error if `provider` is not registered, or if no entry inside
    /// it matches `model` (with the usual exact-then-glob lookup).
    pub fn try_from_provider(model: String, provider: impl AsRef<str>) -> anyhow::Result<Self> {
        let provider_name = provider.as_ref();
        let registry = get_im_providers();
        let imp = registry.get(provider_name).ok_or_else(|| {
            anyhow::anyhow!("image_model_provider '{}' not registered", provider_name)
        })?;
        let provider_elem = imp
            .get(&model)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no entry for model '{}' in image_model_provider '{}'",
                    model,
                    provider_name
                )
            })?
            .clone();
        let api_model_id = imp.resolve_model_id(&model)?;
        Ok(Self {
            model: api_model_id,
            provider: provider_elem,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// Generate images from a text prompt.
    ///
    /// Contract: an `Ok` result carries at least one image.  A call that ends
    /// with no image — a safety filter, a text-only answer — is an `Err`
    /// quoting the reason the provider gave, so callers never inspect an empty
    /// vector.  Options the target model cannot honour fail before anything is
    /// sent; see [`ImageModelOptions`].
    pub async fn generate(
        &self,
        prompt: &str,
        options: &ImageModelOptions,
    ) -> anyhow::Result<ImageModelOutput> {
        let req = ImageModelRequest {
            model: &self.model,
            prompt,
            provider: &self.provider,
            options,
        };
        let ImageModelProviderElem::API { schema, .. } = &self.provider;
        let api = api::provider_api(schema);
        let (url, header_map, body) = wire_parts(&api.marshal_request(&req)?)?;

        let client = reqwest::Client::new();
        let response =
            send_with_retry(&client, &url, header_map, &body, api.as_ref(), &req).await?;
        let response_text = response.text().await?;

        let response_value: Value =
            serde_json::from_str::<serde_json::Value>(&response_text)?.into();

        api.unmarshal_response(response_value)
    }
}

/// Unpacks the `{"url", "header", "body"}` object a marshal produces into the
/// pieces reqwest needs.
fn wire_parts(marshaled: &Value) -> anyhow::Result<(String, HeaderMap, serde_json::Value)> {
    let obj = marshaled
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid Marshal"))?;

    let url = obj
        .get("url")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("No URL in marshaled request"))?
        .to_owned();

    let mut header_map = HeaderMap::new();
    if let Some(header_obj) = obj.get("header").and_then(|v| v.as_object()) {
        for (key, value) in header_obj.iter() {
            if let Some(val_str) = value.as_str() {
                header_map.insert(
                    HeaderName::from_bytes(key.as_bytes())?,
                    HeaderValue::from_str(val_str)?,
                );
            }
        }
    }

    let body = obj
        .get("body")
        .ok_or_else(|| anyhow::anyhow!("No body in marshaled request"))?;
    let body: serde_json::Value = body.clone().into();

    Ok((url, header_map, body))
}

/// POSTs the request, retrying transient 429s with backoff, and returns the
/// successful (2xx) response unconsumed. Bails on a non-2xx response or
/// exhausted retries, quoting the body — for image generation that body is
/// usually the only place the rejected option is named.
async fn send_with_retry(
    client: &reqwest::Client,
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
    provider: &(dyn api::ImageProviderApi + Send + Sync),
    req: &ImageModelRequest<'_>,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
    const MAX_WAIT_SECS: u64 = 10;
    for attempt in 0..=MAX_RETRIES {
        let response = client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        if status.as_u16() == 429 && attempt < MAX_RETRIES {
            let wait_secs = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(1u64 << attempt)
                .min(MAX_WAIT_SECS);
            let text = response.text().await?;
            // Permanent quota/credit exhaustion never recovers; don't retry.
            if provider.is_permanent_quota_error(&text) {
                log::warn!("Quota exhausted (429), not retrying: {text}");
                anyhow::bail!("API request failed with status {status}: {text}");
            }
            log::warn!(
                "Rate limited (429). Retrying after {}s (attempt {}/{}): {}",
                wait_secs,
                attempt + 1,
                MAX_RETRIES,
                text
            );
            tokio::time::sleep(std::time::Duration::from_secs(wait_secs)).await;
            continue;
        }
        let text = response.text().await.unwrap_or_default();
        // Unlike lang_model's copy, a provider may add a sentence tying the
        // failure to the caller's options; the body is kept either way.
        match provider.explain_error(req, status.as_u16(), &text) {
            Some(hint) => anyhow::bail!("API request failed with status {status} ({hint}): {text}"),
            None => anyhow::bail!("API request failed with status {status}: {text}"),
        }
    }
    unreachable!("retry loop returns or bails on every path")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        image_model::{
            AspectRatio, ImageModelAPISchema, ImageModelProvider, ImageQuality, ImageSize,
            get_im_providers_mut,
        },
        to_value,
    };

    /// Register a one-off [`ImageModelProvider`] under a unique key in the
    /// global registry and build an [`ImageModel`] from it via
    /// [`ImageModel::try_from_provider`].  Test fixtures only.
    ///
    /// Tests that spend money on a real endpoint are named `live_*`, so
    /// `cargo test -- --skip live` runs everything else.
    fn build_test_model(
        provider_name: &str,
        model: &str,
        elem: ImageModelProviderElem,
    ) -> ImageModel {
        let mut imp = ImageModelProvider::new();
        imp.insert(model.into(), elem);
        get_im_providers_mut().insert(provider_name.into(), imp);
        ImageModel::try_from_provider(model.to_string(), provider_name).unwrap()
    }

    #[test]
    fn wire_parts_unpacks_url_header_and_body() {
        let marshaled = to_value!({
            "url": "https://example.com/v1/images/generations",
            "header": {"content-type": "application/json", "Authorization": "Bearer k"},
            "body": {"model": "m", "prompt": "p"}
        });

        let (url, headers, body) = wire_parts(&marshaled).unwrap();
        assert_eq!(url, "https://example.com/v1/images/generations");
        assert_eq!(headers.get("authorization").unwrap(), "Bearer k");
        assert_eq!(body["prompt"], serde_json::json!("p"));
    }

    #[test]
    fn wire_parts_rejects_an_incomplete_marshal() {
        assert!(wire_parts(&to_value!({"header": {}, "body": {}})).is_err());
        assert!(wire_parts(&to_value!({"url": "https://example.com"})).is_err());
        assert!(wire_parts(&to_value!("not an object")).is_err());
    }

    #[test]
    fn try_from_provider_reports_an_unknown_registry_or_model() {
        let err = ImageModel::try_from_provider("openai/gpt-image-1".into(), "no-such-registry")
            .err()
            .expect("an unregistered provider must not resolve")
            .to_string();
        assert!(err.contains("not registered"), "unexpected message: {err}");

        build_test_model(
            "test_image_lookup",
            "openai/gpt-image-1",
            ImageModelProvider::openai("sk-test".into()),
        );
        let err = ImageModel::try_from_provider("openai/nope".into(), "test_image_lookup")
            .err()
            .expect("an unregistered model must not resolve")
            .to_string();
        assert!(err.contains("no entry for model"), "unexpected: {err}");
    }

    fn model_at(provider_name: &str, url: &str) -> ImageModel {
        build_test_model(
            provider_name,
            "test-image-model",
            ImageModelProviderElem::API {
                schema: ImageModelAPISchema::OpenAI,
                url: url.parse().unwrap(),
                api_key: None,
            },
        )
    }

    /// Verifies that the runtime retries on 429 and succeeds when the server
    /// recovers. The mock returns 429 for the first two requests, then 200.
    #[tokio::test]
    async fn generate_retries_on_429() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, body::Body, response::Response, routing::post};

        let call_count = Arc::new(Mutex::new(0u32));
        let count = call_count.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let count = count.clone();
                async move {
                    let mut n = count.lock().unwrap();
                    *n += 1;
                    let current = *n;
                    drop(n);

                    if current <= 2 {
                        // Return 429 with retry-after: 0 so the sleep is instant.
                        Response::builder()
                            .status(429)
                            .header("retry-after", "0")
                            .body(Body::from(r#"{"error":"rate limited"}"#))
                            .unwrap()
                    } else {
                        Response::builder()
                            .status(200)
                            .header("content-type", "application/json")
                            .body(Body::from(
                                r#"{"created":1,"data":[{"b64_json":"iVBORw0KGgo="}]}"#,
                            ))
                            .unwrap()
                    }
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let out = model_at("test_image_retry_mock", &format!("http://{addr}/"))
            .generate("a red cube", &ImageModelOptions::default())
            .await
            .unwrap();

        assert_eq!(
            *call_count.lock().unwrap(),
            3,
            "should have made 3 total attempts (2x 429 retried + 1 success)"
        );
        assert_eq!(out.images.len(), 1);
    }

    /// A non-retryable failure keeps the provider's body and gains the
    /// provider's explanation in front of it. The mock answers with the body a
    /// real `gemini-3.1-flash-lite-image` 400 carried for `imageSize: 512`.
    #[tokio::test]
    async fn generate_explains_a_refusal_in_terms_of_the_option_set() {
        use axum::{Router, body::Body, response::Response};

        const REFUSAL: &str = r#"{"error":{"code":400,"message":"Image size 512 is not supported for this model","status":"INVALID_ARGUMENT"}}"#;
        // Gemini puts the model id and `:generateContent` in the path, so any
        // path is answered.
        let app = Router::new().fallback(|| async {
            Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(Body::from(REFUSAL))
                .unwrap()
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let model = build_test_model(
            "test_image_explain_mock",
            "gemini-3.1-flash-lite-image",
            ImageModelProviderElem::API {
                schema: ImageModelAPISchema::Gemini,
                url: format!("http://{addr}/").parse().unwrap(),
                api_key: None,
            },
        );
        let options = ImageModelOptions {
            image_size: Some(ImageSize::Size512),
            ..Default::default()
        };
        let err = model
            .generate("a red cube", &options)
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("400"), "{err}");
        assert!(
            err.contains("`image_size: 512`"),
            "the explanation names the option: {err}"
        );
        assert!(
            err.contains("Image size 512 is not supported"),
            "the provider's own body is kept: {err}"
        );
    }

    /// Verifies a permanent quota 429 (`insufficient_quota`) is not retried.
    #[tokio::test]
    async fn generate_does_not_retry_on_permanent_429() {
        use std::sync::{Arc, Mutex};

        use axum::{Router, body::Body, response::Response, routing::post};

        let call_count = Arc::new(Mutex::new(0u32));
        let count = call_count.clone();
        let app = Router::new().route(
            "/",
            post(move || {
                let count = count.clone();
                async move {
                    *count.lock().unwrap() += 1;
                    Response::builder()
                        .status(429)
                        .body(Body::from(
                            r#"{"error":{"type":"insufficient_quota","code":"insufficient_quota","message":"You exceeded your current quota"}}"#,
                        ))
                        .unwrap()
                }
            }),
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let result = model_at("test_image_permanent_429_mock", &format!("http://{addr}/"))
            .generate("a red cube", &ImageModelOptions::default())
            .await;

        assert!(result.is_err(), "permanent quota 429 must fail");
        assert_eq!(
            *call_count.lock().unwrap(),
            1,
            "permanent quota 429 must not be retried (1 attempt only)"
        );
    }

    /// Live: OpenAI `gpt-image-1-mini` returns one PNG plus a usage block.
    /// Costs about a cent per run.
    #[tokio::test]
    async fn live_generate_openai_image() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set in .env");

        let model = build_test_model(
            "test_image_openai",
            "gpt-image-1-mini",
            ImageModelProvider::openai(api_key),
        );
        let options = ImageModelOptions {
            quality: Some(ImageQuality::Low),
            ..Default::default()
        };

        let out = model
            .generate("A single red cube on a plain white background", &options)
            .await
            .unwrap_or_else(|e| panic!("OpenAI image generation failed: {e:#}"));

        assert_eq!(out.images.len(), 1, "expected exactly one image");
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("gpt-image must return bytes, got {:?}", out.images[0]);
        };
        println!(
            "openai: mime={mime_type} bytes={} text={:?} usage={:?}",
            data.len(),
            out.text,
            out.usage
        );
        assert_eq!(mime_type, "image/png");
        assert_eq!(
            &data.as_ref()[..4],
            &[0x89, 0x50, 0x4E, 0x47],
            "expected PNG magic bytes"
        );
        assert!(
            out.usage.is_some(),
            "gpt-image reports token usage; got none"
        );
    }

    /// Live: Gemini `gemini-3.1-flash-lite-image` over `:generateContent` with
    /// `responseModalities` and an `imageConfig`. Costs about a cent per run.
    #[tokio::test]
    async fn live_generate_gemini_image() {
        dotenvy::dotenv().ok();
        let api_key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY must be set in .env");

        let model = build_test_model(
            "test_image_gemini",
            "gemini-3.1-flash-lite-image",
            ImageModelProvider::gemini(api_key),
        );
        let options = ImageModelOptions {
            aspect_ratio: Some(AspectRatio::Ratio16x9),
            ..Default::default()
        };

        let out = model
            .generate("A single red cube on a plain white background", &options)
            .await
            .unwrap_or_else(|e| panic!("Gemini image generation failed: {e:#}"));

        assert_eq!(out.images.len(), 1, "expected exactly one image");
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("Gemini must return inline bytes, got {:?}", out.images[0]);
        };
        let sniffed = infer::get(data.as_ref()).map(|t| t.mime_type());
        println!(
            "gemini: mime={mime_type} sniffed={sniffed:?} bytes={} text={:?} usage={:?}",
            data.len(),
            out.text,
            out.usage
        );
        assert!(
            mime_type.starts_with("image/"),
            "unexpected mime type {mime_type}"
        );
        assert_eq!(
            sniffed.map(|m| m.starts_with("image/")),
            Some(true),
            "the bytes must be recognisable as an image, got {sniffed:?}"
        );
        // Confirms the `imageConfig` actually reached the model: a 16:9 request
        // must not come back square. Skipped if the format is one the `image`
        // crate is not built for here.
        if let Ok(decoded) = image::load_from_memory(data.as_ref()) {
            let (width, height) = image::GenericImageView::dimensions(&decoded);
            println!("gemini: dims={width}x{height}");
            assert!(
                width > height,
                "a 16:9 aspect_ratio must yield a landscape image, got {width}x{height}"
            );
        }
    }
}
