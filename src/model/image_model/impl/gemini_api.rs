//! An image model that calls the Gemini API (`generativelanguage.googleapis.com`)
//! `:generateContent` directly over HTTP, asking for the `IMAGE` response modality.

use anyhow::{Context as _, bail};
use futures::future::BoxFuture;

use crate::{
    datatype::{Bytes, Value},
    message::{PartImage, TokenUsage},
    model::{GeminiImageOptions, ImageOptions, ImageOutput, InferImageModel},
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
    fn body(
        &self,
        prompt: &str,
        options: &GeminiImageOptions,
    ) -> anyhow::Result<serde_json::Value> {
        // Sent as asked, in `imageConfig`'s camelCase: a model that does not take a value
        // says so, and `explain_error` ties that to the option.
        let mut image_config = Value::object_empty();
        let image_obj = image_config.as_object_mut().unwrap();
        if let Some(ratio) = &options.aspect_ratio {
            image_obj.insert("aspectRatio".into(), ratio.as_str().into());
        }
        if let Some(size) = &options.image_size {
            image_obj.insert("imageSize".into(), size.as_str().into());
        }

        let mut config = to_value!({"responseModalities": ["TEXT", "IMAGE"]});
        let config_obj = config.as_object_mut().unwrap();
        if !image_obj.is_empty() {
            config_obj.insert("imageConfig".into(), image_config);
        }
        // Thoughts are not asked for (`includeThoughts`); their tokens still count in usage.
        if let Some(level) = &options.thinking_level {
            config_obj.insert(
                "thinkingConfig".into(),
                to_value!({"thinkingLevel": level.as_str()}),
            );
        }

        let mut body = to_value!({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": config,
        })
        .into();
        // `extra` may not replace the prompt or the response modalities, which include `IMAGE`.
        super::merge_extra(
            &mut body,
            options.extra.as_ref(),
            &["contents", "generationConfig.responseModalities"],
        )?;
        Ok(body)
    }
}

impl InferImageModel for GeminiImageApiModel {
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>> {
        // The id goes in the URL path, where these would change what is called.
        if self.model.contains(['/', '?', '#']) {
            let e = anyhow::anyhow!("a Gemini model id has no '/', '?' or '#': {}", self.model);
            return Box::pin(async move { Err(e) });
        }
        let request = super::read_options::<GeminiImageOptions>(options, &self.model)
            .and_then(|options| self.body(prompt, &options));
        let url = endpoint(&self.model);
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        Box::pin(async move {
            let body = request?;
            let response = send(&url, &api_key, &body, &model).await?;
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
/// paid image job may already have run. A refusal of an option in `body` is explained in
/// front of the API's own message.
async fn send(
    url: &str,
    api_key: &str,
    body: &serde_json::Value,
    model: &str,
) -> anyhow::Result<reqwest::Response> {
    const MAX_RETRIES: u32 = 3;
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
            let wait_secs = super::retry_wait_secs(response.headers(), attempt);
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
        match explain_error(model, &sent_options(body), status.as_u16(), &text) {
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

/// The options `body` sends, read back from it, so a value `extra` set is the one named.
fn sent_options(body: &serde_json::Value) -> GeminiImageOptions {
    let sent = |path: &str| {
        body.pointer(path)
            .map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned))
    };
    GeminiImageOptions {
        aspect_ratio: sent("/generationConfig/imageConfig/aspectRatio"),
        image_size: sent("/generationConfig/imageConfig/imageSize"),
        thinking_level: sent("/generationConfig/thinkingConfig/thinkingLevel"),
        extra: None,
    }
}

/// Names the option behind a refusal, which the API reports by its wire field. Runs only
/// once the model has said no, so no table of what each model takes is kept.
fn explain_error(
    model: &str,
    options: &GeminiImageOptions,
    status: u16,
    body: &str,
) -> Option<String> {
    if status != 400 {
        return None;
    }
    let mut hints = Vec::new();
    if let Some(ratio) = &options.aspect_ratio
        && body.contains("aspect_ratio")
    {
        hints.push(format!(
            "`aspect_ratio: {}` is not accepted by '{model}'; try another ratio, or leave it \
             unset to let the model choose",
            ratio,
        ));
    }
    if let Some(size) = &options.image_size
        && (body.contains("image_size")
            || (body.contains("Image size") && body.contains("not supported")))
    {
        hints.push(format!(
            "`image_size: {}` is not accepted by '{model}'; try another size, or leave it \
             unset to use the model's default size",
            size,
        ));
    }
    if let Some(level) = &options.thinking_level
        && (body.contains("thinking_level") || body.contains("Thinking level"))
    {
        hints.push(format!(
            "`thinking_level: {}` is not accepted by '{model}'; try another level, or \
             leave it unset for the model's default",
            level,
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
                .filter(|data| !data.is_empty())
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
    // Without any count, nothing was reported.
    if [
        "promptTokenCount",
        "candidatesTokenCount",
        "totalTokenCount",
    ]
    .iter()
    .all(|key| count(key).is_none())
    {
        return None;
    }
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
    use crate::model::{OpenAIImageOptions, OpenRouterImageOptions};

    /// An 8-byte PNG header, enough for `infer` to recognise the format.
    const PNG_HEADER_B64: &str = "iVBORw0KGgo=";

    /// The body a real `gemini-3.1-flash-lite-image` 400 carried for `imageSize: 512`.
    const SIZE_REFUSAL: &str = r#"{"error":{"code":400,"message":"Image size 512 is not supported for this model","status":"INVALID_ARGUMENT"}}"#;

    fn body(options: &GeminiImageOptions) -> serde_json::Value {
        GeminiImageApiModel::new("gemini-3.1-flash-image", "AIza-test")
            .body("a red cube on a white table", options)
            .unwrap()
    }

    fn set(aspect_ratio: Option<&str>, image_size: Option<&str>) -> GeminiImageOptions {
        GeminiImageOptions {
            aspect_ratio: aspect_ratio.map(str::to_owned),
            image_size: image_size.map(str::to_owned),
            ..Default::default()
        }
    }

    #[test]
    fn body_sends_options_under_their_wire_names() {
        assert_eq!(
            body(&GeminiImageOptions::default()),
            serde_json::json!({
                "contents": [{"parts": [{"text": "a red cube on a white table"}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            }),
            "text and image only, and no thoughts asked for"
        );
        let all = GeminiImageOptions {
            thinking_level: Some("high".to_owned()),
            ..set(Some("16:9"), Some("512"))
        };
        assert_eq!(
            body(&all)["generationConfig"],
            serde_json::json!({
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": "16:9", "imageSize": "512"},
                "thinkingConfig": {"thinkingLevel": "high"},
            })
        );
        let level_alone = GeminiImageOptions {
            thinking_level: Some("minimal".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            body(&level_alone)["generationConfig"],
            serde_json::json!({
                "responseModalities": ["TEXT", "IMAGE"],
                "thinkingConfig": {"thinkingLevel": "minimal"},
            })
        );
    }

    #[tokio::test]
    async fn generate_refuses_bad_options_before_sending() {
        let model = GeminiImageApiModel::new("gemini-3.1-flash-image", "AIza-test");
        // An `n` is not a candidate count, nor a `quality` a size: both are refused before
        // anything is sent, which the fake key would fail otherwise.
        let openai = OpenAIImageOptions {
            n: Some(2),
            ..Default::default()
        };
        let openrouter = OpenRouterImageOptions {
            quality: Some("high".to_owned()),
            ..Default::default()
        };
        for (options, field) in [
            (&openai as &dyn ImageOptions, "n"),
            (&openrouter, "quality"),
        ] {
            let err = model
                .generate("a red cube", Some(options))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains(&format!("unknown field `{field}`")), "{err}");
        }

        // An `aspect_ratio` built for OpenRouter is Gemini's as well.
        let options = OpenRouterImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        };
        let read: GeminiImageOptions = super::super::read_options(Some(&options), "m").unwrap();
        assert_eq!(
            body(&read)["generationConfig"]["imageConfig"],
            serde_json::json!({"aspectRatio": "16:9"})
        );
    }

    #[test]
    fn explain_error_names_the_refused_option() {
        // Bodies real 400s carried, each refusing one value.
        const RATIO_REFUSAL: &str = r#"{"error":{"code":400,"message":"aspect_ratio must be one of '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'","status":"INVALID_ARGUMENT"}}"#;
        const OTHER_SIZE_REFUSAL: &str = r#"{"error":{"code":400,"message":"Unsupported image_size '512'. Supported values are: 1K, 2K, 4K"}}"#;
        const LEVEL_REFUSAL: &str = r#"{"error":{"code":400,"message":"Thinking level MEDIUM is not supported for this model. Please retry with other thinking level.","status":"INVALID_ARGUMENT"}}"#;
        let model = "gemini-3.1-flash-lite-image";
        let level = GeminiImageOptions {
            thinking_level: Some("medium".to_owned()),
            ..Default::default()
        };

        for (options, refusal, named) in [
            (set(Some("1:8"), None), RATIO_REFUSAL, "`aspect_ratio: 1:8`"),
            (set(None, Some("512")), SIZE_REFUSAL, "`image_size: 512`"),
            (
                set(None, Some("512")),
                OTHER_SIZE_REFUSAL,
                "`image_size: 512`",
            ),
            (level, LEVEL_REFUSAL, "`thinking_level: medium`"),
        ] {
            let hint = explain_error(model, &options, 400, refusal)
                .unwrap_or_else(|| panic!("{named} is explained"));
            assert!(hint.contains(named), "{hint}");
            assert!(hint.contains(model), "{hint}");
            // The advice holds for any model, so it names no value to use instead.
            assert!(hint.contains("leave it unset"), "{hint}");
            assert!(!hint.contains("1K"), "{hint}");

            // Another status, or no value set: nothing to explain.
            assert!(explain_error(model, &options, 429, refusal).is_none());
            assert!(explain_error(model, &set(None, None), 400, refusal).is_none());
        }
        // Another error, with a size set.
        let unrelated = r#"{"error":{"message":"API key not valid"}}"#;
        assert!(explain_error(model, &set(Some("1:8"), Some("512")), 400, unrelated).is_none());
    }

    #[test]
    fn extra_merges_into_the_request_body() {
        let options = GeminiImageOptions {
            extra: Some(to_value!({
                "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}],
                "generationConfig": {"imageConfig": {"personGeneration": "ALLOW_ALL"}},
            })),
            ..set(Some("16:9"), Some("2K"))
        };
        let sent = body(&options);
        assert_eq!(
            sent["safetySettings"][0]["threshold"], "BLOCK_NONE",
            "a new field is added"
        );
        assert_eq!(
            sent["generationConfig"]["imageConfig"],
            serde_json::json!({"aspectRatio": "16:9", "imageSize": "2K", "personGeneration": "ALLOW_ALL"}),
            "a nested object keeps its typed fields"
        );
        assert_eq!(
            sent["generationConfig"]["responseModalities"],
            serde_json::json!(["TEXT", "IMAGE"])
        );

        let options = GeminiImageOptions {
            extra: Some(to_value!({"generationConfig": {"imageConfig": {"aspectRatio": "1:1"}}})),
            ..set(Some("16:9"), None)
        };
        assert_eq!(
            body(&options)["generationConfig"]["imageConfig"]["aspectRatio"],
            "1:1",
            "extra wins on conflict"
        );
    }

    #[test]
    fn extra_must_be_an_object_and_leave_reserved_keys_alone() {
        let model = GeminiImageApiModel::new("gemini-3.1-flash-image", "AIza-test");
        let err_for = |extra| {
            let options = GeminiImageOptions {
                extra: Some(extra),
                ..Default::default()
            };
            model.body("a red cube", &options).unwrap_err().to_string()
        };
        assert!(
            err_for(to_value!("x")).contains("`extra` must be a JSON object"),
            "a non-object is refused"
        );
        assert_eq!(
            err_for(to_value!({"contents": []})),
            "`extra` may not set `contents`"
        );
        assert_eq!(
            err_for(to_value!({"generationConfig": {"responseModalities": ["IMAGE"]}})),
            "`extra` may not set `generationConfig.responseModalities`"
        );
        assert_eq!(
            err_for(serde_json::json!({"generationConfig": null}).into()),
            "`extra` may not replace `generationConfig`, which holds \
             `generationConfig.responseModalities`"
        );
        // Beside the reserved key, `generationConfig` is open.
        let options = GeminiImageOptions {
            extra: Some(to_value!({"generationConfig": {"candidateCount": 1}})),
            ..Default::default()
        };
        assert_eq!(body(&options)["generationConfig"]["candidateCount"], 1);
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
    fn no_image_reports_why() {
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

        // Empty data is no image either.
        let response = to_value!({
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": ""}}]}}]
        });
        let err = parse_response(&response).unwrap_err().to_string();
        assert!(err.contains("1 inlineData part(s)"), "{err}");
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

        // No count at all is no usage, not zero tokens.
        assert!(parse_usage(&to_value!({"trafficType": "ON_DEMAND"})).is_none());
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

    async fn send_to(url: &str, options: &GeminiImageOptions) -> anyhow::Result<reqwest::Response> {
        send(
            url,
            "AIza-test",
            &body(options),
            "gemini-3.1-flash-lite-image",
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

        send_to(&format!("http://{addr}/"), &GeminiImageOptions::default())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn send_retries_only_what_may_pass() {
        type Respond = fn(u32) -> (u16, &'static str);
        const UNAVAILABLE: &str = r#"{"error":{"status":"UNAVAILABLE"}}"#;
        // How the server answers, the calls it then gets, and the error they end in, if any.
        let cases: [(Respond, u32, Option<&str>); 5] = [
            (
                |attempt| match attempt {
                    1 => (503, UNAVAILABLE),
                    _ => (200, "{}"),
                },
                2,
                None,
            ),
            // A RetryInfo marks a limit that passes.
            (
                |attempt| match attempt {
                    1 => (
                        429,
                        r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"0s"}]}}"#,
                    ),
                    _ => (200, "{}"),
                },
                2,
                None,
            ),
            (|_| (503, UNAVAILABLE), 4, Some("503")),
            // The image may already have been made and billed.
            (
                |_| (500, r#"{"error":{"status":"INTERNAL"}}"#),
                1,
                Some("500"),
            ),
            (
                |_| {
                    (
                        429,
                        r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}"#,
                    )
                },
                1,
                Some("RESOURCE_EXHAUSTED"),
            ),
        ];
        for (i, (respond, expected_calls, expected_error)) in cases.into_iter().enumerate() {
            let (url, calls) = mock(respond).await;
            let result = send_to(&url, &GeminiImageOptions::default()).await;
            let calls = calls.load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(calls, expected_calls, "case {i}");
            match expected_error {
                None => assert!(result.is_ok(), "case {i}: {:?}", result.err()),
                Some(text) => {
                    let err = result.unwrap_err().to_string();
                    assert!(err.contains(text), "case {i}: {err}");
                }
            }
        }
    }

    #[tokio::test]
    async fn generate_refuses_a_model_id_that_would_change_the_url() {
        assert_eq!(
            endpoint("gemini-3.1-flash-image"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent"
        );
        for model in ["models/gemini", "gemini?alt=sse", "gemini#x"] {
            let err = GeminiImageApiModel::new(model, "AIza-test")
                .generate("a red cube", None)
                .await
                .unwrap_err();
            assert!(err.to_string().contains("model id"), "{err}");
        }
    }

    #[tokio::test]
    async fn send_explains_a_size_refusal() {
        let (url, calls) = mock(|_| (400, SIZE_REFUSAL)).await;

        let err = send_to(&url, &set(None, Some("512")))
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

        // A size set through `extra` is the one sent, so it is the one named.
        let options = GeminiImageOptions {
            image_size: Some("1K".to_owned()),
            extra: Some(to_value!({"generationConfig": {"imageConfig": {"imageSize": "512"}}})),
            ..Default::default()
        };
        let err = send_to(&url, &options).await.unwrap_err().to_string();
        assert!(err.contains("`image_size: 512`"), "{err}");
    }

    #[tokio::test]
    #[ignore = "spends Gemini credit and needs GEMINI_API_KEY; run with --ignored"]
    async fn live_generate() {
        dotenvy::dotenv().ok();
        let Ok(model) = GeminiImageApiModel::from_env("gemini-3.1-flash-lite-image") else {
            eprintln!("GEMINI_API_KEY is not set; skipping");
            return;
        };
        let options = GeminiImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        };

        let out = model
            .generate(
                "A single red cube on a plain white background",
                Some(&options),
            )
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
        let usage = out.usage.expect("Gemini reports usage");
        assert!(usage.output_tokens > 0, "{usage:?}");
    }
}
