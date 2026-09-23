use anyhow::bail;
use url::Url;

use crate::{
    datatype::{Bytes, Value},
    image_model::{
        ImageModelAPISchema, ImageModelOutput, ImageModelProvider, ImageModelProviderElem,
        ImageModelRequest, ImageQuality,
    },
    message::{PartImage, TokenUsage},
    to_value,
};

impl ImageModelProvider {
    /// The Gemini API base, same as the text side: the model id and the
    /// `:generateContent` action are appended per request.
    ///
    /// Only the Nano Banana (`gemini-*-image`) models answer with an image;
    /// pointing this at a text model produces an error from the API, because
    /// the request carries `responseModalities` and `imageConfig`.
    pub fn gemini(api_key: String) -> ImageModelProviderElem {
        ImageModelProviderElem::API {
            schema: ImageModelAPISchema::Gemini,
            url: Url::parse("https://generativelanguage.googleapis.com/v1beta/models/").unwrap(),
            api_key: Some(api_key),
        }
    }
}

/// The ratios `imageConfig.aspectRatio` accepts, in the reference's own order.
/// A requested ratio is snapped to the nearest of these; the API takes no other
/// value.
const SUPPORTED_ASPECT_RATIOS: &[(&str, u32, u32)] = &[
    ("1:1", 1, 1),
    ("1:4", 1, 4),
    ("4:1", 4, 1),
    ("1:8", 1, 8),
    ("8:1", 8, 1),
    ("2:3", 2, 3),
    ("3:2", 3, 2),
    ("3:4", 3, 4),
    ("4:3", 4, 3),
    ("4:5", 4, 5),
    ("5:4", 5, 4),
    ("9:16", 9, 16),
    ("16:9", 16, 9),
    ("21:9", 21, 9),
];

/// The Gemini `imageSize` a quality level maps to, with the quality's own
/// spelling for messages. `imageSize` is the only resolution knob the API
/// exposes; its `4K` step is out of reach of the three-step quality option.
///
/// Models differ in which sizes they take, checked by calling each:
/// `gemini-3.1-flash-image` takes all three;
/// `gemini-3.1-flash-lite-image` takes `1K` only; `gemini-3-pro-image` refuses
/// `512` and honours `2K` (2048x2048); `gemini-2.5-flash-image` refuses `512`
/// and accepts `2K` but still returns 1024x1024. That is deliberately not
/// encoded as a table here — the request goes out as asked and a refusal is
/// explained by [`GeminiImageApi::explain_error`].
fn image_size_for(quality: ImageQuality) -> (&'static str, &'static str) {
    match quality {
        ImageQuality::Low => ("low", "512"),
        ImageQuality::Medium => ("medium", "1K"),
        ImageQuality::High => ("high", "2K"),
    }
}

/// Picks the supported ratio closest to `ratio`, comparing in log space so the
/// distance is relative: 2.0 is as far from 1.0 as 0.5 is, which a plain
/// subtraction would get wrong and which matters because the table is nearly
/// symmetric around square.
fn nearest_supported_aspect_ratio(ratio: f64) -> &'static str {
    let distance =
        |&(_, w, h): &(&str, u32, u32)| (ratio / (f64::from(w) / f64::from(h))).ln().abs();
    SUPPORTED_ASPECT_RATIOS
        .iter()
        .min_by(|a, b| distance(a).total_cmp(&distance(b)))
        .map(|&(name, _, _)| name)
        .expect("SUPPORTED_ASPECT_RATIOS is never empty")
}

#[derive(Clone, Debug, Default)]
pub struct GeminiImageApi;

impl super::ImageProviderApi for GeminiImageApi {
    fn marshal_request(&self, req: &ImageModelRequest<'_>) -> anyhow::Result<Value> {
        let ImageModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;

        if let Some(n) = options.n
            && n != 1
        {
            // `candidateCount` is not documented as working for image models, so
            // asking for any count other than one is refused rather than
            // silently answered with one image.
            bail!("Gemini returns a single image per request; requested n = {n}");
        }

        let url = format!("{}{}:generateContent", url, req.model);

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("x-goog-api-key".into(), api_key.into());
        }

        let mut image_config = Value::object_empty();
        if let Some(aspect) = options.aspect_ratio {
            let ratio = aspect.ratio();
            if !(ratio.is_finite() && ratio > 0.0) {
                bail!("invalid aspect ratio '{aspect}'");
            }
            image_config.as_object_mut().unwrap().insert(
                "aspectRatio".into(),
                nearest_supported_aspect_ratio(ratio).into(),
            );
        }
        if let Some(quality) = options.quality {
            let (_, image_size) = image_size_for(quality);
            image_config
                .as_object_mut()
                .unwrap()
                .insert("imageSize".into(), image_size.into());
        }

        // `output_format` and `background` have no `generateContent` equivalent
        // (`responseFormat.image` is a different request surface, and its mime
        // enum has no PNG), so both are dropped; the response's own `mimeType`
        // is what the output reports.
        let mut generation_config = to_value!({
            "responseModalities": ["TEXT", "IMAGE"],
        });
        if !image_config.as_object().unwrap().is_empty() {
            generation_config
                .as_object_mut()
                .unwrap()
                .insert("imageConfig".into(), image_config);
        }
        if options.include_drafts == Some(true) {
            // The model thinks either way; this only makes it return the
            // thought parts, draft images included.
            generation_config.as_object_mut().unwrap().insert(
                "thinkingConfig".into(),
                to_value!({"includeThoughts": true}),
            );
        }

        let body = to_value!({
            "contents": [{"parts": [{"text": req.prompt}]}],
            "generationConfig": generation_config,
        });

        Ok(to_value!({
            "url": url,
            "header": header,
            "body": body,
        }))
    }

    fn unmarshal_response(&self, val: Value) -> anyhow::Result<ImageModelOutput> {
        let parts = val
            .pointer("/candidates/0/content/parts")
            .and_then(|v| v.as_array());

        let mut images = Vec::new();
        let mut texts: Vec<&str> = Vec::new();
        let mut undecodable = 0usize;
        let mut drafts = Vec::new();
        let mut thoughts: Vec<&str> = Vec::new();
        for part in parts.into_iter().flatten() {
            // Thinking image models (Nano Banana Pro, and the flash models with
            // thinking on) mark their interim work with `thought: true`: text
            // summaries, and also draft images of the composition. Neither is
            // the answer: they go to `drafts` / `thoughts`, never `images` or
            // `text`. They only arrive when `include_drafts` asked for them.
            let is_thought = part.pointer("/thought").and_then(|v| v.as_bool()) == Some(true);
            // REST answers in camelCase; the snake_case spelling is accepted
            // too because that is what the request side uses.
            let inline = part
                .pointer("/inlineData")
                .or_else(|| part.pointer("/inline_data"));
            if let Some(inline) = inline {
                // One unusable payload does not condemn the response: a later
                // part may still carry a good image, and if none does, the
                // zero-image bail below says how many were dropped.
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
                    .map(|m| m.to_owned())
                    .or_else(|| infer::get(data.as_ref()).map(|t| t.mime_type().to_string()))
                    .unwrap_or_else(|| "image/png".to_string());
                let image = PartImage::Embedded { mime_type, data };
                if is_thought {
                    drafts.push(image);
                } else {
                    images.push(image);
                }
            } else if let Some(part_text) = part.pointer("/text").and_then(|v| v.as_str()) {
                if is_thought {
                    thoughts.push(part_text);
                } else {
                    texts.push(part_text);
                }
            }
        }
        // Separate parts are separate lines; concatenating them runs sentences
        // together.
        let text = texts.join("\n");

        if images.is_empty() {
            // Quote whatever the response said about why, so the caller does not
            // have to re-run with a debugger to find out.
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
            // Drafts are not the answer, so a response carrying only drafts is
            // still a failure.
            if !drafts.is_empty() {
                reasons.push(format!(
                    "{} draft image(s) marked `thought` and no final image",
                    drafts.len()
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
                reasons.push("no reason reported".to_string());
            }
            bail!("Gemini returned no image ({})", reasons.join("; "));
        }

        let thoughts = thoughts.join("\n");
        Ok(ImageModelOutput {
            drafts,
            thoughts: (!thoughts.is_empty()).then_some(thoughts),
            images,
            text: (!text.is_empty()).then_some(text),
            usage: parse_usage(&val),
        })
    }

    fn explain_error(
        &self,
        req: &ImageModelRequest<'_>,
        status: u16,
        body: &str,
    ) -> Option<String> {
        // The live 400 reads "Image size 512 is not supported for this model".
        // It names `imageSize`, which the caller never set.
        let quality = req.options.quality?;
        if status != 400 || !body.contains("Image size") || !body.contains("not supported") {
            return None;
        }
        let (quality_name, image_size) = image_size_for(quality);
        Some(format!(
            "`quality: {quality_name}` was sent as Gemini imageSize {image_size}, which '{}' \
             does not accept; try another quality, or leave quality unset to use the \
             model's default size",
            req.model
        ))
    }

    fn is_permanent_quota_error(&self, body: &str) -> bool {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };
        let error = &json["error"];
        // RESOURCE_EXHAUSTED covers both; a RetryInfo detail marks the transient case.
        error["status"] == "RESOURCE_EXHAUSTED"
            && !error["details"].as_array().into_iter().flatten().any(|d| {
                d["@type"]
                    .as_str()
                    .is_some_and(|t| t.ends_with("google.rpc.RetryInfo"))
            })
    }
}

/// Parses Gemini `usageMetadata` (`promptTokenCount` / `candidatesTokenCount`).
/// `promptTokenCount` includes any cached-content tokens (folded into `input_tokens`).
fn parse_usage(root: &Value) -> Option<TokenUsage> {
    let usage = root
        .pointer("/usageMetadata")
        .filter(|u| !u.is_null())?
        .as_object()?;
    Some(TokenUsage {
        input_tokens: usage
            .get("promptTokenCount")
            .and_then(|v| v.as_unsigned())
            .unwrap_or(0),
        output_tokens: usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_unsigned())
            .unwrap_or(0),
        cache_creation_input_tokens: None,
        cache_read_input_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{super::ImageProviderApi as _, *};
    use crate::image_model::{AspectRatio, ImageModelOptions};

    /// An 8-byte PNG header, enough for `infer` to recognise the format.
    const PNG_HEADER_B64: &str = "iVBORw0KGgo=";

    fn marshal(model: &str, options: &ImageModelOptions) -> anyhow::Result<Value> {
        let provider = ImageModelProvider::gemini("AIza-test".to_string());
        let req = ImageModelRequest {
            model,
            prompt: "a red cube on a white table",
            provider: &provider,
            options,
        };
        GeminiImageApi.marshal_request(&req)
    }

    #[test]
    fn marshal_minimal_request() {
        let marshaled = marshal("gemini-3.1-flash-image", &ImageModelOptions::default()).unwrap();

        assert_eq!(
            marshaled.pointer("/url").and_then(|v| v.as_str()),
            Some(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent"
            )
        );
        assert_eq!(
            marshaled
                .pointer("/header/x-goog-api-key")
                .and_then(|v| v.as_str()),
            Some("AIza-test")
        );
        assert_eq!(
            marshaled
                .pointer("/body/contents/0/parts/0/text")
                .and_then(|v| v.as_str()),
            Some("a red cube on a white table")
        );

        let modalities = marshaled
            .pointer("/body/generationConfig/responseModalities")
            .and_then(|v| v.as_array())
            .expect("responseModalities must be present");
        let modalities: Vec<&str> = modalities.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(modalities, vec!["TEXT", "IMAGE"]);

        assert!(
            marshaled
                .pointer("/body/generationConfig/imageConfig")
                .is_none(),
            "an empty imageConfig must not be sent"
        );
    }

    #[test]
    fn marshal_maps_quality_to_image_size() {
        for (quality, expected) in [
            (ImageQuality::Low, "512"),
            (ImageQuality::Medium, "1K"),
            (ImageQuality::High, "2K"),
        ] {
            let options = ImageModelOptions {
                quality: Some(quality),
                ..Default::default()
            };
            let marshaled = marshal("gemini-3.1-flash-image", &options).unwrap();
            assert_eq!(
                marshaled
                    .pointer("/body/generationConfig/imageConfig/imageSize")
                    .and_then(|v| v.as_str()),
                Some(expected),
                "{quality:?}"
            );
        }
    }

    #[test]
    fn marshal_sends_image_size_to_every_model() {
        // No per-model table: the model decides, and a refusal is explained
        // afterwards. Even a size lite rejects goes out as asked.
        let options = ImageModelOptions {
            quality: Some(ImageQuality::Low),
            ..Default::default()
        };
        let marshaled = marshal("gemini-3.1-flash-lite-image", &options).unwrap();
        assert_eq!(
            marshaled
                .pointer("/body/generationConfig/imageConfig/imageSize")
                .and_then(|v| v.as_str()),
            Some("512")
        );
    }

    #[test]
    fn explain_error_ties_an_image_size_refusal_to_quality() {
        // The body a real gemini-3.1-flash-lite-image 400 carried.
        const REFUSAL: &str = r#"{"error":{"code":400,"message":"Image size 512 is not supported for this model","status":"INVALID_ARGUMENT"}}"#;
        let provider = ImageModelProvider::gemini("AIza-test".to_string());
        let with_quality = ImageModelOptions {
            quality: Some(ImageQuality::Low),
            ..Default::default()
        };
        let req = |options| ImageModelRequest {
            model: "gemini-3.1-flash-lite-image",
            prompt: "a red cube",
            provider: &provider,
            options,
        };

        let hint = GeminiImageApi
            .explain_error(&req(&with_quality), 400, REFUSAL)
            .expect("an imageSize refusal is explained");
        assert!(hint.contains("`quality: low`"), "{hint}");
        assert!(hint.contains("imageSize 512"), "{hint}");
        assert!(hint.contains("gemini-3.1-flash-lite-image"), "{hint}");
        assert!(
            hint.contains("leave quality unset"),
            "the advice must hold for any model, not name a size: {hint}"
        );
        assert!(!hint.contains("medium"), "{hint}");

        // Nothing to explain: another status, another error, or no quality set.
        assert!(
            GeminiImageApi
                .explain_error(&req(&with_quality), 429, REFUSAL)
                .is_none()
        );
        assert!(
            GeminiImageApi
                .explain_error(
                    &req(&with_quality),
                    400,
                    r#"{"error":{"message":"API key not valid"}}"#
                )
                .is_none()
        );
        assert!(
            GeminiImageApi
                .explain_error(&req(&ImageModelOptions::default()), 400, REFUSAL)
                .is_none()
        );
    }

    #[test]
    fn marshal_snaps_aspect_ratio_to_a_supported_value() {
        let cases = [
            (AspectRatio { w: 1, h: 1 }, "1:1"),
            (AspectRatio { w: 16, h: 9 }, "16:9"),
            (AspectRatio { w: 9, h: 16 }, "9:16"),
            // 1.777… and 0.5625, arrived at from pixel dimensions.
            (AspectRatio { w: 1920, h: 1080 }, "16:9"),
            (AspectRatio { w: 1080, h: 1920 }, "9:16"),
            // Not in the table: snapped to the nearest entry.
            (AspectRatio { w: 5, h: 3 }, "16:9"),
            (AspectRatio { w: 7, h: 1 }, "8:1"),
        ];
        for (aspect, expected) in cases {
            let options = ImageModelOptions {
                aspect_ratio: Some(aspect),
                ..Default::default()
            };
            let marshaled = marshal("gemini-3.1-flash-image", &options).unwrap();
            assert_eq!(
                marshaled
                    .pointer("/body/generationConfig/imageConfig/aspectRatio")
                    .and_then(|v| v.as_str()),
                Some(expected),
                "{aspect} must snap to {expected}"
            );
        }
    }

    #[test]
    fn marshal_rejects_multiple_images() {
        let options = ImageModelOptions {
            n: Some(2),
            ..Default::default()
        };
        let err = marshal("gemini-3.1-flash-image", &options)
            .unwrap_err()
            .to_string();
        assert!(err.contains("single image"), "unexpected message: {err}");

        // Zero is not "leave it to the model" either; that is what `None` means.
        let options = ImageModelOptions {
            n: Some(0),
            ..Default::default()
        };
        let err = marshal("gemini-3.1-flash-image", &options)
            .unwrap_err()
            .to_string();
        assert!(err.contains("single image"), "unexpected message: {err}");

        let options = ImageModelOptions {
            n: Some(1),
            ..Default::default()
        };
        assert!(marshal("gemini-3.1-flash-image", &options).is_ok());
    }

    #[test]
    fn marshal_ignores_openai_only_options() {
        let options = ImageModelOptions {
            output_format: Some(crate::image_model::ImageFormat::Jpeg),
            background: Some(crate::image_model::ImageBackground::Transparent),
            ..Default::default()
        };
        let marshaled = marshal("gemini-3.1-flash-image", &options).unwrap();
        let config = marshaled
            .pointer("/body/generationConfig")
            .and_then(|v| v.as_object())
            .unwrap();
        assert_eq!(
            config.keys().collect::<Vec<_>>(),
            vec!["responseModalities"]
        );
    }

    #[test]
    fn unmarshal_text_and_inline_data() {
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

        let out = GeminiImageApi.unmarshal_response(response).unwrap();
        assert_eq!(out.images.len(), 1);
        let PartImage::Embedded { mime_type, data } = &out.images[0] else {
            panic!("expected an embedded image, got {:?}", out.images[0]);
        };
        assert_eq!(mime_type, "image/png");
        assert_eq!(&data.as_ref()[..4], &[0x89, 0x50, 0x4E, 0x47]);
        assert_eq!(out.text.as_deref(), Some("Here is your generated image:"));
        let usage = out.usage.expect("usageMetadata must be parsed");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 1290);
    }

    #[test]
    fn unmarshal_reports_the_finish_reason_when_no_image_came_back() {
        let response = to_value!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "I can't create that."}]},
                "finishReason": "IMAGE_SAFETY"
            }]
        });

        let err = GeminiImageApi
            .unmarshal_response(response)
            .unwrap_err()
            .to_string();
        assert!(err.contains("IMAGE_SAFETY"), "unexpected message: {err}");
        assert!(
            err.contains("I can't create that."),
            "the model's own explanation must survive: {err}"
        );
    }

    #[test]
    fn unmarshal_reports_a_prompt_level_block() {
        let response = to_value!({
            "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}
        });

        let err = GeminiImageApi
            .unmarshal_response(response)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("PROHIBITED_CONTENT"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn unmarshal_skips_thought_parts_in_text() {
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "thinking about cubes", "thought": true},
                    {"text": "A red cube."},
                    {"text": "Rendered at 16:9."},
                    {"inlineData": {"mimeType": "image/webp", "data": PNG_HEADER_B64}}
                ]}
            }]
        });

        let out = GeminiImageApi.unmarshal_response(response).unwrap();
        assert_eq!(
            out.text.as_deref(),
            Some("A red cube.\nRendered at 16:9."),
            "separate text parts are separate lines"
        );
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(
            mime_type, "image/webp",
            "the response's own mimeType wins over sniffing"
        );
    }

    #[test]
    fn marshal_asks_for_thoughts_only_when_drafts_are_wanted() {
        let default = marshal("gemini-3-pro-image", &ImageModelOptions::default()).unwrap();
        assert!(
            default
                .pointer("/body/generationConfig/thinkingConfig")
                .is_none(),
            "thoughts are not requested by default"
        );

        let options = ImageModelOptions {
            include_drafts: Some(true),
            ..Default::default()
        };
        let marshaled = marshal("gemini-3-pro-image", &options).unwrap();
        assert_eq!(
            marshaled
                .pointer("/body/generationConfig/thinkingConfig/includeThoughts")
                .and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn unmarshal_separates_drafts_from_the_final_image() {
        // The part order a real gemini-3-pro-image response with
        // `includeThoughts` had: thought text, draft, thought text, final.
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "Defining the visual parameters.", "thought": true},
                    {"inlineData": {"mimeType": "image/jpeg", "data": PNG_HEADER_B64}, "thought": true},
                    {"text": "Checking the draft against the prompt.", "thought": true},
                    {"inlineData": {"mimeType": "image/png", "data": PNG_HEADER_B64}}
                ]}
            }]
        });

        let out = GeminiImageApi.unmarshal_response(response).unwrap();
        assert_eq!(out.images.len(), 1, "images holds the final render only");
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(
            mime_type, "image/png",
            "the one in images is the final render"
        );
        assert_eq!(out.drafts.len(), 1, "the draft is kept apart");
        let PartImage::Embedded { mime_type, .. } = &out.drafts[0] else {
            panic!("expected an embedded draft");
        };
        assert_eq!(mime_type, "image/jpeg");
        assert_eq!(
            out.thoughts.as_deref(),
            Some("Defining the visual parameters.\nChecking the draft against the prompt.")
        );
        assert_eq!(out.text, None, "thought text is not the caption");

        // Drafts alone are not a result: the call fails and says why.
        let drafts_only = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": PNG_HEADER_B64}, "thought": true}
                ]},
                "finishReason": "STOP"
            }]
        });
        let err = GeminiImageApi
            .unmarshal_response(drafts_only)
            .unwrap_err()
            .to_string();
        assert!(err.contains("1 draft image(s) marked `thought`"), "{err}");
        assert!(err.contains("finishReason: STOP"), "{err}");
    }

    #[test]
    fn unmarshal_skips_an_undecodable_inline_part() {
        let response = to_value!({
            "candidates": [{
                "content": {"parts": [
                    {"inlineData": {"mimeType": "image/png"}},
                    {"inlineData": {"mimeType": "image/png", "data": "not base64!"}},
                    {"inlineData": {"mimeType": "image/png", "data": PNG_HEADER_B64}}
                ]}
            }]
        });

        let out = GeminiImageApi.unmarshal_response(response).unwrap();
        assert_eq!(
            out.images.len(),
            1,
            "the good part must survive its broken siblings"
        );

        // With nothing usable left, the failure says how many parts were dropped.
        let response = to_value!({
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png"}}]}}]
        });
        let err = GeminiImageApi
            .unmarshal_response(response)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("1 inlineData part(s) had no decodable data"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn constructor_serializes_to_the_documented_wire_form() {
        let json = serde_json::to_value(ImageModelProvider::gemini("k".into())).unwrap();
        assert_eq!(json["type"], "api");
        assert_eq!(json["schema"], "gemini");
        assert_eq!(
            json["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/"
        );
        assert_eq!(json["api_key"], "k");
    }

    #[test]
    fn permanent_quota_error_is_recognised() {
        assert!(GeminiImageApi.is_permanent_quota_error(
            r#"{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}"#
        ));
        assert!(
            !GeminiImageApi.is_permanent_quota_error(
                r#"{"error":{"status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"9s"}]}}"#
            ),
            "a RetryInfo detail marks a transient limit"
        );
        assert!(!GeminiImageApi.is_permanent_quota_error("not json"));
    }
}
