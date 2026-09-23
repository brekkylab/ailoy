use anyhow::bail;
use url::Url;

use crate::{
    datatype::{Bytes, Value},
    image_model::{
        ImageBackground, ImageFormat, ImageModelAPISchema, ImageModelOutput, ImageModelProvider,
        ImageModelProviderElem, ImageModelRequest, ImageQuality,
    },
    message::{PartImage, TokenUsage},
    to_value,
};

impl ImageModelProvider {
    /// The OpenAI Images API endpoint, which serves the `gpt-image-*` family.
    /// Distinct from the text-side `openai` constructor: image generation is
    /// its own endpoint, not the Responses API.
    pub fn openai(api_key: String) -> ImageModelProviderElem {
        ImageModelProviderElem::API {
            schema: ImageModelAPISchema::OpenAI,
            url: Url::parse("https://api.openai.com/v1/images/generations").unwrap(),
            api_key: Some(api_key),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct OpenAIImageApi;

impl super::ImageProviderApi for OpenAIImageApi {
    fn marshal_request(&self, req: &ImageModelRequest<'_>) -> anyhow::Result<Value> {
        let ImageModelProviderElem::API { url, api_key, .. } = req.provider;
        let options = req.options;

        let url = url.to_string();

        let mut header = to_value!({
            "content-type": "application/json",
        });
        if let Some(api_key) = api_key.as_ref() {
            header
                .as_object_mut()
                .unwrap()
                .insert("Authorization".into(), format!("Bearer {}", api_key).into());
        }

        // Every model this endpoint still serves belongs to the gpt-image
        // family, so there is no per-family branching: DALL·E 2/3 were removed
        // from the API. `response_format` is never sent either — the
        // endpoint answers it with `unknown_parameter` whatever the model.
        let mut body = to_value!({
            "model": req.model,
            "prompt": req.prompt,
        });
        {
            let body = body.as_object_mut().unwrap();

            if let Some(n) = options.n {
                if !(1..=10).contains(&n) {
                    bail!("OpenAI accepts n between 1 and 10; requested n = {n}");
                }
                body.insert("n".into(), u64::from(n).into());
            }

            // The API takes concrete pixel sizes, not ratios, so the requested
            // ratio only selects between square / landscape / portrait. The
            // arbitrary `WxH` sizes gpt-image-2 and 2.5 accept are not wired up.
            if let Some(aspect) = options.aspect_ratio {
                let ratio = aspect.ratio();
                if !(ratio.is_finite() && ratio > 0.0) {
                    bail!("invalid aspect ratio '{aspect}'");
                }
                let size = if ratio > 1.0 {
                    "1536x1024"
                } else if ratio < 1.0 {
                    "1024x1536"
                } else {
                    "1024x1024"
                };
                body.insert("size".into(), size.into());
            }

            if let Some(quality) = options.quality {
                let wire = match quality {
                    ImageQuality::Low => "low",
                    ImageQuality::Medium => "medium",
                    ImageQuality::High => "high",
                };
                body.insert("quality".into(), wire.into());
            }

            if options.background == Some(ImageBackground::Transparent)
                && options.output_format == Some(ImageFormat::Jpeg)
            {
                bail!(
                    "a transparent background needs an alpha-capable format; \
                     `output_format: jpeg` cannot carry one (use png or webp)"
                );
            }
            if let Some(format) = options.output_format {
                body.insert("output_format".into(), format.as_wire_str().into());
            }
            if let Some(background) = options.background {
                let wire = match background {
                    ImageBackground::Transparent => "transparent",
                    ImageBackground::Opaque => "opaque",
                };
                body.insert("background".into(), wire.into());
            }
        }

        Ok(to_value!({
            "url": url,
            "header": header,
            "body": body,
        }))
    }

    fn unmarshal_response(&self, val: Value) -> anyhow::Result<ImageModelOutput> {
        let data = val
            .pointer("/data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing data[] in OpenAI image response"))?;
        if data.is_empty() {
            bail!("OpenAI returned no images (empty `data`)");
        }

        // `gpt-image-*` echoes the format it actually produced at the top level;
        // that is the authoritative answer. Without it, sniff the bytes — the
        // request options are not visible here, and sniffing is a better source
        // than what was asked for anyway. PNG is the documented default.
        let echoed_mime = val
            .pointer("/output_format")
            .and_then(|v| v.as_str())
            .map(|format| format!("image/{format}"));

        let mut images = Vec::with_capacity(data.len());
        for entry in data {
            if let Some(b64) = entry.pointer("/b64_json").and_then(|v| v.as_str()) {
                let data = Bytes::from_base64(b64)?;
                let mime_type = echoed_mime
                    .clone()
                    .or_else(|| infer::get(data.as_ref()).map(|t| t.mime_type().to_string()))
                    .unwrap_or_else(|| "image/png".to_string());
                images.push(PartImage::Embedded { mime_type, data });
            } else if let Some(url) = entry.pointer("/url").and_then(|v| v.as_str()) {
                // A provider that returns a link instead of bytes; the URL
                // expires in an hour, which is the caller's problem to notice.
                images.push(PartImage::Url {
                    url: url.to_owned(),
                });
            } else {
                bail!("OpenAI image entry carries neither `b64_json` nor `url`");
            }
        }

        // When the response carries a rewritten prompt, surface it as text.
        let text = val
            .pointer("/data/0/revised_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_owned());

        Ok(ImageModelOutput {
            drafts: Vec::new(),
            thoughts: None,
            images,
            text,
            usage: parse_usage(&val),
        })
    }

    fn is_permanent_quota_error(&self, body: &str) -> bool {
        let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
            return false;
        };
        let error = &json["error"];
        error["type"] == "insufficient_quota" || error["code"] == "insufficient_quota"
    }
}

/// Parses the `usage` block the endpoint returns, `Option` because a response
/// that omits it is not an error. The `*_tokens_details` breakdowns have
/// nowhere to go in [`TokenUsage`] and are dropped.
fn parse_usage(root: &Value) -> Option<TokenUsage> {
    let usage = root
        .pointer("/usage")
        .filter(|u| !u.is_null())?
        .as_object()?;
    Some(TokenUsage {
        input_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_unsigned())
            .unwrap_or(0),
        output_tokens: usage
            .get("output_tokens")
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
        let provider = ImageModelProvider::openai("sk-test".to_string());
        let req = ImageModelRequest {
            model,
            prompt: "a red cube on a white table",
            provider: &provider,
            options,
        };
        OpenAIImageApi.marshal_request(&req)
    }

    fn body_of(marshaled: &Value) -> &Value {
        marshaled.pointer("/body").expect("body must exist")
    }

    #[test]
    fn marshal_minimal_request_sends_only_model_and_prompt() {
        let marshaled = marshal("gpt-image-1", &ImageModelOptions::default()).unwrap();

        assert_eq!(
            marshaled.pointer("/url").and_then(|v| v.as_str()),
            Some("https://api.openai.com/v1/images/generations")
        );
        assert_eq!(
            marshaled
                .pointer("/header/Authorization")
                .and_then(|v| v.as_str()),
            Some("Bearer sk-test")
        );
        assert_eq!(
            marshaled
                .pointer("/header/content-type")
                .and_then(|v| v.as_str()),
            Some("application/json")
        );

        let body = body_of(&marshaled).as_object().unwrap();
        assert_eq!(
            body.keys().collect::<Vec<_>>(),
            vec!["model", "prompt"],
            "unset options must not appear on the wire"
        );
        assert_eq!(
            body.get("model").and_then(|v| v.as_str()),
            Some("gpt-image-1")
        );
        assert_eq!(
            body.get("prompt").and_then(|v| v.as_str()),
            Some("a red cube on a white table")
        );
    }

    #[test]
    fn marshal_maps_every_option_for_gpt_image() {
        let options = ImageModelOptions {
            n: Some(3),
            aspect_ratio: Some(AspectRatio { w: 16, h: 9 }),
            quality: Some(ImageQuality::High),
            output_format: Some(ImageFormat::Webp),
            background: Some(ImageBackground::Transparent),
            // Gemini's fields; neither may leak onto the OpenAI wire.
            image_size: Some("2K".to_string()),
            include_drafts: Some(true),
        };
        let marshaled = marshal("gpt-image-1", &options).unwrap();
        let body = body_of(&marshaled);
        let mut keys: Vec<&str> = body
            .as_object()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "background",
                "model",
                "n",
                "output_format",
                "prompt",
                "quality",
                "size"
            ],
            "only OpenAI's own fields reach the wire"
        );

        assert_eq!(body.pointer("/n").and_then(|v| v.as_integer()), Some(3));
        assert_eq!(
            body.pointer("/size").and_then(|v| v.as_str()),
            Some("1536x1024")
        );
        assert_eq!(
            body.pointer("/quality").and_then(|v| v.as_str()),
            Some("high")
        );
        assert_eq!(
            body.pointer("/output_format").and_then(|v| v.as_str()),
            Some("webp")
        );
        assert_eq!(
            body.pointer("/background").and_then(|v| v.as_str()),
            Some("transparent")
        );
        assert!(
            body.pointer("/response_format").is_none(),
            "gpt-image rejects response_format"
        );
    }

    /// The endpoint rejects `response_format` with `unknown_parameter` whatever
    /// the model, so no request may ever carry it.
    #[test]
    fn marshal_never_sends_response_format() {
        for model in ["gpt-image-1-mini", "gpt-image-2.5-sunburst"] {
            let marshaled = marshal(model, &ImageModelOptions::default()).unwrap();
            let body = body_of(&marshaled).as_object().unwrap();
            assert!(
                !body.contains_key("response_format"),
                "{model} must not send response_format"
            );
            assert_eq!(
                body.keys().collect::<Vec<_>>(),
                vec!["model", "prompt"],
                "{model} must send nothing beyond the request itself"
            );
        }
    }

    #[test]
    fn marshal_rejects_an_out_of_range_n() {
        for n in [0, 11] {
            let options = ImageModelOptions {
                n: Some(n),
                ..Default::default()
            };
            let err = marshal("gpt-image-1", &options)
                .err()
                .unwrap_or_else(|| panic!("n = {n} must be refused"))
                .to_string();
            assert!(err.contains("between 1 and 10"), "unexpected: {err}");
        }

        for n in [1, 10] {
            let options = ImageModelOptions {
                n: Some(n),
                ..Default::default()
            };
            let marshaled = marshal("gpt-image-1", &options)
                .unwrap_or_else(|e| panic!("n = {n} must be accepted: {e}"));
            assert_eq!(
                body_of(&marshaled)
                    .pointer("/n")
                    .and_then(|v| v.as_integer()),
                Some(i64::from(n))
            );
        }
    }

    #[test]
    fn marshal_rejects_transparent_background_on_jpeg() {
        let options = ImageModelOptions {
            output_format: Some(ImageFormat::Jpeg),
            background: Some(ImageBackground::Transparent),
            ..Default::default()
        };
        let err = marshal("gpt-image-1", &options).unwrap_err().to_string();
        assert!(
            err.contains("transparent background"),
            "unexpected message: {err}"
        );

        // Opaque JPEG is a perfectly good request.
        let options = ImageModelOptions {
            background: Some(ImageBackground::Opaque),
            ..options
        };
        assert!(marshal("gpt-image-1", &options).is_ok());
    }

    #[test]
    fn marshal_maps_aspect_ratio_onto_the_three_sizes() {
        let cases = [
            (AspectRatio { w: 1, h: 1 }, "1024x1024"),
            (AspectRatio { w: 16, h: 9 }, "1536x1024"),
            (AspectRatio { w: 9, h: 16 }, "1024x1536"),
        ];
        for (aspect, expected) in cases {
            let options = ImageModelOptions {
                aspect_ratio: Some(aspect),
                ..Default::default()
            };
            assert_eq!(
                body_of(&marshal("gpt-image-1", &options).unwrap())
                    .pointer("/size")
                    .and_then(|v| v.as_str()),
                Some(expected),
                "size for {aspect}"
            );
        }
    }

    #[test]
    fn unmarshal_gpt_image_response() {
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

        let out = OpenAIImageApi.unmarshal_response(response).unwrap();
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
    fn unmarshal_sniffs_mime_when_the_response_does_not_echo_the_format() {
        let response = to_value!({
            "created": 1713833628,
            "data": [{"b64_json": PNG_HEADER_B64}]
        });

        let out = OpenAIImageApi.unmarshal_response(response).unwrap();
        let PartImage::Embedded { mime_type, .. } = &out.images[0] else {
            panic!("expected an embedded image");
        };
        assert_eq!(mime_type, "image/png");
        assert!(
            out.usage.is_none(),
            "a response without a `usage` block reports none"
        );
    }

    #[test]
    fn unmarshal_url_only_response() {
        let response = to_value!({
            "created": 1713833628,
            "data": [{
                "url": "https://images.example.com/private/img.png",
                "revised_prompt": "A vivid red cube resting on a white table"
            }]
        });

        let out = OpenAIImageApi.unmarshal_response(response).unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(
            out.images[0],
            PartImage::Url {
                url: "https://images.example.com/private/img.png".to_string()
            }
        );
        assert_eq!(
            out.text.as_deref(),
            Some("A vivid red cube resting on a white table")
        );
    }

    #[test]
    fn unmarshal_empty_data_is_an_error() {
        let err = OpenAIImageApi
            .unmarshal_response(to_value!({"created": 1, "data": []}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no images"), "unexpected message: {err}");

        assert!(
            OpenAIImageApi
                .unmarshal_response(to_value!({"created": 1}))
                .is_err(),
            "a response without `data` is an error too"
        );
    }

    #[test]
    fn constructor_serializes_to_the_documented_wire_form() {
        let json = serde_json::to_value(ImageModelProvider::openai("k".into())).unwrap();
        assert_eq!(json["type"], "api");
        assert_eq!(json["schema"], "openai");
        assert_eq!(json["url"], "https://api.openai.com/v1/images/generations");
        assert_eq!(json["api_key"], "k");
    }

    #[test]
    fn permanent_quota_error_is_recognised() {
        assert!(OpenAIImageApi.is_permanent_quota_error(
            r#"{"error":{"message":"quota","type":"insufficient_quota"}}"#
        ));
        assert!(
            OpenAIImageApi.is_permanent_quota_error(r#"{"error":{"code":"insufficient_quota"}}"#)
        );
        assert!(
            !OpenAIImageApi.is_permanent_quota_error(r#"{"error":{"type":"rate_limit_exceeded"}}"#)
        );
        assert!(!OpenAIImageApi.is_permanent_quota_error("not json"));
    }
}
