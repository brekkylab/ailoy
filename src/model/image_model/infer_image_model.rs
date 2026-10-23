use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    message::{PartImage, TokenUsage},
};

/// Options for one image generation call, in one section per API.
///
/// Each section holds one API's parameters under that API's own names, and only models
/// that call that API read it; the others ignore it. An unset section, like an unset field,
/// leaves that API's defaults in place. Fill several sections to switch between models of
/// different APIs with the same options.
///
/// Values are sent as written: the API decides which it takes, and a refusal comes back as
/// its error. A combination an API never accepts fails before anything is sent. Unknown
/// sections and keys are ignored when deserializing, not rejected.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOptions {
    /// The OpenAI Images API's parameters, read by models with an `openai/…` desc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai: Option<OpenAiImageOptions>,

    /// The Gemini API's parameters, read by models with a `google/…` desc: the org is
    /// Google, the API is Gemini.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemini: Option<GeminiImageOptions>,

    /// The OpenRouter Images API's parameters, read by models with an `openrouter/…` desc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openrouter: Option<OpenRouterImageOptions>,
}

impl ImageModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The parameters of OpenAI's Images API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct OpenAiImageOptions {
    /// How many images to generate, 1 to 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The pixel size, as `"<width>x<height>"` (e.g. `"1536x1024"`) or `"auto"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// How much care goes into the render, e.g. `"low"` or `"high"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,

    /// The encoding of the returned image, e.g. `"png"` or `"webp"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,

    /// The background behind the subject, e.g. `"transparent"` or `"opaque"`. A
    /// transparent one needs an alpha-capable format, so it is refused together with a
    /// `"jpeg"` output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,

    /// Extra request fields, merged into this API's request body after the fields above;
    /// for parameters these options do not cover. Written as the API's own JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// The image parameters of the Gemini API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct GeminiImageOptions {
    /// The width-to-height ratio, e.g. `"16:9"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    /// The output resolution as a tier, e.g. `"1K"` or `"2K"`: the tier sets roughly the
    /// pixel count, [`aspect_ratio`](Self::aspect_ratio) its shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,

    /// How much the model reasons before drawing, e.g. `"low"` or `"high"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,

    /// Extra request fields, merged into this API's request body after the fields above;
    /// for parameters these options do not cover. Written as the API's own JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// The parameters of OpenRouter's Images API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct OpenRouterImageOptions {
    /// How many images to generate, 1 to 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The pixel size, as `"<width>x<height>"` (e.g. `"1024x1024"`) or a size tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// The output resolution as a tier, e.g. `"1K"` or `"2K"`: the tier sets roughly the
    /// pixel count, [`aspect_ratio`](Self::aspect_ratio) its shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,

    /// The width-to-height ratio, e.g. `"16:9"` or `"auto"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    /// How much care goes into the render, e.g. `"low"` or `"high"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<String>,

    /// The encoding of the returned image, e.g. `"png"` or `"webp"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,

    /// The background behind the subject, e.g. `"transparent"` or `"opaque"`. A
    /// transparent one needs an alpha-capable format, so it is refused together with a
    /// `"jpeg"` output format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<String>,

    /// Extra request fields, merged into this API's request body after the fields above;
    /// for parameters these options do not cover. Written as the API's own JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// The result of one [`InferImageModel::generate`] call.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ImageOutput {
    /// The generated images, never empty. Normally [`PartImage::Embedded`] with the mime
    /// type the provider reported; a provider that answers with a link yields
    /// [`PartImage::Url`], which may be short-lived.
    pub images: Vec<PartImage>,

    /// Text the model returned alongside the images, such as a caption or the prompt as
    /// the provider rewrote it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Tokens billed for the call, when the provider reports them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

/// A model that generates images from a text prompt.
pub trait InferImageModel: Send + Sync {
    /// Generate images from `prompt`.
    ///
    /// An `Ok` result carries at least one image. A call that ends with none, such as one
    /// stopped by a safety filter, is an `Err` quoting the reason the provider gave.
    fn generate(
        &self,
        prompt: &str,
        options: &ImageModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_omit_unset_sections_and_fields() {
        assert_eq!(
            serde_json::to_string(&ImageModelOptions::new()).unwrap(),
            "{}"
        );
        let options = ImageModelOptions {
            gemini: Some(GeminiImageOptions::default()),
            ..Default::default()
        };
        assert_eq!(serde_json::to_string(&options).unwrap(), r#"{"gemini":{}}"#);
    }

    #[test]
    fn options_serde_round_trip() {
        let json = serde_json::json!({
            "openai": {
                "n": 2,
                "size": "1536x1024",
                "quality": "high",
                "output_format": "webp",
                "background": "transparent",
                "extra": {"moderation": "low"}
            },
            "gemini": {
                "aspect_ratio": "16:9",
                "image_size": "2K",
                "thinking_level": "minimal",
                "extra": {"generationConfig": {"candidateCount": 1}}
            },
            "openrouter": {
                "n": 1,
                "size": "1024x1024",
                "resolution": "1.5K",
                "aspect_ratio": "9:19.5",
                "quality": "auto",
                "output_format": "svg",
                "background": "auto",
                "extra": {"seed": 7}
            }
        });
        let options: ImageModelOptions = serde_json::from_value(json.clone()).unwrap();
        let gemini = options.gemini.as_ref().unwrap();
        assert_eq!(gemini.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(gemini.thinking_level.as_deref(), Some("minimal"));
        let openrouter = options.openrouter.as_ref().unwrap();
        assert_eq!(openrouter.resolution.as_deref(), Some("1.5K"));
        assert_eq!(
            serde_json::to_value(&openrouter.extra).unwrap(),
            serde_json::json!({"seed": 7})
        );
        assert_eq!(serde_json::to_value(&options).unwrap(), json);
    }

    #[test]
    fn unknown_sections_and_keys_are_ignored() {
        let options: ImageModelOptions = serde_json::from_value(serde_json::json!({
            "nobody": {"n": 1},
            "openai": {"n": 2, "seed": 7}
        }))
        .unwrap();
        assert_eq!(options.openai.unwrap().n, Some(2));
    }
}
