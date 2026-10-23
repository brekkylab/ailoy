use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::message::{PartImage, TokenUsage};

/// Options for one image generation call, in one section per API.
///
/// Each section holds one API's parameters under that API's own names, and only models
/// that call that API read it; the others ignore it. An unset section, like an unset field,
/// leaves that API's defaults in place. Fill several sections to switch between models of
/// different APIs with the same options.
///
/// A combination an API never accepts fails before anything is sent; anything else goes
/// out as asked, and a refusal comes back as the API's error. Unknown sections and keys
/// are ignored when deserializing, not rejected.
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

    /// The pixel size, as `"<width>x<height>"` (e.g. `"1536x1024"`) or `"auto"`, sent as
    /// written. A string, since the sizes a model accepts differ by model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// How much care goes into the render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenAiImageQuality>,

    /// The encoding of the returned image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenAiImageFormat>,

    /// The background behind the subject. `Transparent` needs an alpha-capable format, so
    /// it is refused together with `output_format: jpeg`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenAiImageBackground>,
}

/// The image parameters of the Gemini API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct GeminiImageOptions {
    /// The width-to-height ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<GeminiAspectRatio>,

    /// The output resolution as a tier: the tier sets roughly the pixel count,
    /// [`aspect_ratio`](Self::aspect_ratio) its shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<GeminiImageSize>,

    /// How much the model reasons before drawing; not every model takes every level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<GeminiThinkingLevel>,
}

/// The parameters of OpenRouter's Images API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct OpenRouterImageOptions {
    /// How many images to generate, 1 to 10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The pixel size, as `"<width>x<height>"` or a size tier, sent as written. A string,
    /// since the sizes a model accepts differ by model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// The output resolution as a tier: the tier sets roughly the pixel count,
    /// [`aspect_ratio`](Self::aspect_ratio) its shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<OpenRouterResolution>,

    /// The width-to-height ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<OpenRouterAspectRatio>,

    /// How much care goes into the render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<OpenRouterImageQuality>,

    /// The encoding of the returned image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OpenRouterImageFormat>,

    /// The background behind the subject. `Transparent` needs an alpha-capable format, so
    /// it is refused together with `output_format: jpeg`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<OpenRouterImageBackground>,
}

// Each enum's wire string is its serde name; `strum` gives the same string as
// `&'static str` (`let wire: &str = x.into()`), so the two attributes sit on one variant.
// The enums are per API, even where their values coincide, since each API changes its own.

/// How much care goes into the render. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenAiImageQuality {
    /// The model picks.
    Auto,
    Low,
    Medium,
    High,
    // Named explicitly: `snake_case` would spell it `x_high`.
    #[serde(rename = "xhigh")]
    #[strum(serialize = "xhigh")]
    XHigh,
    Max,
}

/// The encoding of the returned image. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenAiImageFormat {
    Png,
    Jpeg,
    Webp,
}

impl OpenAiImageFormat {
    /// The IANA media type for this encoding.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }
}

/// The background behind the subject. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenAiImageBackground {
    /// The model picks.
    Auto,
    Transparent,
    Opaque,
}

/// A width-to-height ratio. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum GeminiAspectRatio {
    #[serde(rename = "1:1")]
    #[strum(serialize = "1:1")]
    Ratio1x1,
    #[serde(rename = "1:4")]
    #[strum(serialize = "1:4")]
    Ratio1x4,
    #[serde(rename = "4:1")]
    #[strum(serialize = "4:1")]
    Ratio4x1,
    #[serde(rename = "1:8")]
    #[strum(serialize = "1:8")]
    Ratio1x8,
    #[serde(rename = "8:1")]
    #[strum(serialize = "8:1")]
    Ratio8x1,
    #[serde(rename = "2:3")]
    #[strum(serialize = "2:3")]
    Ratio2x3,
    #[serde(rename = "3:2")]
    #[strum(serialize = "3:2")]
    Ratio3x2,
    #[serde(rename = "3:4")]
    #[strum(serialize = "3:4")]
    Ratio3x4,
    #[serde(rename = "4:3")]
    #[strum(serialize = "4:3")]
    Ratio4x3,
    #[serde(rename = "4:5")]
    #[strum(serialize = "4:5")]
    Ratio4x5,
    #[serde(rename = "5:4")]
    #[strum(serialize = "5:4")]
    Ratio5x4,
    #[serde(rename = "9:16")]
    #[strum(serialize = "9:16")]
    Ratio9x16,
    #[serde(rename = "16:9")]
    #[strum(serialize = "16:9")]
    Ratio16x9,
    #[serde(rename = "21:9")]
    #[strum(serialize = "21:9")]
    Ratio21x9,
}

/// How much a model reasons before drawing. Not every model takes every level.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum GeminiThinkingLevel {
    Minimal,
    Low,
    Medium,
    High,
}

/// An output resolution tier. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum GeminiImageSize {
    #[serde(rename = "512")]
    #[strum(serialize = "512")]
    Size512,
    #[serde(rename = "1K")]
    #[strum(serialize = "1K")]
    Size1K,
    #[serde(rename = "2K")]
    #[strum(serialize = "2K")]
    Size2K,
    #[serde(rename = "4K")]
    #[strum(serialize = "4K")]
    Size4K,
}

/// An output resolution tier. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum OpenRouterResolution {
    #[serde(rename = "512")]
    #[strum(serialize = "512")]
    Res512,
    #[serde(rename = "768")]
    #[strum(serialize = "768")]
    Res768,
    #[serde(rename = "1K")]
    #[strum(serialize = "1K")]
    Res1K,
    // `p` stands for the decimal point.
    #[serde(rename = "1.5K")]
    #[strum(serialize = "1.5K")]
    Res1p5K,
    #[serde(rename = "2K")]
    #[strum(serialize = "2K")]
    Res2K,
    #[serde(rename = "4K")]
    #[strum(serialize = "4K")]
    Res4K,
}

/// A width-to-height ratio. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum OpenRouterAspectRatio {
    /// The model picks.
    #[serde(rename = "auto")]
    #[strum(serialize = "auto")]
    Auto,
    #[serde(rename = "1:1")]
    #[strum(serialize = "1:1")]
    Ratio1x1,
    #[serde(rename = "1:4")]
    #[strum(serialize = "1:4")]
    Ratio1x4,
    #[serde(rename = "4:1")]
    #[strum(serialize = "4:1")]
    Ratio4x1,
    #[serde(rename = "1:8")]
    #[strum(serialize = "1:8")]
    Ratio1x8,
    #[serde(rename = "8:1")]
    #[strum(serialize = "8:1")]
    Ratio8x1,
    #[serde(rename = "1:2")]
    #[strum(serialize = "1:2")]
    Ratio1x2,
    #[serde(rename = "2:1")]
    #[strum(serialize = "2:1")]
    Ratio2x1,
    #[serde(rename = "2:3")]
    #[strum(serialize = "2:3")]
    Ratio2x3,
    #[serde(rename = "3:2")]
    #[strum(serialize = "3:2")]
    Ratio3x2,
    #[serde(rename = "3:4")]
    #[strum(serialize = "3:4")]
    Ratio3x4,
    #[serde(rename = "4:3")]
    #[strum(serialize = "4:3")]
    Ratio4x3,
    #[serde(rename = "4:5")]
    #[strum(serialize = "4:5")]
    Ratio4x5,
    #[serde(rename = "5:4")]
    #[strum(serialize = "5:4")]
    Ratio5x4,
    #[serde(rename = "5:7")]
    #[strum(serialize = "5:7")]
    Ratio5x7,
    #[serde(rename = "7:5")]
    #[strum(serialize = "7:5")]
    Ratio7x5,
    #[serde(rename = "9:16")]
    #[strum(serialize = "9:16")]
    Ratio9x16,
    #[serde(rename = "16:9")]
    #[strum(serialize = "16:9")]
    Ratio16x9,
    // `p` stands for the decimal point.
    #[serde(rename = "9:19.5")]
    #[strum(serialize = "9:19.5")]
    Ratio9x19p5,
    #[serde(rename = "19.5:9")]
    #[strum(serialize = "19.5:9")]
    Ratio19p5x9,
    #[serde(rename = "9:20")]
    #[strum(serialize = "9:20")]
    Ratio9x20,
    #[serde(rename = "20:9")]
    #[strum(serialize = "20:9")]
    Ratio20x9,
    #[serde(rename = "9:21")]
    #[strum(serialize = "9:21")]
    Ratio9x21,
    #[serde(rename = "21:9")]
    #[strum(serialize = "21:9")]
    Ratio21x9,
}

/// How much care goes into the render. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenRouterImageQuality {
    /// The model picks.
    Auto,
    Low,
    Medium,
    High,
    // Named explicitly: `snake_case` would spell it `x_high`.
    #[serde(rename = "xhigh")]
    #[strum(serialize = "xhigh")]
    XHigh,
    Max,
}

/// The encoding of the returned image. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenRouterImageFormat {
    Png,
    Jpeg,
    Webp,
    Svg,
}

impl OpenRouterImageFormat {
    /// The IANA media type for this encoding.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
            Self::Svg => "image/svg+xml",
        }
    }
}

/// The background behind the subject. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum OpenRouterImageBackground {
    /// The model picks.
    Auto,
    Transparent,
    Opaque,
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
                "quality": "xhigh",
                "output_format": "webp",
                "background": "transparent"
            },
            "gemini": {"aspect_ratio": "16:9", "image_size": "2K", "thinking_level": "minimal"},
            "openrouter": {
                "n": 1,
                "size": "1024x1024",
                "resolution": "1.5K",
                "aspect_ratio": "9:19.5",
                "quality": "auto",
                "output_format": "svg",
                "background": "auto"
            }
        });
        let options: ImageModelOptions = serde_json::from_value(json.clone()).unwrap();
        let openai = options.openai.as_ref().unwrap();
        assert_eq!(openai.quality, Some(OpenAiImageQuality::XHigh));
        assert_eq!(openai.output_format, Some(OpenAiImageFormat::Webp));
        let gemini = options.gemini.as_ref().unwrap();
        assert_eq!(gemini.aspect_ratio, Some(GeminiAspectRatio::Ratio16x9));
        assert_eq!(gemini.image_size, Some(GeminiImageSize::Size2K));
        assert_eq!(gemini.thinking_level, Some(GeminiThinkingLevel::Minimal));
        let openrouter = options.openrouter.as_ref().unwrap();
        assert_eq!(openrouter.resolution, Some(OpenRouterResolution::Res1p5K));
        assert_eq!(
            openrouter.aspect_ratio,
            Some(OpenRouterAspectRatio::Ratio9x19p5)
        );
        assert_eq!(serde_json::to_value(&options).unwrap(), json);
    }

    #[test]
    fn a_section_takes_only_its_own_values() {
        let gemini = |json| serde_json::from_value::<GeminiImageOptions>(json);
        assert!(gemini(serde_json::json!({"aspect_ratio": "auto"})).is_err());
        assert!(gemini(serde_json::json!({"image_size": "768"})).is_err());
        assert!(gemini(serde_json::json!({"image_size": "1k"})).is_err());
        let openai = |json| serde_json::from_value::<OpenAiImageOptions>(json);
        assert!(openai(serde_json::json!({"output_format": "svg"})).is_err());
    }

    #[test]
    fn the_openrouter_section_takes_its_wider_values() {
        for (ratio, expected) in [
            ("9:19.5", OpenRouterAspectRatio::Ratio9x19p5),
            ("19.5:9", OpenRouterAspectRatio::Ratio19p5x9),
            ("auto", OpenRouterAspectRatio::Auto),
        ] {
            let options: OpenRouterImageOptions =
                serde_json::from_value(serde_json::json!({"aspect_ratio": ratio})).unwrap();
            assert_eq!(options.aspect_ratio, Some(expected));
        }
        for (resolution, expected) in [
            ("1.5K", OpenRouterResolution::Res1p5K),
            ("768", OpenRouterResolution::Res768),
        ] {
            let options: OpenRouterImageOptions =
                serde_json::from_value(serde_json::json!({"resolution": resolution})).unwrap();
            assert_eq!(options.resolution, Some(expected));
        }
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

    /// The JSON a caller writes and the string sent on the wire must be the same.
    fn assert_wire_matches_serde<T>(values: &[T])
    where
        T: Copy + Serialize + Into<&'static str> + std::fmt::Debug,
    {
        for &value in values {
            let wire: &'static str = value.into();
            assert_eq!(
                serde_json::to_value(value).unwrap(),
                serde_json::json!(wire),
                "{value:?}"
            );
        }
    }

    #[test]
    fn wire_strings_match_serde_names() {
        assert_wire_matches_serde(&[
            GeminiThinkingLevel::Minimal,
            GeminiThinkingLevel::Low,
            GeminiThinkingLevel::Medium,
            GeminiThinkingLevel::High,
        ]);
        let level: &str = GeminiThinkingLevel::Minimal.into();
        assert_eq!(level, "minimal");
        assert_wire_matches_serde(&[
            GeminiAspectRatio::Ratio1x1,
            GeminiAspectRatio::Ratio1x4,
            GeminiAspectRatio::Ratio4x1,
            GeminiAspectRatio::Ratio1x8,
            GeminiAspectRatio::Ratio8x1,
            GeminiAspectRatio::Ratio2x3,
            GeminiAspectRatio::Ratio3x2,
            GeminiAspectRatio::Ratio3x4,
            GeminiAspectRatio::Ratio4x3,
            GeminiAspectRatio::Ratio4x5,
            GeminiAspectRatio::Ratio5x4,
            GeminiAspectRatio::Ratio9x16,
            GeminiAspectRatio::Ratio16x9,
            GeminiAspectRatio::Ratio21x9,
        ]);
        assert_wire_matches_serde(&[
            OpenRouterAspectRatio::Auto,
            OpenRouterAspectRatio::Ratio1x1,
            OpenRouterAspectRatio::Ratio1x4,
            OpenRouterAspectRatio::Ratio4x1,
            OpenRouterAspectRatio::Ratio1x8,
            OpenRouterAspectRatio::Ratio8x1,
            OpenRouterAspectRatio::Ratio1x2,
            OpenRouterAspectRatio::Ratio2x1,
            OpenRouterAspectRatio::Ratio2x3,
            OpenRouterAspectRatio::Ratio3x2,
            OpenRouterAspectRatio::Ratio3x4,
            OpenRouterAspectRatio::Ratio4x3,
            OpenRouterAspectRatio::Ratio4x5,
            OpenRouterAspectRatio::Ratio5x4,
            OpenRouterAspectRatio::Ratio5x7,
            OpenRouterAspectRatio::Ratio7x5,
            OpenRouterAspectRatio::Ratio9x16,
            OpenRouterAspectRatio::Ratio16x9,
            OpenRouterAspectRatio::Ratio9x19p5,
            OpenRouterAspectRatio::Ratio19p5x9,
            OpenRouterAspectRatio::Ratio9x20,
            OpenRouterAspectRatio::Ratio20x9,
            OpenRouterAspectRatio::Ratio9x21,
            OpenRouterAspectRatio::Ratio21x9,
        ]);
        assert_wire_matches_serde(&[
            GeminiImageSize::Size512,
            GeminiImageSize::Size1K,
            GeminiImageSize::Size2K,
            GeminiImageSize::Size4K,
        ]);
        assert_wire_matches_serde(&[
            OpenRouterResolution::Res512,
            OpenRouterResolution::Res768,
            OpenRouterResolution::Res1K,
            OpenRouterResolution::Res1p5K,
            OpenRouterResolution::Res2K,
            OpenRouterResolution::Res4K,
        ]);
        for qualities in [
            [
                OpenAiImageQuality::Auto,
                OpenAiImageQuality::Low,
                OpenAiImageQuality::Medium,
                OpenAiImageQuality::High,
                OpenAiImageQuality::XHigh,
                OpenAiImageQuality::Max,
            ]
            .map(<&str>::from),
            [
                OpenRouterImageQuality::Auto,
                OpenRouterImageQuality::Low,
                OpenRouterImageQuality::Medium,
                OpenRouterImageQuality::High,
                OpenRouterImageQuality::XHigh,
                OpenRouterImageQuality::Max,
            ]
            .map(<&str>::from),
        ] {
            assert_eq!(qualities, ["auto", "low", "medium", "high", "xhigh", "max"]);
        }
        assert_wire_matches_serde(&[
            OpenAiImageQuality::Auto,
            OpenAiImageQuality::Low,
            OpenAiImageQuality::Medium,
            OpenAiImageQuality::High,
            OpenAiImageQuality::XHigh,
            OpenAiImageQuality::Max,
        ]);
        assert_wire_matches_serde(&[
            OpenRouterImageQuality::Auto,
            OpenRouterImageQuality::Low,
            OpenRouterImageQuality::Medium,
            OpenRouterImageQuality::High,
            OpenRouterImageQuality::XHigh,
            OpenRouterImageQuality::Max,
        ]);
        assert_wire_matches_serde(&[
            OpenAiImageFormat::Png,
            OpenAiImageFormat::Jpeg,
            OpenAiImageFormat::Webp,
        ]);
        assert_wire_matches_serde(&[
            OpenRouterImageFormat::Png,
            OpenRouterImageFormat::Jpeg,
            OpenRouterImageFormat::Webp,
            OpenRouterImageFormat::Svg,
        ]);
        assert_wire_matches_serde(&[
            OpenAiImageBackground::Auto,
            OpenAiImageBackground::Transparent,
            OpenAiImageBackground::Opaque,
        ]);
        assert_wire_matches_serde(&[
            OpenRouterImageBackground::Auto,
            OpenRouterImageBackground::Transparent,
            OpenRouterImageBackground::Opaque,
        ]);

        let ratio: &str = OpenRouterAspectRatio::Ratio19p5x9.into();
        assert_eq!(ratio, "19.5:9");
        let size: &str = GeminiImageSize::Size512.into();
        assert_eq!(size, "512");
    }

    #[test]
    fn image_format_mime_types() {
        assert_eq!(OpenAiImageFormat::Png.mime_type(), "image/png");
        assert_eq!(OpenAiImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(OpenAiImageFormat::Webp.mime_type(), "image/webp");
        assert_eq!(OpenRouterImageFormat::Webp.mime_type(), "image/webp");
        assert_eq!(OpenRouterImageFormat::Svg.mime_type(), "image/svg+xml");
    }
}
