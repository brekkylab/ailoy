use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::message::{PartImage, TokenUsage};

/// Options for one image generation call.
///
/// Every field is optional; `None` leaves the provider's own default in place. Each
/// provider sends the fields its API takes, under that API's own names, and skips the
/// rest; no field is translated into another concept for an API that lacks it. A
/// combination an API never accepts fails before anything is sent; anything else goes out
/// as asked, and a refusal comes back as the API's error.
///
/// Flat like [`LangModelOptions`](crate::model::LangModelOptions).
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOptions {
    /// How many images to generate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// The pixel size, as `"<width>x<height>"` (e.g. `"1536x1024"`) or a keyword such as
    /// `"auto"`, sent as written.
    ///
    /// A string, since the sizes a model accepts differ by model; a refused size comes
    /// back as the API's error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// How much care goes into the render.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,

    /// The encoding of the returned bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageFormat>,

    /// The background behind the subject. `Transparent` needs an alpha-capable format, so
    /// it is refused together with `output_format: jpeg`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageBackground>,

    /// The width-to-height ratio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<AspectRatio>,

    /// The output resolution as a tier. The tier sets roughly the pixel count,
    /// [`aspect_ratio`](Self::aspect_ratio) its shape.
    ///
    /// Models differ in which tiers they take; a refused one comes back as an error that
    /// names this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_size: Option<ImageSize>,
}

impl ImageModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

// Each enum's wire string is its serde name; `strum` gives the same string as
// `&'static str` (`let wire: &str = x.into()`), so the two attributes sit on one variant.

/// An output aspect ratio. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum AspectRatio {
    /// The model picks the ratio.
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

/// An output resolution. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum ImageSize {
    #[serde(rename = "512")]
    #[strum(serialize = "512")]
    Size512,
    #[serde(rename = "768")]
    #[strum(serialize = "768")]
    Size768,
    #[serde(rename = "1K")]
    #[strum(serialize = "1K")]
    Size1K,
    // `p` stands for the decimal point.
    #[serde(rename = "1.5K")]
    #[strum(serialize = "1.5K")]
    Size1p5K,
    #[serde(rename = "2K")]
    #[strum(serialize = "2K")]
    Size2K,
    #[serde(rename = "4K")]
    #[strum(serialize = "4K")]
    Size4K,
}

/// A render quality. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageQuality {
    /// The model picks the quality.
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

/// An output encoding. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Svg,
}

impl ImageFormat {
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

/// A background. Not every model takes every value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageBackground {
    /// The model picks the background.
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
    fn options_omit_unset_fields() {
        let json = serde_json::to_string(&ImageModelOptions::new()).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn options_serde_round_trip() {
        let options = ImageModelOptions {
            n: Some(2),
            size: Some("1536x1024".to_string()),
            quality: Some(ImageQuality::High),
            image_size: Some(ImageSize::Size2K),
            output_format: Some(ImageFormat::Webp),
            background: Some(ImageBackground::Transparent),
            aspect_ratio: Some(AspectRatio::Ratio16x9),
        };
        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "n": 2,
                "size": "1536x1024",
                "quality": "high",
                "aspect_ratio": "16:9",
                "image_size": "2K",
                "output_format": "webp",
                "background": "transparent"
            })
        );
        let restored: ImageModelOptions = serde_json::from_value(json).unwrap();
        assert_eq!(restored.size, options.size);
        assert_eq!(restored.aspect_ratio, options.aspect_ratio);
        assert_eq!(restored.quality, options.quality);
        assert_eq!(restored.image_size, options.image_size);
        assert_eq!(restored.output_format, options.output_format);
        assert_eq!(restored.background, options.background);
        assert_eq!(restored.n, options.n);
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
            AspectRatio::Auto,
            AspectRatio::Ratio1x1,
            AspectRatio::Ratio1x4,
            AspectRatio::Ratio4x1,
            AspectRatio::Ratio1x8,
            AspectRatio::Ratio8x1,
            AspectRatio::Ratio1x2,
            AspectRatio::Ratio2x1,
            AspectRatio::Ratio2x3,
            AspectRatio::Ratio3x2,
            AspectRatio::Ratio3x4,
            AspectRatio::Ratio4x3,
            AspectRatio::Ratio4x5,
            AspectRatio::Ratio5x4,
            AspectRatio::Ratio5x7,
            AspectRatio::Ratio7x5,
            AspectRatio::Ratio9x16,
            AspectRatio::Ratio16x9,
            AspectRatio::Ratio9x19p5,
            AspectRatio::Ratio19p5x9,
            AspectRatio::Ratio9x20,
            AspectRatio::Ratio20x9,
            AspectRatio::Ratio9x21,
            AspectRatio::Ratio21x9,
        ]);
        assert_wire_matches_serde(&[
            ImageSize::Size512,
            ImageSize::Size768,
            ImageSize::Size1K,
            ImageSize::Size1p5K,
            ImageSize::Size2K,
            ImageSize::Size4K,
        ]);
        assert_wire_matches_serde(&[
            ImageQuality::Auto,
            ImageQuality::Low,
            ImageQuality::Medium,
            ImageQuality::High,
            ImageQuality::XHigh,
            ImageQuality::Max,
        ]);
        let xhigh: &str = ImageQuality::XHigh.into();
        assert_eq!(
            xhigh, "xhigh",
            "the API's spelling, not snake_case's x_high"
        );
        assert_wire_matches_serde(&[
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Webp,
            ImageFormat::Svg,
        ]);
        assert_wire_matches_serde(&[
            ImageBackground::Auto,
            ImageBackground::Transparent,
            ImageBackground::Opaque,
        ]);

        let ratio: &str = AspectRatio::Ratio16x9.into();
        assert_eq!(ratio, "16:9");
        let size: &str = ImageSize::Size512.into();
        assert_eq!(size, "512");
        let ratio: &str = AspectRatio::Ratio9x19p5.into();
        assert_eq!(ratio, "9:19.5");
        let size: &str = ImageSize::Size1p5K.into();
        assert_eq!(size, "1.5K");
        let quality: &str = ImageQuality::Auto.into();
        assert_eq!(quality, "auto");
        assert!(
            serde_json::from_str::<ImageSize>("\"1k\"").is_err(),
            "lowercase k is not a size"
        );
    }

    #[test]
    fn image_format_mime_types() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Webp.mime_type(), "image/webp");
        assert_eq!(ImageFormat::Svg.mime_type(), "image/svg+xml");
    }
}
