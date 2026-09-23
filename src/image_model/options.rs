use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Options for one image generation call.
///
/// Every field is optional; `None` leaves the provider's own default in place.
/// Each field carries one provider's parameter under that provider's own name
/// and is read only by that provider; the others skip it. No field is
/// translated into a different concept for a provider that lacks it, so a value
/// always means what the owning provider's API says. A combination the model
/// cannot honour fails in `marshal_request` rather than being sent.
///
/// The groups describe the providers supported today, not a rule: the
/// two current APIs happen to share no option (they describe even an image's
/// shape differently), which is why there is no common group. Adding a
/// provider can change the grouping — a field another provider also takes
/// under the same meaning becomes shared — and past a few providers, a common
/// core plus one section per provider (`openai: {..}`, `gemini: {..}`) is
/// worth considering over this flat layout, kept flat for now like
/// [`LangModelOptions`](crate::lang_model::LangModelOptions).
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOptions {
    // ── OpenAI ────────────────────────────────────────────────────────────
    /// How many images to generate, sent as `n` (the API takes 1–10).  `None`
    /// means one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Pixel size, sent verbatim as `size` (e.g. `"1536x1024"`, `"auto"`).
    ///
    /// A string so the sizes a model accepts, which differ by model, need no
    /// code change; a refused size comes back as an error listing the accepted
    /// ones. `None` leaves the choice to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,

    /// Render quality, sent as `quality`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,

    /// Encoding of the returned bytes, sent as `output_format`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageFormat>,

    /// Background, sent as `background`.  `Transparent` needs an alpha-capable
    /// format, so it is refused together with `output_format: jpeg`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageBackground>,

    // ── Gemini ────────────────────────────────────────────────────────────
    /// Width-to-height ratio, sent as `imageConfig.aspectRatio`.  `None`
    /// leaves the choice to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<AspectRatio>,

    /// Output resolution, sent as `imageConfig.imageSize`.  The pixel
    /// dimensions follow from this and [`aspect_ratio`](Self::aspect_ratio)
    /// together: the size sets roughly the pixel count, the ratio its shape.
    ///
    /// Models differ in which sizes they take; a refused size comes back as an
    /// error that names this field. `None` sends no size, so the model uses its
    /// own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<ImageSize>,
}

impl ImageModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

// The wire string of each enum below is its serde name, and `strum` gives the
// same string as `&'static str` through `Into` (`let wire: &str = x.into()`),
// so the marshals need no hand-written match. The two attributes sit on the
// same variant to keep them from drifting apart.

/// Gemini's output aspect ratio, the values its `imageConfig.aspectRatio`
/// accepts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum AspectRatio {
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

/// Gemini's output resolution, the values its `imageConfig.imageSize`
/// accepts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
pub enum ImageSize {
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

/// OpenAI's render quality, the values its `quality` accepts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageQuality {
    Low,
    Medium,
    High,
    /// Introduced with the `gpt-image-2.5` models; earlier models refuse it.
    /// Named explicitly: `snake_case` would spell it `x_high`.
    #[serde(rename = "xhigh")]
    #[strum(serialize = "xhigh")]
    XHigh,
    /// Introduced with the `gpt-image-2.5` models; earlier models refuse it.
    Max,
}

/// OpenAI's output encoding, the values its `output_format` accepts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
}

impl ImageFormat {
    /// The IANA media type for this encoding.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }
}

/// OpenAI's background, the values its `background` accepts.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema, strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ImageBackground {
    Transparent,
    Opaque,
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

    /// The serde name and the `strum` string are two attributes on each
    /// variant; the JSON a caller writes and the value sent on the wire must
    /// be the same string.
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
            AspectRatio::Ratio1x1,
            AspectRatio::Ratio1x4,
            AspectRatio::Ratio4x1,
            AspectRatio::Ratio1x8,
            AspectRatio::Ratio8x1,
            AspectRatio::Ratio2x3,
            AspectRatio::Ratio3x2,
            AspectRatio::Ratio3x4,
            AspectRatio::Ratio4x3,
            AspectRatio::Ratio4x5,
            AspectRatio::Ratio5x4,
            AspectRatio::Ratio9x16,
            AspectRatio::Ratio16x9,
            AspectRatio::Ratio21x9,
        ]);
        assert_wire_matches_serde(&[
            ImageSize::Size512,
            ImageSize::Size1K,
            ImageSize::Size2K,
            ImageSize::Size4K,
        ]);
        assert_wire_matches_serde(&[
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
        assert_wire_matches_serde(&[ImageFormat::Png, ImageFormat::Jpeg, ImageFormat::Webp]);
        assert_wire_matches_serde(&[ImageBackground::Transparent, ImageBackground::Opaque]);

        let ratio: &str = AspectRatio::Ratio16x9.into();
        assert_eq!(ratio, "16:9");
        let size: &str = ImageSize::Size512.into();
        assert_eq!(size, "512");
        assert!(
            serde_json::from_str::<ImageSize>("\"1k\"").is_err(),
            "lowercase k is not a size"
        );
    }

    #[test]
    fn image_format_wire_and_mime_values() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Webp.mime_type(), "image/webp");
    }
}
