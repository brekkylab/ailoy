use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Provider-neutral knobs for one image generation call.
///
/// Every field is optional and `None` leaves the provider's own default in
/// place.  Fields are grouped by who reads them: a common block every provider
/// maps, then one block per provider holding that provider's own parameters
/// under its own names, which other providers skip.  No field is translated
/// into a different concept for a provider that lacks it.  A combination the
/// model cannot honour fails in `marshal_request` instead of quietly returning
/// an image that ignores what was asked for.
///
/// Kept flat, like [`LangModelOptions`](crate::lang_model::LangModelOptions),
/// while there are two providers and few knobs only one of them has. If more
/// providers are added and provider-specific options pile up, consider a
/// common core plus one section per provider (`openai: {..}`, `gemini: {..}`,
/// each read only by its provider) instead: past that point a flat struct
/// grows fields most providers ignore, or tempts one shared field into
/// meaning different things per provider.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOptions {
    // ── Common ────────────────────────────────────────────────────────────
    /// How many images to generate.  `None` means one.
    ///
    /// A provider that cannot return that many from one request refuses the
    /// count at marshal time rather than silently returning fewer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    // ── OpenAI only ───────────────────────────────────────────────────────
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

    // ── Gemini only ───────────────────────────────────────────────────────
    /// Width-to-height ratio, sent verbatim as `imageConfig.aspectRatio`
    /// (e.g. `"16:9"`).
    ///
    /// A string so ratios Google adds need no code change; a refused ratio
    /// comes back as an error listing the accepted ones. `None` leaves the
    /// choice to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,

    /// Output resolution, sent verbatim as `imageConfig.imageSize` (the API
    /// documents `"512"`, `"1K"`, `"2K"`, `"4K"`).
    ///
    /// A string rather than an enum so a size Google adds needs no code
    /// change. Models differ in which sizes they take; a refused size comes
    /// back as an error that names this field. `None` sends no size, so the
    /// model uses its own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,

    /// Also return the model's draft images and reasoning text.
    ///
    /// Thinking image models draw interim versions, check them against the
    /// prompt, and then render the final image. The drafts are hidden unless
    /// asked for; `Some(true)` sends `thinkingConfig.includeThoughts`, and they come back
    /// in [`ImageModelOutput::drafts`] and [`ImageModelOutput::thoughts`],
    /// never in `images`. Roughly doubles the response size.
    ///
    /// [`ImageModelOutput::drafts`]: crate::image_model::ImageModelOutput::drafts
    /// [`ImageModelOutput::thoughts`]: crate::image_model::ImageModelOutput::thoughts
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_drafts: Option<bool>,
}

impl ImageModelOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

/// OpenAI's render quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    Low,
    Medium,
    High,
}

/// Encoding of the returned image bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
}

impl ImageFormat {
    /// The IANA media type for this encoding, which is also what the wire
    /// value is when a provider echoes the format back as a bare word.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Webp => "image/webp",
        }
    }

    /// The bare wire word (`"png"`, `"jpeg"`, `"webp"`) OpenAI's
    /// `output_format` takes.
    pub fn as_wire_str(&self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }
}

/// Whether the generated image keeps an opaque background or an alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
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
            image_size: Some("2K".to_string()),
            output_format: Some(ImageFormat::Webp),
            background: Some(ImageBackground::Transparent),
            aspect_ratio: Some("16:9".to_string()),
            include_drafts: Some(true),
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
                "background": "transparent",
                "include_drafts": true
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
        assert_eq!(restored.include_drafts, options.include_drafts);
    }

    #[test]
    fn image_format_wire_and_mime_values() {
        assert_eq!(ImageFormat::Png.mime_type(), "image/png");
        assert_eq!(ImageFormat::Jpeg.mime_type(), "image/jpeg");
        assert_eq!(ImageFormat::Webp.mime_type(), "image/webp");
        assert_eq!(ImageFormat::Jpeg.as_wire_str(), "jpeg");
    }
}
