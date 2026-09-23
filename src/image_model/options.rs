use std::{borrow::Cow, fmt, str::FromStr};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Provider-neutral knobs for one image generation call.
///
/// Every field is optional and `None` leaves the provider's own default in
/// place.  The wire mapping is per-provider and deliberately lossy: a knob the
/// target model has no equivalent for is dropped silently (each field documents
/// where that happens), while a combination the model cannot honour fails in
/// `marshal_request` instead of quietly returning an image that ignores what
/// was asked for.
///
/// Kept flat, like [`LangModelOptions`](crate::lang_model::LangModelOptions),
/// while there are two providers and few knobs only one of them has. If more
/// providers are added and provider-specific options pile up, consider a
/// common core plus one section per provider (`openai: {..}`, `gemini: {..}`,
/// each read only by its provider) instead: past that point a flat struct
/// either grows fields most providers ignore or forces one shared field to
/// mean different things per provider, as `quality` already does here.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ImageModelOptions {
    /// How many images to generate.  `None` means one.
    ///
    /// Only OpenAI's `gpt-image-*` family honours values above 1 (1–10);
    /// Gemini rejects them at marshal time rather than silently returning a
    /// single image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Desired width-to-height ratio.
    ///
    /// Neither provider takes a ratio literally: OpenAI picks the closest of
    /// its fixed `size` values (square / landscape / portrait) and Gemini picks
    /// the nearest of the ratios its `imageConfig.aspectRatio` accepts.  `None`
    /// leaves the choice to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<AspectRatio>,

    /// Render quality, coarsely.  Maps to OpenAI's `quality` and to Gemini's
    /// `imageConfig.imageSize` (`low` → `512`, `medium` → `1K`, `high` → `2K`),
    /// which is the only resolution knob that API exposes.
    ///
    /// Not every Gemini model takes every size: `gemini-3.1-flash-lite-image`
    /// takes `1K` only, and `gemini-3-pro-image` / `gemini-2.5-flash-image`
    /// refuse `512`. The request is sent as asked and the
    /// model's refusal comes back as an error that names this option. Leaving
    /// `quality` unset sends no size at all, so the model uses its own default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,

    /// Encoding of the returned bytes.  Honoured by OpenAI's `gpt-image-*`
    /// only; Gemini decides for itself (the response's own mime type is what
    /// [`ImageModelOutput`] carries), so the field is ignored there.
    ///
    /// [`ImageModelOutput`]: crate::image_model::ImageModelOutput
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<ImageFormat>,

    /// Whether the image background should be transparent.  Honoured by
    /// OpenAI's `gpt-image-*` only; ignored by Gemini.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<ImageBackground>,

    /// Also return the model's draft images and reasoning text.
    ///
    /// Thinking image models (Gemini's Nano Banana Pro, and the flash models
    /// that think) draw interim versions, check them against the prompt, and
    /// then render the final image. The drafts are hidden unless asked for;
    /// `Some(true)` asks for them, and they come back in
    /// [`ImageModelOutput::drafts`] and [`ImageModelOutput::thoughts`], never
    /// in `images`. Roughly doubles the response size. Ignored by OpenAI,
    /// which has no drafts.
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

/// A width-to-height ratio, serialized as the `"W:H"` string (e.g. `"16:9"`).
///
/// Kept as the two integers rather than a float so it round-trips through JSON
/// exactly as written; [`ratio`](Self::ratio) is what the marshals compare
/// against the discrete sizes their APIs accept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AspectRatio {
    pub w: u32,
    pub h: u32,
}

impl AspectRatio {
    /// Width divided by height: `> 1` is landscape, `< 1` portrait, `1` square.
    pub fn ratio(&self) -> f64 {
        f64::from(self.w) / f64::from(self.h)
    }
}

impl fmt::Display for AspectRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.w, self.h)
    }
}

impl FromStr for AspectRatio {
    type Err = anyhow::Error;

    /// Parses `"W:H"`.  Both sides must be positive integers — a zero would
    /// make [`ratio`](Self::ratio) meaningless (zero or infinite), so it is
    /// rejected here instead of at the provider.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (w, h) = s
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("invalid aspect ratio '{s}': expected \"W:H\""))?;
        let parse = |part: &str, side: &str| -> anyhow::Result<u32> {
            let v: u32 = part.parse().map_err(|_| {
                anyhow::anyhow!("invalid aspect ratio '{s}': {side} is not a number")
            })?;
            if v == 0 {
                anyhow::bail!("invalid aspect ratio '{s}': {side} must be greater than zero");
            }
            Ok(v)
        };
        Ok(Self {
            w: parse(w, "width")?,
            h: parse(h, "height")?,
        })
    }
}

impl Serialize for AspectRatio {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AspectRatio {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl JsonSchema for AspectRatio {
    fn schema_name() -> Cow<'static, str> {
        "AspectRatio".into()
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        schemars::json_schema!({
            "type": "string",
            "pattern": r"^[1-9][0-9]*:[1-9][0-9]*$"
        })
    }
}

/// Coarse render-quality request, three steps wide because that is the
/// granularity every supported provider agrees on.
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
    fn aspect_ratio_parses_and_displays() {
        let ar: AspectRatio = "16:9".parse().unwrap();
        assert_eq!(ar, AspectRatio { w: 16, h: 9 });
        assert_eq!(ar.to_string(), "16:9");
        assert!((ar.ratio() - 16.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn aspect_ratio_serde_round_trip() {
        let ar = AspectRatio { w: 4, h: 3 };
        let json = serde_json::to_string(&ar).unwrap();
        assert_eq!(json, "\"4:3\"");
        assert_eq!(serde_json::from_str::<AspectRatio>(&json).unwrap(), ar);
    }

    #[test]
    fn aspect_ratio_rejects_invalid_strings() {
        for s in [
            "", "16", "16:9:4", "a:b", "16:", ":9", "16:0", "0:9", "-16:9", "16 : 9", "16/9",
        ] {
            assert!(s.parse::<AspectRatio>().is_err(), "{s:?} must not parse");
        }
        assert!(serde_json::from_str::<AspectRatio>("\"16\"").is_err());
        assert!(serde_json::from_str::<AspectRatio>("169").is_err());
    }

    #[test]
    fn options_omit_unset_fields() {
        let json = serde_json::to_string(&ImageModelOptions::new()).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn options_serde_round_trip() {
        let options = ImageModelOptions {
            n: Some(2),
            aspect_ratio: Some(AspectRatio { w: 16, h: 9 }),
            quality: Some(ImageQuality::High),
            output_format: Some(ImageFormat::Webp),
            background: Some(ImageBackground::Transparent),
            include_drafts: Some(true),
        };
        let json = serde_json::to_value(&options).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "n": 2,
                "aspect_ratio": "16:9",
                "quality": "high",
                "output_format": "webp",
                "background": "transparent",
                "include_drafts": true
            })
        );
        let restored: ImageModelOptions = serde_json::from_value(json).unwrap();
        assert_eq!(restored.aspect_ratio, options.aspect_ratio);
        assert_eq!(restored.quality, options.quality);
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
