use std::{borrow::Cow, fmt, str::FromStr};

use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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

    /// Desired width-to-height ratio.
    ///
    /// Providers accept concrete sizes or a fixed list of ratios, not an
    /// arbitrary ratio, so this snaps to the closest one the provider accepts.
    /// `None` leaves the choice to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<AspectRatio>,

    // ── OpenAI only ───────────────────────────────────────────────────────
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
    /// prompt, and then render the final image. The drafts are hidden unless asked for;
    /// `Some(true)` sends `thinkingConfig.includeThoughts`, and they come back
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
            image_size: Some("2K".to_string()),
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
                "image_size": "2K",
                "output_format": "webp",
                "background": "transparent",
                "include_drafts": true
            })
        );
        let restored: ImageModelOptions = serde_json::from_value(json).unwrap();
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
