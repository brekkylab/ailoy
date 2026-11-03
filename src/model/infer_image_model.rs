use futures::future::BoxFuture;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::DeserializeOwned, de::Error as _};

use crate::{
    datatype::Value,
    message::{PartImage, TokenUsage},
};

/// Options for one image generation call: one API's parameters, under that API's own names.
///
/// In JSON the options are one flat object, such as `{"size": "1536x1024", "quality":
/// "high"}`. A model reads them as its own API's parameters, whichever variant they were
/// built as, so options whose fields another API also has work there too; a field its API
/// does not have is an error naming the field, raised before anything is sent.
///
/// Values are sent as written: the API decides which it takes, and a refusal comes back as
/// its error. A combination an API never accepts fails before anything is sent.
///
/// The variant says nothing about the API meant: one deserialized from JSON is the first
/// whose fields hold every key, so `{"n": 1}` written for OpenRouter is [`Self::OpenAI`].
/// Do not branch on it. What the type does is check, where options arrive as JSON, that some
/// API takes them, and give their JSON Schema.
#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum ImageModelOptions {
    /// The OpenAI Images API's parameters, for models with an `openai/…` desc.
    OpenAI(OpenAIImageOptions),

    /// The Gemini API's parameters, for models with a `google/…` desc.
    Gemini(GeminiImageOptions),

    /// The OpenRouter Images API's parameters, for models with an `openrouter/…` desc.
    OpenRouter(OpenRouterImageOptions),
}

/// Takes the first API whose fields hold every key; with none, says why each refused.
impl<'de> Deserialize<'de> for ImageModelOptions {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let openai = match serde_json::from_value(value.clone()) {
            Ok(options) => return Ok(Self::OpenAI(options)),
            Err(e) => e,
        };
        let gemini = match serde_json::from_value(value.clone()) {
            Ok(options) => return Ok(Self::Gemini(options)),
            Err(e) => e,
        };
        let openrouter = match serde_json::from_value(value) {
            Ok(options) => return Ok(Self::OpenRouter(options)),
            Err(e) => e,
        };
        Err(D::Error::custom(format!(
            "no image API takes these options (OpenAI: {openai}; Gemini: {gemini}; \
             OpenRouter: {openrouter})"
        )))
    }
}

impl From<OpenAIImageOptions> for ImageModelOptions {
    fn from(options: OpenAIImageOptions) -> Self {
        Self::OpenAI(options)
    }
}

impl From<GeminiImageOptions> for ImageModelOptions {
    fn from(options: GeminiImageOptions) -> Self {
        Self::Gemini(options)
    }
}

impl From<OpenRouterImageOptions> for ImageModelOptions {
    fn from(options: OpenRouterImageOptions) -> Self {
        Self::OpenRouter(options)
    }
}

/// Options [`InferImageModel::generate`] takes: one API's parameters, which a model reads as
/// its own API's. Every serializable type is one, so [`ImageModelOptions`], an API's options
/// type, a `serde_json::Value` or a model's own options type is passed as it is; anything but
/// a JSON object of parameters fails when the model reads it.
pub trait ImageOptions {
    /// These options as JSON: one flat object of parameters.
    fn to_json(&self) -> serde_json::Result<serde_json::Value>;
}

impl<T: Serialize> ImageOptions for T {
    fn to_json(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self)
    }
}

impl dyn ImageOptions + '_ {
    /// These options as `T`, the parameters of a model's API: a field `T` does not have is an
    /// error naming it. A model of your own reads its options this way.
    pub fn read<T: DeserializeOwned>(&self) -> serde_json::Result<T> {
        self.to_json().and_then(serde_json::from_value)
    }
}

/// The parameters of OpenAI's Images API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenAIImageOptions {
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

    /// Request fields these options do not cover, in the API's own JSON, deep-merged into the
    /// request body after the fields above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// The image parameters of the Gemini API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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

    /// Request fields these options do not cover, in the API's own JSON, deep-merged into the
    /// request body after the fields above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

/// The parameters of OpenRouter's Images API.
#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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

    /// Request fields these options do not cover, in the API's own JSON, deep-merged into the
    /// request body after the fields above.
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
    /// Generate images from `prompt`, with `options` or else the API's defaults.
    ///
    /// An `Ok` result carries at least one image. A call that ends with none, such as one
    /// stopped by a safety filter, is an `Err` quoting the reason the provider gave.
    fn generate(
        &self,
        prompt: &str,
        options: Option<&dyn ImageOptions>,
    ) -> BoxFuture<'static, anyhow::Result<ImageOutput>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_are_one_flat_object_without_unset_fields() {
        let options: ImageModelOptions = GeminiImageOptions {
            aspect_ratio: Some("16:9".to_owned()),
            ..Default::default()
        }
        .into();
        assert!(matches!(options, ImageModelOptions::Gemini(_)));
        assert_eq!(
            serde_json::to_string(&options).unwrap(),
            r#"{"aspect_ratio":"16:9"}"#
        );
        let empty = ImageModelOptions::OpenAI(OpenAIImageOptions::default());
        assert_eq!(serde_json::to_string(&empty).unwrap(), "{}");
    }

    #[test]
    fn options_serde_round_trip() {
        for json in [
            serde_json::json!({
                "n": 2,
                "size": "1536x1024",
                "quality": "high",
                "output_format": "webp",
                "background": "transparent",
                "extra": {"moderation": "low"}
            }),
            serde_json::json!({
                "aspect_ratio": "16:9",
                "image_size": "2K",
                "thinking_level": "minimal",
                "extra": {"generationConfig": {"candidateCount": 1}}
            }),
            serde_json::json!({
                "n": 1,
                "size": "1024x1024",
                "resolution": "1.5K",
                "aspect_ratio": "9:19.5",
                "quality": "auto",
                "output_format": "svg",
                "background": "auto",
                "extra": {"seed": 7}
            }),
        ] {
            let options: ImageModelOptions = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(serde_json::to_value(&options).unwrap(), json);
        }
    }

    #[test]
    fn deserializing_takes_the_first_api_that_has_every_field() {
        let read = |json| serde_json::from_value::<ImageModelOptions>(json).unwrap();
        assert!(matches!(
            read(serde_json::json!({"n": 2, "quality": "high"})),
            ImageModelOptions::OpenAI(_)
        ));
        assert!(matches!(
            read(serde_json::json!({"aspect_ratio": "16:9", "thinking_level": "high"})),
            ImageModelOptions::Gemini(_)
        ));
        assert!(matches!(
            read(serde_json::json!({"aspect_ratio": "16:9", "resolution": "2K"})),
            ImageModelOptions::OpenRouter(_)
        ));
    }

    #[test]
    fn options_no_api_takes_say_why_each_refused() {
        let err = serde_json::from_value::<ImageModelOptions>(serde_json::json!({
            "n": 2,
            "thinking_level": "high"
        }))
        .unwrap_err()
        .to_string();
        assert!(err.contains("no image API takes these options"), "{err}");
        assert!(err.contains("unknown field `thinking_level`"), "{err}");
        assert!(err.contains("unknown field `n`"), "{err}");

        let err = serde_json::from_value::<ImageModelOptions>(serde_json::json!({"n": "two"}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid type"), "{err}");
    }
}
