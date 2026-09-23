mod gemini;
mod openai;

pub use gemini::GeminiImageApi;
pub use openai::OpenAIImageApi;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    image_model::{ImageModelOutput, ImageModelRequest},
};

/// Wire protocol used when calling an image generation API.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImageModelAPISchema {
    /// OpenAI Images API (`POST /v1/images/generations`), serving the
    /// `gpt-image-*` family. DALL·E 2/3 were removed from the API,
    /// so this schema has no per-family branching.
    #[default]
    #[serde(rename = "openai")]
    OpenAI,

    /// Gemini API `POST {base}{model}:generateContent` with
    /// `generationConfig.responseModalities` including `IMAGE` — the Nano
    /// Banana (`gemini-*-image`) models.  Same base URL, auth header and
    /// response skeleton as the text-side Gemini schema.
    Gemini,
}

/// Provider-specific request building and response decoding for image
/// generation, dispatched dynamically so a caller maps an
/// [`ImageModelAPISchema`] to its implementation once and reuses it for the
/// whole call.
///
/// Deliberately not the text side's [`Marshal`](crate::message::Marshal) /
/// [`Unmarshal`](crate::message::Unmarshal) pair: `Unmarshal` is bounded on
/// `Delta` because it exists to accumulate streaming chunks, and an image
/// response is not a delta.  `marshal_request` also has to be fallible, which
/// `Marshal` (returning a bare `Value`) cannot express — see its own doc.
pub trait ImageProviderApi {
    /// Build the wire request as the `{"url", "header", "body"}` object the
    /// runtime unpacks.
    ///
    /// Fallible because option combinations that the target model cannot honour
    /// (`n > 1` on Gemini, a transparent background on a JPEG)
    /// are better refused here than sent and silently ignored — or, worse,
    /// answered with a 400 whose message says nothing about which option caused
    /// it.
    fn marshal_request(&self, req: &ImageModelRequest<'_>) -> anyhow::Result<Value>;

    /// Decode a successful response body.
    ///
    /// Contract: an `Ok` result carries at least one image.  A response that
    /// finished without producing one (safety filter, text-only answer) is an
    /// `Err` quoting whatever reason the provider gave, so callers never have
    /// to inspect an empty vector.
    fn unmarshal_response(&self, val: Value) -> anyhow::Result<ImageModelOutput>;

    /// Classifies a 429 body as permanent quota exhaustion (don't retry) vs a
    /// transient rate limit. Defaults to transient.
    fn is_permanent_quota_error(&self, _body: &str) -> bool {
        false
    }

    /// Explains a failed response in terms of the options the caller set.
    ///
    /// A provider's error names its own wire field, which may not be the
    /// option the caller touched (Gemini rejects `imageSize`, which the caller
    /// reached through `quality`). Returning `Some` puts that explanation in
    /// front of the provider's body in the error; the body itself is always
    /// kept. The model is left to decide what it accepts — this runs only
    /// after it has said no — so no per-model rule table has to be kept up to
    /// date. Defaults to no explanation.
    fn explain_error(
        &self,
        _req: &ImageModelRequest<'_>,
        _status: u16,
        _body: &str,
    ) -> Option<String> {
        None
    }
}

/// Maps a wire schema to its provider implementation.
pub fn provider_api(schema: &ImageModelAPISchema) -> Box<dyn ImageProviderApi + Send + Sync> {
    match schema {
        ImageModelAPISchema::OpenAI => Box::new(OpenAIImageApi),
        ImageModelAPISchema::Gemini => Box::new(GeminiImageApi),
    }
}
