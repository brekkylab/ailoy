//! Builds image models from a desc.
//!
//! A desc is `<org>/<name>`. For `openai` and `google` the name is the model's official
//! API id; for `openrouter` it is OpenRouter's model slug as is, `/` and all. A model of your
//! own is any type that implements [`InferImageModel`], used directly.
//!
//! ```ignore
//! use ailoy::model::*;
//!
//! let model = create_image_model("openai/gpt-image-1").await?;
//! let options: ImageModelOptions = OpenAIImageOptions {
//!     quality: Some("high".to_owned()),
//!     ..Default::default()
//! }
//! .into();
//! let output = model.generate("a red cube", Some(&options)).await?;
//!
//! let model = create_image_model("google/gemini-3.1-flash-image").await?;
//! let output = model.generate("a red cube", None).await?;
//!
//! // Through OpenRouter, which runs `bytedance-seed/seedream-5-0-pro`.
//! let model = create_image_model("openrouter/bytedance-seed/seedream-5-0-pro").await?;
//! ```
//!
//! | Org          | Model                       | Key                  |
//! |--------------|-----------------------------|----------------------|
//! | `openai`     | [`GptImageApiModel`]        | `OPENAI_API_KEY`     |
//! | `google`     | [`GeminiImageApiModel`]     | `GEMINI_API_KEY`     |
//! | `openrouter` | [`OpenRouterImageApiModel`] | `OPENROUTER_API_KEY` |

use std::sync::Arc;

use anyhow::{Context as _, bail};

use crate::model::{
    GeminiImageApiModel, GptImageApiModel, InferImageModel, OpenRouterImageApiModel,
};

/// Builds an image model from a desc of the form `<org>/<name>`, e.g.
/// `"openai/gpt-image-1"`; everything after the first `/` is the name.
pub async fn create_image_model(desc: &str) -> anyhow::Result<Arc<dyn InferImageModel>> {
    let (org, name) = match desc.split_once('/') {
        Some((org, name)) => (org, Some(name)),
        None => (desc, None),
    };
    let name = || name.context("an image model needs a model name, as in \"<org>/<name>\"");
    Ok(match org {
        "openai" => Arc::new(GptImageApiModel::from_env(name()?)?),
        "google" => Arc::new(GeminiImageApiModel::from_env(name()?)?),
        "openrouter" => Arc::new(OpenRouterImageApiModel::from_env(name()?)?),
        _ => bail!("no image models for org '{org}'; the orgs are openai, google and openrouter"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_an_unknown_org_or_a_missing_name_is_refused() {
        let err = create_image_model("nobody/model").await.err().unwrap();
        assert!(err.to_string().contains("'nobody'"), "{err}");
        assert!(err.to_string().contains("openrouter"), "{err}");

        for desc in ["openai", "google", "openrouter"] {
            let err = create_image_model(desc).await.err().unwrap();
            assert!(err.to_string().contains("model name"), "{desc}: {err}");
        }
    }

    #[tokio::test]
    async fn test_openrouter_builds_from_its_slug() {
        // Built, or stopped at the key; another test may load `.env` meanwhile, so either
        // passes. The name gets past the checks before the key either way.
        match create_image_model("openrouter/x-org/some-model").await {
            // An out-of-range `n` is refused before sending, in OpenRouter's words.
            Ok(model) => {
                let options = crate::model::ImageModelOptions::OpenRouter(
                    crate::model::OpenRouterImageOptions {
                        n: Some(0),
                        ..Default::default()
                    },
                );
                let err = model.generate("x", Some(&options)).await.unwrap_err();
                assert!(err.to_string().contains("OpenRouter"), "{err}");
            }
            Err(err) => assert!(err.to_string().contains("OPENROUTER_API_KEY"), "{err}"),
        }
    }
}
