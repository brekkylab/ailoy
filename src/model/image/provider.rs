//! Builds image models from a desc, in a table of their own beside the language models'.
//!
//! A desc is `<org>/<name>`, the name being the model's official API id, and the provider
//! set for the org decides how it runs: the org's own API, or OpenRouter.
//!
//! ```ignore
//! use ailoy::model::*;
//!
//! let model = create_image_model("openai/gpt-image-1").await?;
//! let output = model.generate("a red cube", &ImageModelOptions::default()).await?;
//!
//! // Run Gemini through OpenRouter instead, or an org that has no image API here.
//! set_image_model_provider("google", Backend::OpenRouter);
//! set_image_model_provider("bytedance-seed", Backend::OpenRouter);
//! let model = create_image_model("bytedance-seed/seedream-5-0-pro").await?;
//!
//! // A model of your own.
//! set_image_model_provider("mine", ImageModelProvider::custom(|_, name| Ok(Arc::new(MyModel::new(name)))));
//! ```
//!
//! | Org      | Own API                 | Through OpenRouter            |
//! |----------|-------------------------|-------------------------------|
//! | `openai` | [`GptImageApiModel`]    | [`OpenRouterImageApiModel`]   |
//! | `google` | [`GeminiImageApiModel`] | [`OpenRouterImageApiModel`]   |
//! | any      | —                       | [`OpenRouterImageApiModel`]   |
//!
//! The default for `openai` and `google` is the own API, which reads its key from
//! `OPENAI_API_KEY` or `GEMINI_API_KEY`; OpenRouter reads `OPENROUTER_API_KEY` and runs the
//! desc as its slug. Any other org runs once a provider is set for it.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use anyhow::{Context as _, bail};

use crate::model::{
    Backend, GeminiImageApiModel, GptImageApiModel, InferImageModel, OpenRouterImageApiModel,
};

/// Builds an image model of your own from the desc's org and the name after it, `None`
/// when the desc names only the org.
pub type CustomImageModelProvider =
    Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferImageModel>> + Send + Sync>;

/// How the image models of an org are built.
#[derive(Clone)]
pub enum ImageModelProvider {
    Backend(Backend),
    Custom(CustomImageModelProvider),
}

impl ImageModelProvider {
    pub fn custom(
        f: impl Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferImageModel>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::Custom(Arc::new(f))
    }
}

impl From<Backend> for ImageModelProvider {
    fn from(backend: Backend) -> Self {
        Self::Backend(backend)
    }
}

fn required(name: Option<&str>) -> anyhow::Result<&str> {
    name.context("this backend needs a model name, as in \"<org>/<name>\"")
}

/// Builds `name` of `org` to run through `backend`: as its API model id on the org's own
/// API, and as `<org>/<name>` on OpenRouter, whose slugs for these orgs' models are the
/// same ids.
fn build(
    backend: Backend,
    org: &str,
    name: Option<&str>,
) -> anyhow::Result<Arc<dyn InferImageModel>> {
    Ok(match (backend, org) {
        (Backend::Api, "openai") => Arc::new(GptImageApiModel::from_env(required(name)?)?),
        (Backend::Api, "google") => Arc::new(GeminiImageApiModel::from_env(required(name)?)?),
        (Backend::Api, _) => bail!("'{org}' has no image API"),
        (Backend::OpenRouter, _) => Arc::new(OpenRouterImageApiModel::from_env(format!(
            "{org}/{}",
            required(name)?
        ))?),
        (backend, _) => {
            bail!("image models run only through the org's own API or OpenRouter, not {backend}")
        }
    })
}

/// Providers keyed by org, seeded to run each one through its own API.
static IMAGE_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, ImageModelProvider>>> =
    LazyLock::new(|| {
        RwLock::new(
            [("openai", Backend::Api), ("google", Backend::Api)]
                .into_iter()
                .map(|(org, backend)| (org.to_owned(), backend.into()))
                .collect(),
        )
    });

/// Runs the image models of `org` through `provider`, replacing the provider it had.
pub fn set_image_model_provider(org: impl Into<String>, provider: impl Into<ImageModelProvider>) {
    IMAGE_MODEL_PROVIDERS
        .write()
        .unwrap()
        .insert(org.into(), provider.into());
}

/// The provider a desc's image model is built through, the one set for its org, if there
/// is one. The desc is as [`create_image_model`] takes it.
pub fn get_image_model_provider(desc: &str) -> Option<ImageModelProvider> {
    let org = desc.split_once('/').map_or(desc, |(org, _)| org);
    IMAGE_MODEL_PROVIDERS.read().unwrap().get(org).cloned()
}

/// Removes the image model provider for `org`, returning it if there was one.
pub fn remove_image_model_provider(org: &str) -> Option<ImageModelProvider> {
    IMAGE_MODEL_PROVIDERS.write().unwrap().remove(org)
}

/// Builds an image model from a desc of the form `<org>/<name>`, e.g.
/// `"openai/gpt-image-1"`, through the provider set for `<org>`.
pub async fn create_image_model(desc: &str) -> anyhow::Result<Arc<dyn InferImageModel>> {
    let (org, name) = match desc.split_once('/') {
        Some((org, name)) => (org, Some(name)),
        None => (desc, None),
    };
    // Clone out so the lock is not held while building.
    let provider = IMAGE_MODEL_PROVIDERS
        .read()
        .unwrap()
        .get(org)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no provider set for image model org '{org}'"))?;
    match provider {
        ImageModelProvider::Backend(backend) => build(backend, org, name),
        ImageModelProvider::Custom(f) => f(org, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_an_image_provider_reads_back_as_it_was_set() {
        let is = |org| match get_image_model_provider(org) {
            Some(ImageModelProvider::Backend(backend)) => Some(backend),
            _ => None,
        };
        assert_eq!(is("openai/gpt-image-1"), Some(Backend::Api));
        assert_eq!(is("google"), Some(Backend::Api));
        assert_eq!(is("anthropic"), None, "the lang model table is apart");

        set_image_model_provider("image-provider-read-back", Backend::Bedrock);
        assert_eq!(
            is("image-provider-read-back/some-model"),
            Some(Backend::Bedrock)
        );
        remove_image_model_provider("image-provider-read-back");
        assert!(get_image_model_provider("image-provider-read-back").is_none());
    }

    #[tokio::test]
    async fn test_unknown_org_and_unsupported_backend() {
        let err = create_image_model("nobody/model").await.err().unwrap();
        assert!(err.to_string().contains("nobody"), "{err}");

        let err = build(Backend::Bedrock, "openai", Some("gpt-image-1"))
            .err()
            .unwrap();
        assert!(err.to_string().contains("bedrock"), "{err}");

        let err = build(Backend::Api, "zai-org", Some("glm-image"))
            .err()
            .unwrap();
        assert!(err.to_string().contains("no image API"), "{err}");

        let err = build(Backend::Api, "openai", None).err().unwrap();
        assert!(err.to_string().contains("model name"), "{err}");
    }

    #[test]
    fn test_openrouter_builds_for_any_org() {
        // Without a key the build fails at the key, past the backend and name checks.
        let result = build(
            Backend::OpenRouter,
            "bytedance-seed",
            Some("seedream-5-0-pro"),
        );
        match std::env::var("OPENROUTER_API_KEY") {
            Ok(key) if !key.trim().is_empty() => assert!(result.is_ok()),
            _ => {
                let err = result.err().unwrap().to_string();
                assert!(err.contains("OPENROUTER_API_KEY"), "{err}");
            }
        }

        let err = build(Backend::OpenRouter, "bytedance-seed", None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("model name"), "{err}");
    }

    #[tokio::test]
    async fn test_a_custom_provider_builds_the_model() {
        struct Fixed;
        impl InferImageModel for Fixed {
            fn generate(
                &self,
                prompt: &str,
                _options: &crate::model::ImageModelOptions,
            ) -> futures::future::BoxFuture<'static, anyhow::Result<crate::model::ImageOutput>>
            {
                let prompt = prompt.to_owned();
                Box::pin(async move { bail!("fixed: {prompt}") })
            }
        }

        set_image_model_provider(
            "image-provider-custom",
            ImageModelProvider::custom(|org, name| {
                assert_eq!((org, name), ("image-provider-custom", Some("m")));
                Ok(Arc::new(Fixed))
            }),
        );
        let model = create_image_model("image-provider-custom/m").await;
        remove_image_model_provider("image-provider-custom");
        let err = model
            .unwrap()
            .generate("a red cube", &Default::default())
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "fixed: a red cube");
    }
}
