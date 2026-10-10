//! Builds language models from a desc, choosing how each one runs.
//!
//! A desc names a model, optionally with a model name after a `/`. The provider set for
//! that model decides how it runs, through its CLI or its API:
//!
//! ```ignore
//! use ailoy::experimental::model::*;
//!
//! // By default, each model runs through its CLI.
//! let model = create_lang_model("claude/sonnet")?;
//! let output = model.infer(&messages, &tools, &LangModelOptions::default()).await?;
//!
//! // Run Claude through the API instead; the same desc now builds a `ClaudeApiModel`.
//! set_lang_model_provider("claude", lang_model_provider::<ClaudeApiModel>());
//! let model = create_lang_model("claude/sonnet")?;
//!
//! // A model of your own: implement `FromModelName`, or pass a closure.
//! set_lang_model_provider("mine", Arc::new(|name| Ok(Arc::new(MyModel::new(name)))));
//! ```
//!
//! | Model      | Default (CLI)   | API                                  |
//! |------------|-----------------|--------------------------------------|
//! | `"claude"` | [`ClaudeModel`] | [`ClaudeApiModel`] (`ANTHROPIC_API_KEY`) |
//! | `"openai"` | [`CodexModel`]  | [`OpenAIApiModel`] (`OPENAI_API_KEY`)    |
//! | `"gemini"` | [`GeminiModel`] | [`GeminiApiModel`] (`GEMINI_API_KEY`)    |

use anyhow::Context as _;
use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use super::{
    ClaudeApiModel, ClaudeModel, CodexModel, GeminiApiModel, GeminiModel, InferLangModel,
    OpenAIApiModel,
};

/// Decides how a model is run (its CLI, its API, …) and builds it. Takes the part of a desc
/// after `<model>/`, `None` when the desc names only the model.
pub type LangModelProvider =
    Arc<dyn Fn(Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> + Send + Sync>;

/// A model that can be built from the name a desc gives it, `None` when the desc gives none.
pub trait FromModelName: InferLangModel + Sized + 'static {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self>;
}

/// A provider that builds `M` through [`FromModelName`].
///
/// ```ignore
/// set_lang_model_provider("claude", lang_model_provider::<ClaudeApiModel>());
/// ```
pub fn lang_model_provider<M: FromModelName>() -> LangModelProvider {
    Arc::new(|name| Ok(Arc::new(M::from_model_name(name)?)))
}

impl FromModelName for ClaudeModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

/// The key is read from `ANTHROPIC_API_KEY` each time a model is built.
impl FromModelName for ClaudeApiModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(name.context("the API needs a model name, as in \"<model>/<name>\"")?)
    }
}

impl FromModelName for CodexModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

impl FromModelName for GeminiModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

/// The key is read from `OPENAI_API_KEY` each time a model is built.
impl FromModelName for OpenAIApiModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(name.context("the API needs a model name, as in \"<model>/<name>\"")?)
    }
}

/// The key is read from `GEMINI_API_KEY` each time a model is built.
impl FromModelName for GeminiApiModel {
    fn from_model_name(name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(name.context("the API needs a model name, as in \"<model>/<name>\"")?)
    }
}

/// Providers keyed by model, seeded to run each one through its CLI.
static LANG_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, LangModelProvider>>> =
    LazyLock::new(|| {
        RwLock::new(HashMap::from([
            ("claude".to_owned(), lang_model_provider::<ClaudeModel>()),
            ("openai".to_owned(), lang_model_provider::<CodexModel>()),
            ("gemini".to_owned(), lang_model_provider::<GeminiModel>()),
        ]))
    });

/// Runs `model` through `provider`, replacing the provider it had.
pub fn set_lang_model_provider(model: impl Into<String>, provider: LangModelProvider) {
    LANG_MODEL_PROVIDERS
        .write()
        .unwrap()
        .insert(model.into(), provider);
}

/// Removes the provider for `model`, returning it if there was one.
pub fn remove_lang_model_provider(model: &str) -> Option<LangModelProvider> {
    LANG_MODEL_PROVIDERS.write().unwrap().remove(model)
}

/// Builds a model from a desc of the form `<model>` or `<model>/<name>`,
/// e.g. `"claude"` or `"openai/gpt-5"`, through the provider set for `<model>`.
pub fn create_lang_model(desc: &str) -> anyhow::Result<Arc<dyn InferLangModel>> {
    let (model, name) = match desc.split_once('/') {
        Some((model, name)) => (model, Some(name)),
        None => (desc, None),
    };
    // Clone out so the provider runs without holding the lock and may itself set one.
    let provider = LANG_MODEL_PROVIDERS
        .read()
        .unwrap()
        .get(model)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no provider set for lang model '{model}'"))?;
    provider(name)
}
