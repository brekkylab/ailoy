//! Builds language models from a desc, choosing how each one runs.
//!
//! A desc names a model, optionally with a model name after a `/`. The provider set for
//! that model decides how it runs, through its CLI, its API, Bedrock or OpenRouter:
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
//! // Or on Bedrock: an alias runs its global inference profile, any other name the id given.
//! set_lang_model_provider("claude", lang_model_provider::<BedrockModel>());
//! let model = create_lang_model("claude/sonnet")?;
//!
//! // A model of your own: implement `FromModelName`, or pass a closure.
//! set_lang_model_provider("mine", Arc::new(|_, name| Ok(Arc::new(MyModel::new(name)))));
//! ```
//!
//! | Model        | CLI                | API                  | Bedrock          | OpenRouter          |
//! |--------------|--------------------|----------------------|------------------|---------------------|
//! | `"claude"`   | [`ClaudeCliModel`] | [`ClaudeApiModel`]   | [`BedrockModel`] | [`OpenRouterModel`] |
//! | `"gpt"`      | [`CodexCliModel`]  | [`GptApiModel`]      | [`BedrockModel`] | [`OpenRouterModel`] |
//! | `"gemini"`   | [`GeminiCliModel`] | [`GeminiApiModel`]   | —                | [`OpenRouterModel`] |
//! | `"deepseek"` | —                  | [`DeepSeekApiModel`] | [`BedrockModel`] | [`OpenRouterModel`] |
//! | `"kimi"`     | —                  | [`KimiApiModel`]     | [`BedrockModel`] | [`OpenRouterModel`] |
//! | `"glm"`      | —                  | [`GlmApiModel`]      | [`BedrockModel`] | [`OpenRouterModel`] |
//!
//! The default is the CLI where there is one, the API otherwise. Each is set with
//! `lang_model_provider::<M>()`. The API models read their key from `ANTHROPIC_API_KEY`,
//! `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY` and
//! `ZAI_API_KEY`; Bedrock from
//! `AWS_BEARER_TOKEN_BEDROCK` (and `AWS_REGION`); OpenRouter from `OPENROUTER_API_KEY`. All
//! but the CLIs need a model name in the desc.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use anyhow::Context as _;

use super::{
    BedrockModel, ClaudeApiModel, ClaudeCliModel, CodexCliModel, DeepSeekApiModel, GeminiApiModel,
    GeminiCliModel, GlmApiModel, GptApiModel, InferLangModel, KimiApiModel, OpenRouterModel,
    r#impl::claude_alias,
};

/// Decides how a model is run (its CLI, its API, …) and builds it. Takes the desc's model
/// (e.g. `"claude"`) and the part after `<model>/`, `None` when the desc names only the
/// model.
pub type LangModelProvider =
    Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> + Send + Sync>;

/// A model that can be built from a desc: its model (e.g. `"claude"`), which a backend
/// serving many vendors' models reads the vendor from, and its name, `None` when the desc
/// gives none.
pub trait FromModelName: InferLangModel + Sized + 'static {
    fn from_model_name(model: &str, name: Option<&str>) -> anyhow::Result<Self>;
}

/// A provider that builds `M` through [`FromModelName`].
///
/// ```ignore
/// set_lang_model_provider("claude", lang_model_provider::<ClaudeApiModel>());
/// ```
pub fn lang_model_provider<M: FromModelName>() -> LangModelProvider {
    Arc::new(|model, name| Ok(Arc::new(M::from_model_name(model, name)?)))
}

/// The vendor that makes `model`, as Bedrock names it.
fn bedrock_vendor(model: &str) -> &str {
    match model {
        "claude" => "anthropic",
        "gpt" => "openai",
        "kimi" => "moonshot",
        "glm" => "zai",
        other => other,
    }
}

/// The vendor that makes `model`, as OpenRouter names it.
fn openrouter_vendor(model: &str) -> &str {
    match model {
        "claude" => "anthropic",
        "gpt" => "openai",
        "gemini" => "google",
        "kimi" => "moonshotai",
        "glm" => "z-ai",
        other => other,
    }
}

fn required(name: Option<&str>) -> anyhow::Result<&str> {
    name.context("this provider needs a model name, as in \"<model>/<name>\"")
}

impl FromModelName for ClaudeCliModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

impl FromModelName for CodexCliModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

impl FromModelName for GeminiCliModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        let model = Self::new();
        Ok(match name {
            Some(name) => model.with_model(name),
            None => model,
        })
    }
}

/// The key is read from `ANTHROPIC_API_KEY` each time a model is built.
impl FromModelName for ClaudeApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The key is read from `OPENAI_API_KEY` each time a model is built.
impl FromModelName for GptApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The key is read from `GEMINI_API_KEY` each time a model is built.
impl FromModelName for GeminiApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The key is read from `DEEPSEEK_API_KEY` each time a model is built.
impl FromModelName for DeepSeekApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The key is read from `KIMI_API_KEY` each time a model is built.
impl FromModelName for KimiApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The key is read from `ZAI_API_KEY` each time a model is built.
impl FromModelName for GlmApiModel {
    fn from_model_name(_: &str, name: Option<&str>) -> anyhow::Result<Self> {
        Self::from_env(required(name)?)
    }
}

/// The name is the Bedrock model id, or for `"claude"` one of the CLI's aliases
/// (`"claude/sonnet"` runs `global.anthropic.claude-sonnet-5-5`). `<vendor>.` is put in
/// front of an id unless it already names
/// a vendor or region (has a `.`) or is an ARN: `"claude/claude-3-5-haiku-20241022-v1:0"`
/// runs `anthropic.claude-3-5-haiku-20241022-v1:0`, and
/// `"claude/global.anthropic.claude-haiku-4-5-20251001-v1:0"` that inference profile. The
/// key is read from `AWS_BEARER_TOKEN_BEDROCK` each time a model is built.
impl FromModelName for BedrockModel {
    fn from_model_name(model: &str, name: Option<&str>) -> anyhow::Result<Self> {
        let name = required(name)?;
        // Claude's aliases run the global inference profile of their model.
        if model == "claude"
            && let Some(id) = claude_alias(name)
        {
            return Self::from_env(format!("global.anthropic.{id}"));
        }
        let id = if name.contains('.') || name.starts_with("arn:") {
            name.to_owned()
        } else {
            format!("{}.{name}", bedrock_vendor(model))
        };
        Self::from_env(id)
    }
}

/// The name is the OpenRouter model id, with `<vendor>/` put in front unless it has a `/`
/// already: `"gemini/gemini-2.5-flash"` runs `google/gemini-2.5-flash`. The key is read
/// from `OPENROUTER_API_KEY` each time a model is built.
impl FromModelName for OpenRouterModel {
    fn from_model_name(model: &str, name: Option<&str>) -> anyhow::Result<Self> {
        let name = required(name)?;
        let id = if name.contains('/') {
            name.to_owned()
        } else {
            format!("{}/{name}", openrouter_vendor(model))
        };
        Self::from_env(id)
    }
}

/// Providers keyed by model, seeded to run each one through its CLI, or its API when it
/// has no CLI.
static LANG_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, LangModelProvider>>> =
    LazyLock::new(|| {
        RwLock::new(HashMap::from([
            ("claude".to_owned(), lang_model_provider::<ClaudeCliModel>()),
            ("gpt".to_owned(), lang_model_provider::<CodexCliModel>()),
            ("gemini".to_owned(), lang_model_provider::<GeminiCliModel>()),
            (
                "deepseek".to_owned(),
                lang_model_provider::<DeepSeekApiModel>(),
            ),
            ("kimi".to_owned(), lang_model_provider::<KimiApiModel>()),
            ("glm".to_owned(), lang_model_provider::<GlmApiModel>()),
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
/// e.g. `"claude"` or `"gpt/gpt-6-sol"`, through the provider set for `<model>`.
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
    provider(model, name)
}
