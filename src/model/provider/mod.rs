//! Builds language models from a desc, choosing how each one runs.
//!
//! A desc is `<org>/<name>`, or `<org>` alone for a CLI's default model. The provider set
//! for the org decides how it runs: its CLI, its own API, Bedrock, OpenRouter or vLLM.
//!
//! ```ignore
//! use ailoy::model::*;
//!
//! // By default, each model runs through its CLI.
//! let model = create_lang_model("anthropic/claude-sonnet-5-5").await?;
//! let output = model.infer(&messages, &tools, &LangModelOptions::default()).await?;
//!
//! // Run Claude through the API instead; the same desc now builds a `ClaudeApiModel`.
//! set_lang_model_provider("anthropic", Backend::Api);
//! let model = create_lang_model("anthropic/claude-sonnet-5-5").await?;
//!
//! // Or on Bedrock, which runs `global.anthropic.claude-sonnet-5-5`.
//! set_lang_model_provider("anthropic", Backend::Bedrock);
//! let model = create_lang_model("anthropic/claude-sonnet-5-5").await?;
//!
//! // A model of your own.
//! set_lang_model_provider("mine", LangModelProvider::custom(|_, name| Ok(Arc::new(MyModel::new(name)))));
//! ```
//!
//! # Naming
//!
//! A desc names a model as the one who makes it does, and the same desc runs it through
//! every backend:
//!
//! | Models               | Naming                          | Desc                                 |
//! |----------------------|---------------------------------|--------------------------------------|
//! | Claude               | `anthropic/<official API id>`   | `anthropic/claude-sonnet-5-5`        |
//! | GPT                  | `openai/<official API id>`      | `openai/gpt-6.1-sol`                 |
//! | Gemini               | `google/<official API id>`      | `google/gemini-3.1-pro`              |
//! | DeepSeek             | Hugging Face repo id            | `deepseek-ai/DeepSeek-V3.2`          |
//! | Kimi                 | Hugging Face repo id            | `moonshotai/Kimi-K2-Thinking`        |
//! | GLM                  | Hugging Face repo id            | `zai-org/GLM-4.6`                    |
//! | Qwen                 | Hugging Face repo id            | `Qwen/Qwen3-235B-A22B-Instruct-2507` |
//!
//! `<org>` alone runs the CLI's default model.
//!
//! # Backends
//!
//! | Org           | CLI                | APIs                                                                             |
//! |---------------|--------------------|----------------------------------------------------------------------------------|
//! | `anthropic`   | [`ClaudeCliModel`] | [`ClaudeApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`]                  |
//! | `openai`      | [`CodexCliModel`]  | [`GptApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`]                     |
//! | `google`      | [`GeminiCliModel`] | [`GeminiApiModel`], [`OpenRouterApiModel`]                                       |
//! | `deepseek-ai` | —                  | [`DeepSeekApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`VllmModel`] |
//! | `moonshotai`  | —                  | [`KimiApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`VllmModel`]     |
//! | `zai-org`     | —                  | [`GlmApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`VllmModel`]      |
//! | `Qwen`        | —                  | [`QwenApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`VllmModel`]     |
//!
//! The default is the CLI where there is one, otherwise the org's own API. The own APIs
//! read their key from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
//! `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `ZAI_API_KEY` and `DASHSCOPE_API_KEY`; Bedrock from
//! `AWS_BEARER_TOKEN_BEDROCK` (and `AWS_REGION`); OpenRouter from `OPENROUTER_API_KEY`;
//! vLLM from the server at `VLLM_BASE_URL` (by default `http://localhost:8000/v1`), with
//! `VLLM_API_KEY` if it is set. All but the CLIs need a name in the desc.

mod catalog;

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use anyhow::{Context as _, bail};

use super::{
    BedrockApiModel, ClaudeApiModel, ClaudeCliModel, CodexCliModel, DeepSeekApiModel,
    GeminiApiModel, GeminiCliModel, GlmApiModel, GptApiModel, InferLangModel, KimiApiModel,
    OpenRouterApiModel, QwenApiModel, VllmModel, r#impl::env_key_and_region,
};

/// How a model is run, spelled `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"`
/// in its serde form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// The org's CLI: Claude Code, Codex or Gemini CLI.
    Cli,
    /// The org's own API.
    Api,
    Bedrock,
    OpenRouter,
    Vllm,
}

/// Builds a model of your own from the desc's org and the name after it, `None` when the
/// desc names only the org.
pub type CustomLangModelProvider =
    Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> + Send + Sync>;

/// How the models of an org are built.
#[derive(Clone)]
pub enum LangModelProvider {
    Backend(Backend),
    Custom(CustomLangModelProvider),
}

impl LangModelProvider {
    pub fn custom(
        f: impl Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self::Custom(Arc::new(f))
    }
}

impl From<Backend> for LangModelProvider {
    fn from(backend: Backend) -> Self {
        Self::Backend(backend)
    }
}

/// Whether the org's models are closed, named by their official API id; the rest are
/// open-weight, named by their Hugging Face repo id.
fn is_closed(org: &str) -> bool {
    matches!(org, "anthropic" | "openai" | "google")
}

/// The vendor that makes the org's models, as Bedrock names it.
fn bedrock_vendor(org: &str) -> &str {
    match org {
        "deepseek-ai" => "deepseek",
        "moonshotai" => "moonshot",
        "zai-org" => "zai",
        "Qwen" => "qwen",
        other => other,
    }
}

/// The vendor that makes the org's models, as OpenRouter names it.
fn openrouter_vendor(org: &str) -> &str {
    match org {
        "deepseek-ai" => "deepseek",
        "zai-org" => "z-ai",
        "Qwen" => "qwen",
        other => other,
    }
}

/// The Bedrock id `name` of `org` runs as when there is no catalog to look it up in.
fn bedrock_rule(org: &str, name: &str) -> String {
    let vendor = bedrock_vendor(org);
    if is_closed(org) {
        format!("global.{vendor}.{name}")
    } else {
        format!("{vendor}.{}", name.to_lowercase())
    }
}

/// The OpenRouter id `name` of `org` runs as when there is no catalog to look it up in.
fn openrouter_rule(org: &str, name: &str) -> String {
    let name = name.to_lowercase();
    let name = match org {
        // OpenRouter dots Claude's versions: `claude-sonnet-5.5`.
        "anthropic" => catalog::normalize(&name),
        _ => name,
    };
    format!("{}/{name}", openrouter_vendor(org))
}

fn required(name: Option<&str>) -> anyhow::Result<&str> {
    name.context("this backend needs a model name, as in \"<org>/<name>\"")
}

/// Builds `name` of `org` to run through `backend`, as the id that backend knows it by:
///
/// - CLI: `name`, as `--model`.
/// - Own API: `name`, lowercased for open-weight models (`glm-4.6`). DeepSeek's names no
///   version, so it takes its own names (`deepseek-ai/deepseek-chat`), not Hugging Face's.
/// - Bedrock: looked up in the region's catalog (see [`catalog::BedrockCatalog::find`]),
///   else [`bedrock_rule`]: `claude-sonnet-5-5` runs `global.anthropic.claude-sonnet-5-5`,
///   and with the catalog `claude-sonnet-4-5` runs
///   `global.anthropic.claude-sonnet-4-5-20250929-v1:0`.
/// - OpenRouter: looked up in OpenRouter's catalog (see
///   [`catalog::OpenRouterCatalog::find`]), else [`openrouter_rule`]: `claude-sonnet-5-5`
///   runs `anthropic/claude-sonnet-5.5`.
/// - vLLM: `<org>/<name>`, the Hugging Face repo id it serves the model as.
///
/// The catalogs are fetched once per process; to run an id as is, build the model directly
/// (`BedrockApiModel::from_env("arn:…")`).
async fn build(
    backend: Backend,
    org: &str,
    name: Option<&str>,
) -> anyhow::Result<Arc<dyn InferLangModel>> {
    Ok(match (backend, org) {
        (Backend::Cli, "anthropic") => {
            let model = ClaudeCliModel::new();
            Arc::new(match name {
                Some(name) => model.with_model(name),
                None => model,
            })
        }
        (Backend::Cli, "openai") => {
            let model = CodexCliModel::new();
            Arc::new(match name {
                Some(name) => model.with_model(name),
                None => model,
            })
        }
        (Backend::Cli, "google") => {
            let model = GeminiCliModel::new();
            Arc::new(match name {
                Some(name) => model.with_model(name),
                None => model,
            })
        }
        (Backend::Cli, _) => bail!("'{org}' has no CLI"),

        (Backend::Api, "anthropic") => Arc::new(ClaudeApiModel::from_env(required(name)?)?),
        (Backend::Api, "openai") => Arc::new(GptApiModel::from_env(required(name)?)?),
        (Backend::Api, "google") => Arc::new(GeminiApiModel::from_env(required(name)?)?),
        (Backend::Api, "deepseek-ai") => {
            Arc::new(DeepSeekApiModel::from_env(required(name)?.to_lowercase())?)
        }
        (Backend::Api, "moonshotai") => {
            Arc::new(KimiApiModel::from_env(required(name)?.to_lowercase())?)
        }
        (Backend::Api, "zai-org") => {
            Arc::new(GlmApiModel::from_env(required(name)?.to_lowercase())?)
        }
        (Backend::Api, "Qwen") => Arc::new(QwenApiModel::from_env(required(name)?.to_lowercase())?),
        (Backend::Api, _) => bail!("'{org}' has no API of its own"),

        (Backend::Bedrock, _) => {
            let name = required(name)?;
            let (api_key, region) = env_key_and_region()?;
            let id = catalog::bedrock_catalog(&region, &api_key)
                .await
                .and_then(|c| c.find(bedrock_vendor(org), name))
                .unwrap_or_else(|| bedrock_rule(org, name));
            Arc::new(BedrockApiModel::new(id, api_key, region))
        }
        (Backend::OpenRouter, _) => {
            let name = required(name)?;
            let id = catalog::openrouter_catalog()
                .await
                .and_then(|c| c.find(org, openrouter_vendor(org), name))
                .unwrap_or_else(|| openrouter_rule(org, name));
            Arc::new(OpenRouterApiModel::from_env(id)?)
        }
        (Backend::Vllm, _) => Arc::new(VllmModel::from_env(format!("{org}/{}", required(name)?))?),
    })
}

/// Providers keyed by org, seeded to run each one through its CLI, or its API when it has
/// no CLI.
static LANG_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, LangModelProvider>>> =
    LazyLock::new(|| {
        RwLock::new(
            [
                ("anthropic", Backend::Cli),
                ("openai", Backend::Cli),
                ("google", Backend::Cli),
                ("deepseek-ai", Backend::Api),
                ("moonshotai", Backend::Api),
                ("zai-org", Backend::Api),
                ("Qwen", Backend::Api),
            ]
            .into_iter()
            .map(|(org, backend)| (org.to_owned(), backend.into()))
            .collect(),
        )
    });

/// Runs the models of `org` through `provider`, replacing the provider it had.
pub fn set_lang_model_provider(org: impl Into<String>, provider: impl Into<LangModelProvider>) {
    LANG_MODEL_PROVIDERS
        .write()
        .unwrap()
        .insert(org.into(), provider.into());
}

/// Removes the provider for `org`, returning it if there was one.
pub fn remove_lang_model_provider(org: &str) -> Option<LangModelProvider> {
    LANG_MODEL_PROVIDERS.write().unwrap().remove(org)
}

/// Builds a model from a desc of the form `<org>` or `<org>/<name>`, e.g. `"anthropic"` or
/// `"openai/gpt-6.1-sol"`, through the provider set for `<org>`. Claude, GPT and Gemini
/// are named by their official API id, open-weight models by their Hugging Face repo id.
pub async fn create_lang_model(desc: &str) -> anyhow::Result<Arc<dyn InferLangModel>> {
    let (org, name) = match desc.split_once('/') {
        Some((org, name)) => (org, Some(name)),
        None => (desc, None),
    };
    // Clone out so the lock is not held while building.
    let provider = LANG_MODEL_PROVIDERS
        .read()
        .unwrap()
        .get(org)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("no provider set for lang model org '{org}'"))?;
    match provider {
        LangModelProvider::Backend(backend) => build(backend, org, name).await,
        LangModelProvider::Custom(f) => f(org, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rules_without_a_catalog() {
        assert_eq!(
            bedrock_rule("anthropic", "claude-sonnet-5-5"),
            "global.anthropic.claude-sonnet-5-5"
        );
        assert_eq!(
            bedrock_rule("openai", "gpt-6.1-sol"),
            "global.openai.gpt-6.1-sol"
        );
        assert_eq!(
            bedrock_rule("moonshotai", "Kimi-K2-Thinking"),
            "moonshot.kimi-k2-thinking"
        );
        assert_eq!(
            openrouter_rule("anthropic", "claude-sonnet-5-5"),
            "anthropic/claude-sonnet-5.5"
        );
        assert_eq!(
            openrouter_rule("openai", "gpt-6.1-sol"),
            "openai/gpt-6.1-sol"
        );
        assert_eq!(openrouter_rule("zai-org", "GLM-4.6"), "z-ai/glm-4.6");
    }

    #[tokio::test]
    async fn test_unknown_org_and_missing_cli() {
        let err = create_lang_model("nobody/model").await.err().unwrap();
        assert!(err.to_string().contains("nobody"), "{err}");

        let err = build(Backend::Cli, "zai-org", Some("GLM-4.6"))
            .await
            .err()
            .unwrap();
        assert!(err.to_string().contains("no CLI"), "{err}");
    }
}
