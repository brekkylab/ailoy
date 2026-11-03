//! Builds language models from a desc, choosing how each one runs.
//!
//! A desc is `<org>/<name>`, or `<org>` alone for a CLI's default model. The provider set
//! for the org decides how it runs: its CLI, its own API, Bedrock, OpenRouter, vLLM or any
//! other OpenAI-compatible server.
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
//! // Or on a server of your own, one per org: Ollama serves `llama3.3:70b` under the org
//! // `ollama`, named after the server.
//! set_lang_model_provider("ollama", LangModelServer::new(Backend::OpenAiCompatible, "http://localhost:11434/v1")?);
//! let model = create_lang_model("ollama/llama3.3:70b").await?;
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
//! | `deepseek-ai` | —                  | [`DeepSeekApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`] |
//! | `moonshotai`  | —                  | [`KimiApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]     |
//! | `zai-org`     | —                  | [`GlmApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]      |
//! | `Qwen`        | —                  | [`QwenApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]     |
//! | any other     | —                  | [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]                       |
//!
//! `cli` and `api` know the orgs in the table; `bedrock`, `openrouter`, `vllm` and
//! `openai-compatible` build a model for any org. The default is the CLI where there is
//! one, otherwise the org's own API. The own APIs read their key from `ANTHROPIC_API_KEY`,
//! `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `ZAI_API_KEY` and
//! `DASHSCOPE_API_KEY`; Bedrock from `AWS_BEARER_TOKEN_BEDROCK` (and `AWS_REGION`); OpenRouter
//! from `OPENROUTER_API_KEY`; vLLM from the server at `VLLM_BASE_URL` (by default
//! `http://localhost:8000/v1`), with `VLLM_API_KEY` if it is set; any other OpenAI-compatible
//! server from `OPENAI_COMPATIBLE_BASE_URL`, with `OPENAI_COMPATIBLE_API_KEY` if it is set. A
//! [`LangModelServer`] names the server instead, one per org. All but the CLIs need a name in
//! the desc.

mod catalog;

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, RwLock},
};

use anyhow::{Context as _, bail};

use super::{
    BedrockApiModel, ClaudeApiModel, ClaudeCliModel, CodexCliModel, DeepSeekApiModel,
    GeminiApiModel, GeminiCliModel, GlmApiModel, GptApiModel, InferLangModel, KimiApiModel,
    OpenAiCompatibleModel, OpenRouterApiModel, QwenApiModel, r#impl::env_key_and_region,
};

/// How a model is run, spelled `"cli"`, `"api"`, `"bedrock"`, `"openrouter"`, `"vllm"` or
/// `"openai-compatible"` in its serde form.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, strum::Display,
)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
pub enum Backend {
    /// The org's CLI: Claude Code, Codex or Gemini CLI.
    Cli,
    /// The org's own API.
    Api,
    Bedrock,
    OpenRouter,
    /// A vLLM server, which serves a model as its Hugging Face repo id: the desc as it is.
    Vllm,
    /// Any server speaking the OpenAI-compatible Chat Completions API (llama-server, Ollama,
    /// LM Studio, SGLang, vLLM too). The org is your name for the server, and the name after
    /// it is the id the server serves the model as: `ollama/llama3.3:70b`.
    #[serde(rename = "openai-compatible")]
    #[strum(serialize = "openai-compatible")]
    OpenAiCompatible,
}

/// Builds a model of your own from the desc's org and the name after it, `None` when the
/// desc names only the org.
pub type CustomLangModelProvider =
    Arc<dyn Fn(&str, Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> + Send + Sync>;

/// How the models of an org are built.
#[derive(Clone)]
pub enum LangModelProvider {
    /// A backend, on the server or key the environment names.
    Backend(Backend),
    /// A backend on a server of your own.
    Server(LangModelServer),
    Custom(CustomLangModelProvider),
}

/// [`Backend::Vllm`] or [`Backend::OpenAiCompatible`] on a server you name, instead of the
/// one in the environment, so each org can run on its own server. In serde form,
/// `{"backend": "openai-compatible", "base_url": "http://localhost:11434/v1"}`, with an
/// `api_key` if the server wants one.
#[derive(Clone, serde::Deserialize)]
#[serde(try_from = "LangModelServerFields")]
pub struct LangModelServer {
    backend: Backend,
    base_url: String,
    api_key: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LangModelServerFields {
    backend: Backend,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
}

impl TryFrom<LangModelServerFields> for LangModelServer {
    type Error = anyhow::Error;

    fn try_from(fields: LangModelServerFields) -> anyhow::Result<Self> {
        let server = Self::new(fields.backend, fields.base_url)?;
        Ok(match fields.api_key {
            Some(api_key) => server.with_api_key(api_key),
            None => server,
        })
    }
}

// Hand-written so the key stays out of logs.
impl std::fmt::Debug for LangModelServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangModelServer")
            .field("backend", &self.backend)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl LangModelServer {
    /// `backend`, [`Backend::Vllm`] or [`Backend::OpenAiCompatible`], on the server at
    /// `base_url` (`"http://localhost:8000/v1"`); any other backend, or a blank URL, is an
    /// error.
    pub fn new(backend: Backend, base_url: impl Into<String>) -> anyhow::Result<Self> {
        Self::check(backend)?;
        let base_url = base_url.into();
        if base_url.trim().is_empty() {
            bail!("a server needs a base_url");
        }
        Ok(Self {
            backend,
            base_url,
            api_key: None,
        })
    }

    fn check(backend: Backend) -> anyhow::Result<()> {
        if matches!(backend, Backend::Vllm | Backend::OpenAiCompatible) {
            Ok(())
        } else {
            bail!("a server runs the 'vllm' or 'openai-compatible' backend, not '{backend}'")
        }
    }

    /// `backend` on the server the environment names: `VLLM_BASE_URL` (by default
    /// `http://localhost:8000/v1`) and `VLLM_API_KEY` for [`Backend::Vllm`],
    /// `OPENAI_COMPATIBLE_BASE_URL` and `OPENAI_COMPATIBLE_API_KEY` for
    /// [`Backend::OpenAiCompatible`]. A blank variable is unset.
    pub fn from_env(backend: Backend) -> anyhow::Result<Self> {
        Self::check(backend)?;
        let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
        let (base_url, api_key) = if backend == Backend::Vllm {
            (
                var("VLLM_BASE_URL").unwrap_or_else(|| "http://localhost:8000/v1".to_owned()),
                var("VLLM_API_KEY"),
            )
        } else {
            (
                var("OPENAI_COMPATIBLE_BASE_URL")
                    .context("OPENAI_COMPATIBLE_BASE_URL is not set")?,
                var("OPENAI_COMPATIBLE_API_KEY"),
            )
        };
        let server = Self::new(backend, base_url)?;
        Ok(match api_key {
            Some(api_key) => server.with_api_key(api_key),
            None => server,
        })
    }

    /// Sends `api_key` as a bearer token.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The id this server knows `name` of `org` by: `<org>/<name>` on vLLM, which serves
    /// Hugging Face repo ids, and `<name>` as it is on any other server, whose org names the
    /// server.
    fn model_id(&self, org: &str, name: Option<&str>) -> anyhow::Result<String> {
        let name = required(name)?;
        Ok(match self.backend {
            Backend::Vllm => format!("{org}/{name}"),
            _ => name.to_owned(),
        })
    }

    /// Builds `name` of `org` on this server, as [`model_id`](Self::model_id) names it.
    fn build(&self, org: &str, name: Option<&str>) -> anyhow::Result<Arc<dyn InferLangModel>> {
        let model = OpenAiCompatibleModel::new(self.model_id(org, name)?, &self.base_url);
        Ok(Arc::new(match &self.api_key {
            Some(api_key) => model.with_api_key(api_key),
            None => model,
        }))
    }
}

impl From<LangModelServer> for LangModelProvider {
    fn from(server: LangModelServer) -> Self {
        Self::Server(server)
    }
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
/// - vLLM and any other OpenAI-compatible server: as [`LangModelServer::build`] says, on the
///   server [`LangModelServer::from_env`] reads.
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
        (Backend::Vllm | Backend::OpenAiCompatible, _) => {
            LangModelServer::from_env(backend)?.build(org, name)?
        }
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

/// The provider a desc's model is built through, the one set for its org, if there is one.
/// The desc is as [`create_lang_model`] takes it: `<org>` or `<org>/<name>`.
pub fn get_lang_model_provider(desc: &str) -> Option<LangModelProvider> {
    let org = desc.split_once('/').map_or(desc, |(org, _)| org);
    LANG_MODEL_PROVIDERS.read().unwrap().get(org).cloned()
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
        LangModelProvider::Server(server) => server.build(org, name),
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

    #[test]
    fn test_a_provider_reads_back_as_it_was_set() {
        let is = |org| match get_lang_model_provider(org) {
            Some(LangModelProvider::Backend(backend)) => Some(backend),
            _ => None,
        };
        assert_eq!(is("anthropic"), Some(Backend::Cli));
        assert_eq!(is("anthropic/claude-sonnet-5-5"), Some(Backend::Cli));
        for backend in [
            Backend::Cli,
            Backend::Api,
            Backend::Bedrock,
            Backend::OpenRouter,
            Backend::Vllm,
            Backend::OpenAiCompatible,
        ] {
            assert_eq!(serde_json::to_value(backend).unwrap(), backend.to_string());
        }
        assert_eq!(Backend::OpenAiCompatible.to_string(), "openai-compatible");
        set_lang_model_provider("provider-read-back", Backend::Bedrock);
        assert_eq!(is("provider-read-back/some-model"), Some(Backend::Bedrock));
        remove_lang_model_provider("provider-read-back");
        assert!(get_lang_model_provider("provider-read-back").is_none());
    }

    #[test]
    fn test_a_server_runs_the_two_server_backends() {
        for backend in [Backend::Vllm, Backend::OpenAiCompatible] {
            let server = LangModelServer::new(backend, "http://localhost:1/v1").unwrap();
            assert_eq!(server.backend(), backend);
            assert_eq!(server.base_url(), "http://localhost:1/v1");
        }
        for backend in [
            Backend::Cli,
            Backend::Api,
            Backend::Bedrock,
            Backend::OpenRouter,
        ] {
            let err = LangModelServer::new(backend, "http://localhost:1/v1")
                .err()
                .unwrap();
            assert!(err.to_string().contains(&backend.to_string()), "{err}");
        }

        let server: LangModelServer = serde_json::from_value(serde_json::json!({
            "backend": "openai-compatible",
            "base_url": "http://localhost:11434/v1",
            "api_key": "k",
        }))
        .unwrap();
        assert_eq!(server.backend(), Backend::OpenAiCompatible);
        assert_eq!(server.api_key.as_deref(), Some("k"));
        for fields in [
            serde_json::json!({"backend": "cli", "base_url": "http://localhost:1/v1"}),
            serde_json::json!({"backend": "vllm"}),
            serde_json::json!({"backend": "vllm", "base_url": " "}),
            serde_json::json!({"backend": "vllm", "base_url": "http://localhost:1/v1", "key": "k"}),
        ] {
            assert!(
                serde_json::from_value::<LangModelServer>(fields.clone()).is_err(),
                "{fields}"
            );
        }
    }

    #[test]
    fn test_a_server_names_the_model_as_the_server_does() {
        let id = |backend, org, name| {
            LangModelServer::new(backend, "http://localhost:1/v1")
                .unwrap()
                .model_id(org, name)
                .unwrap()
        };
        // vLLM serves Hugging Face repo ids, so the desc is the id.
        assert_eq!(
            id(Backend::Vllm, "zai-org", Some("GLM-4.6")),
            "zai-org/GLM-4.6"
        );
        // On any other server the org names the server, and the name is the server's id,
        // slashes and all.
        assert_eq!(
            id(Backend::OpenAiCompatible, "ollama", Some("llama3.3:70b")),
            "llama3.3:70b"
        );
        assert_eq!(
            id(Backend::OpenAiCompatible, "lmstudio", Some("qwen/qwen3-8b")),
            "qwen/qwen3-8b"
        );
        let err = LangModelServer::new(Backend::OpenAiCompatible, "http://localhost:1/v1")
            .unwrap()
            .build("ollama", None)
            .err()
            .unwrap();
        assert!(err.to_string().contains("needs a model name"), "{err}");

        set_lang_model_provider(
            "server-read-back",
            LangModelServer::new(Backend::OpenAiCompatible, "http://localhost:1/v1").unwrap(),
        );
        assert!(matches!(
            get_lang_model_provider("server-read-back/x"),
            Some(LangModelProvider::Server(server)) if server.backend() == Backend::OpenAiCompatible
        ));
        remove_lang_model_provider("server-read-back");
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
