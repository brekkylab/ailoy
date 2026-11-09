//! Builds language models from a desc, choosing how each one runs.
//!
//! A desc is `<org>/<name>`, or `<org>` alone for a CLI's default model. The provider set
//! for the org decides how it runs: its CLI, its own API, Bedrock, OpenRouter or an
//! OpenAI-compatible server.
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
//! // Or on a server of your own, one per org: here Ollama, which serves Llama 3.3 as
//! // `llama3.3:70b`.
//! set_lang_model_provider("meta-llama", LangModelServer::new("http://localhost:11434/v1")?.with_model("Llama-3.3-70B-Instruct", "llama3.3:70b"));
//! let model = create_lang_model("meta-llama/Llama-3.3-70B-Instruct").await?;
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
//! | EXAONE's API         | `LGAI-EXAONE/<official API id>` | `LGAI-EXAONE/k-exaone_v2.5`          |
//!
//! `<org>` alone runs the CLI's default model.
//!
//! # Backends
//!
//! | Org           | CLI                | APIs                                                                                         |
//! |---------------|--------------------|----------------------------------------------------------------------------------------------|
//! | `anthropic`   | [`ClaudeCliModel`] | [`ClaudeApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]   |
//! | `openai`      | [`CodexCliModel`]  | [`GptApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]      |
//! | `google`      | [`GeminiCliModel`] | [`GeminiApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]                        |
//! | `deepseek-ai` | —                  | [`DeepSeekApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`] |
//! | `moonshotai`  | —                  | [`KimiApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]     |
//! | `zai-org`     | —                  | [`GlmApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]      |
//! | `Qwen`        | —                  | [`QwenApiModel`], [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]     |
//! | `LGAI-EXAONE` | —                  | [`OpenAiCompatibleModel`], [`BedrockApiModel`], [`OpenRouterApiModel`]                       |
//! | any other     | —                  | [`BedrockApiModel`], [`OpenRouterApiModel`], [`OpenAiCompatibleModel`]                       |
//!
//! `cli` and `api` know the orgs in the table; `bedrock`, `openrouter` and
//! `openai-compatible` build a model for any org. The default is the CLI where there is
//! one, otherwise the org's own API. The own APIs read their key from `ANTHROPIC_API_KEY`,
//! `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `KIMI_API_KEY`, `ZAI_API_KEY`,
//! `DASHSCOPE_API_KEY` and `EXAONE_API_KEY`; Bedrock from `AWS_BEARER_TOKEN_BEDROCK` (and
//! `AWS_REGION`); OpenRouter from `OPENROUTER_API_KEY`; an OpenAI-compatible server from
//! `OPENAI_COMPATIBLE_BASE_URL`, with `OPENAI_COMPATIBLE_API_KEY` if it is set. A
//! [`LangModelServer`] names the server instead, one per org. EXAONE's API speaks the
//! OpenAI-compatible Chat Completions API too, so its models are [`OpenAiCompatibleModel`]s
//! as well. All but the CLIs need a name in the desc.

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

/// How a model is run, spelled `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or
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
    /// A server speaking the OpenAI-compatible Chat Completions API: vLLM, llama-server,
    /// Ollama, LM Studio or SGLang. The desc stays the model's name; [`LangModelServer`]
    /// finds the id the server serves it as.
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

/// [`Backend::OpenAiCompatible`] on a server you name, instead of the one in the environment,
/// so each org can run on its own server. In serde form, `{"backend": "openai-compatible",
/// "base_url": "http://localhost:11434/v1"}`, with an `api_key` if the server wants one and
/// `models`, a map from a name to the id the server serves it as.
///
/// A desc names the model as its maker does, whatever the server calls it. The id sent is
/// the one `models` maps the name to; else, among the ids the server lists at `/models`, the
/// one equal to `<org>/<name>` or to `<name>` alone, ignoring case; else `<org>/<name>` as it
/// is, the Hugging Face repo id vLLM serves. So `Qwen/Qwen3-0.6B` runs as LM Studio's
/// `qwen/qwen3-0.6b` with nothing mapped, while Ollama's `llama3.3:70b` takes
/// `with_model("Llama-3.3-70B-Instruct", "llama3.3:70b")`.
#[derive(Clone, serde::Deserialize)]
#[serde(try_from = "LangModelServerFields")]
pub struct LangModelServer {
    base_url: String,
    api_key: Option<String>,
    models: HashMap<String, String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LangModelServerFields {
    backend: Backend,
    base_url: String,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    models: HashMap<String, String>,
}

impl TryFrom<LangModelServerFields> for LangModelServer {
    type Error = anyhow::Error;

    fn try_from(fields: LangModelServerFields) -> anyhow::Result<Self> {
        if fields.backend != Backend::OpenAiCompatible {
            bail!(
                "a server runs the 'openai-compatible' backend, not '{}'",
                fields.backend
            );
        }
        let mut server = Self::new(fields.base_url)?;
        server.api_key = fields.api_key;
        server.models = fields.models;
        Ok(server)
    }
}

// Hand-written so the key stays out of logs.
impl std::fmt::Debug for LangModelServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangModelServer")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("models", &self.models)
            .finish()
    }
}

impl LangModelServer {
    /// The server at `base_url` (`"http://localhost:8000/v1"`); a blank URL is an error.
    pub fn new(base_url: impl Into<String>) -> anyhow::Result<Self> {
        let base_url = base_url.into();
        if base_url.trim().is_empty() {
            bail!("a server needs a base_url");
        }
        Ok(Self {
            base_url,
            api_key: None,
            models: HashMap::new(),
        })
    }

    /// The server the environment names: `OPENAI_COMPATIBLE_BASE_URL`, with
    /// `OPENAI_COMPATIBLE_API_KEY` if it is set. A blank variable is unset.
    pub fn from_env() -> anyhow::Result<Self> {
        let mut server = Self::new(
            env_var("OPENAI_COMPATIBLE_BASE_URL")
                .context("OPENAI_COMPATIBLE_BASE_URL is not set")?,
        )?;
        server.api_key = env_var("OPENAI_COMPATIBLE_API_KEY");
        Ok(server)
    }

    /// Sends `api_key` as a bearer token.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Runs `name`, the part of a desc after the org, as `id`, the id this server serves it
    /// as.
    pub fn with_model(mut self, name: impl Into<String>, id: impl Into<String>) -> Self {
        self.models.insert(name.into(), id.into());
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The id this server serves `name` of `org` as, found as the type's doc says. A name
    /// `models` maps is not looked up on the server.
    async fn model_id(&self, org: &str, name: &str) -> String {
        if let Some(id) = self.models.get(name) {
            return id.clone();
        }
        let listed = catalog::server_models(&self.base_url, self.api_key.as_deref()).await;
        listed_id(&listed, org, name)
    }

    /// Builds `name` of `org` on this server, as [`model_id`](Self::model_id) names it.
    async fn build(
        &self,
        org: &str,
        name: Option<&str>,
    ) -> anyhow::Result<Arc<dyn InferLangModel>> {
        let id = self.model_id(org, required(name)?).await;
        let model = OpenAiCompatibleModel::new(id, &self.base_url);
        Ok(Arc::new(match &self.api_key {
            Some(api_key) => model.with_api_key(api_key),
            None => model,
        }))
    }
}

/// The id a server listing `ids` serves `name` of `org` as: the one equal to `<org>/<name>`,
/// else the one equal to `name` alone, both ignoring case; else `<org>/<name>` as it is.
fn listed_id(ids: &[String], org: &str, name: &str) -> String {
    let desc = format!("{org}/{name}");
    ids.iter()
        .find(|id| id.eq_ignore_ascii_case(&desc))
        .or_else(|| ids.iter().find(|id| id.eq_ignore_ascii_case(name)))
        .cloned()
        .unwrap_or(desc)
}

/// The environment variable `name`, `None` when it is unset or blank.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// EXAONE's API, which serves the OpenAI-compatible Chat Completions API.
const EXAONE_API_URL: &str = "https://api.exaone.ai/v1";

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
/// open-weight, named by their Hugging Face repo id, or by their API id where there is none.
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
///   EXAONE's is an [`OpenAiCompatibleModel`] on [`EXAONE_API_URL`], sent `name` as it is.
/// - Bedrock: looked up in the region's catalog (see [`catalog::BedrockCatalog::find`]),
///   else [`bedrock_rule`]: `claude-sonnet-5-5` runs `global.anthropic.claude-sonnet-5-5`,
///   and with the catalog `claude-sonnet-4-5` runs
///   `global.anthropic.claude-sonnet-4-5-20250929-v1:0`.
/// - OpenRouter: looked up in OpenRouter's catalog (see
///   [`catalog::OpenRouterCatalog::find`]), else [`openrouter_rule`]: `claude-sonnet-5-5`
///   runs `anthropic/claude-sonnet-5.5`.
/// - An OpenAI-compatible server: as [`LangModelServer`] says, on the server
///   [`LangModelServer::from_env`] reads.
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
        (Backend::Api, "LGAI-EXAONE") => {
            let name = required(name)?;
            let api_key = env_var("EXAONE_API_KEY").context("EXAONE_API_KEY is not set")?;
            Arc::new(OpenAiCompatibleModel::new(name, EXAONE_API_URL).with_api_key(api_key))
        }
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
        (Backend::OpenAiCompatible, _) => LangModelServer::from_env()?.build(org, name).await?,
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
                ("LGAI-EXAONE", Backend::Api),
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
/// are named by their official API id, open-weight models by their Hugging Face repo id, and
/// a model with no repo, such as EXAONE's `k-exaone_v2.5`, by its API id.
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
        LangModelProvider::Server(server) => server.build(org, name).await,
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
        assert_eq!(is("LGAI-EXAONE/k-exaone_v2.5"), Some(Backend::Api));
        for backend in [
            Backend::Cli,
            Backend::Api,
            Backend::Bedrock,
            Backend::OpenRouter,
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
    fn test_a_server_is_the_openai_compatible_backend_on_a_url() {
        let server = LangModelServer::new("http://localhost:1/v1").unwrap();
        assert_eq!(server.base_url(), "http://localhost:1/v1");
        assert!(LangModelServer::new(" ").is_err());

        let server: LangModelServer = serde_json::from_value(serde_json::json!({
            "backend": "openai-compatible",
            "base_url": "http://localhost:11434/v1",
            "api_key": "k",
            "models": {"Llama-3.3-70B-Instruct": "llama3.3:70b"},
        }))
        .unwrap();
        assert_eq!(server.api_key.as_deref(), Some("k"));
        assert_eq!(
            server
                .models
                .get("Llama-3.3-70B-Instruct")
                .map(String::as_str),
            Some("llama3.3:70b")
        );
        let err = serde_json::from_value::<LangModelServer>(
            serde_json::json!({"backend": "cli", "base_url": "http://localhost:1/v1"}),
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("openai-compatible"), "{err}");
        for fields in [
            serde_json::json!({"backend": "vllm", "base_url": "http://localhost:1/v1"}),
            serde_json::json!({"backend": "openai-compatible"}),
            serde_json::json!({"backend": "openai-compatible", "base_url": " "}),
            serde_json::json!({"backend": "openai-compatible", "base_url": "http://localhost:1/v1", "key": "k"}),
            serde_json::json!({"backend": "openai-compatible", "base_url": "http://localhost:1/v1", "models": ["a"]}),
        ] {
            assert!(
                serde_json::from_value::<LangModelServer>(fields.clone()).is_err(),
                "{fields}"
            );
        }
    }

    #[test]
    fn test_a_listed_id_is_the_desc_or_the_name_ignoring_case() {
        let listed = |ids: &[&str], org, name| {
            let ids: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
            listed_id(&ids, org, name)
        };
        // vLLM lists Hugging Face repo ids: the desc as it is.
        assert_eq!(
            listed(&["zai-org/GLM-4.6"], "zai-org", "GLM-4.6"),
            "zai-org/GLM-4.6"
        );
        // LM Studio lists the repo id in lower case.
        assert_eq!(
            listed(&["qwen3.5-27b", "qwen/qwen3-8b"], "Qwen", "Qwen3-8B"),
            "qwen/qwen3-8b"
        );
        // A server that lists the name without its org.
        assert_eq!(
            listed(&["qwen3.5-27b"], "Qwen", "Qwen3.5-27B"),
            "qwen3.5-27b"
        );
        // The whole desc wins over the name alone.
        assert_eq!(
            listed(&["qwen3-8b", "qwen/qwen3-8b"], "Qwen", "Qwen3-8B"),
            "qwen/qwen3-8b"
        );
        // Nothing listed, or nothing like it: the desc as it is.
        assert_eq!(
            listed(&[], "meta-llama", "Llama-3.3-70B-Instruct"),
            "meta-llama/Llama-3.3-70B-Instruct"
        );
        assert_eq!(
            listed(&["llama3.3:70b"], "meta-llama", "Llama-3.3-70B-Instruct"),
            "meta-llama/Llama-3.3-70B-Instruct"
        );
    }

    #[tokio::test]
    async fn test_a_server_maps_a_name_and_asks_for_the_rest() {
        // Nothing listens on port 1, so the server lists nothing.
        let server = LangModelServer::new("http://127.0.0.1:1/v1")
            .unwrap()
            .with_model("Llama-3.3-70B-Instruct", "llama3.3:70b");
        assert_eq!(
            server
                .model_id("meta-llama", "Llama-3.3-70B-Instruct")
                .await,
            "llama3.3:70b"
        );
        assert_eq!(server.model_id("Qwen", "Qwen3-8B").await, "Qwen/Qwen3-8B");
        let err = server.build("meta-llama", None).await.err().unwrap();
        assert!(err.to_string().contains("needs a model name"), "{err}");

        set_lang_model_provider(
            "server-read-back",
            LangModelServer::new("http://localhost:1/v1").unwrap(),
        );
        assert!(matches!(
            get_lang_model_provider("server-read-back/x"),
            Some(LangModelProvider::Server(server)) if server.base_url() == "http://localhost:1/v1"
        ));
        remove_lang_model_provider("server-read-back");
    }

    #[tokio::test]
    async fn test_exaone_runs_on_its_own_api() {
        let err = build(Backend::Api, "LGAI-EXAONE", None)
            .await
            .err()
            .unwrap();
        assert!(err.to_string().contains("needs a model name"), "{err}");
        // Built when the key is set; otherwise the key is what is missing.
        if let Err(err) = build(Backend::Api, "LGAI-EXAONE", Some("k-exaone_v2.5")).await {
            assert!(env_var("EXAONE_API_KEY").is_none());
            assert!(err.to_string().contains("EXAONE_API_KEY"), "{err}");
        }
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
