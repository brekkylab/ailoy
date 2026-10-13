//! Finds the id Bedrock and OpenRouter serve a model as, by matching a desc's name against
//! the catalogs they list.
//!
//! Each catalog is fetched once per process (per region, for Bedrock) and kept, a failed
//! fetch included, so building a model costs at most one round of requests.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
};

use anyhow::Context as _;
use serde::Deserialize;
use tokio::sync::Mutex;

fn is_num(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `name` in the form the catalogs are compared in: lowercase, spaces as `-`, without a
/// trailing date (`-20250929`), and with each pair of short numbers as one version
/// (`4-5` → `4.5`). So `claude-sonnet-4-5`, `claude-sonnet-4-5-20250929`,
/// `claude-sonnet-4.5` and `Claude Sonnet 4.5` all become `claude-sonnet-4.5`.
pub(super) fn normalize(name: &str) -> String {
    let mut name = name.to_lowercase().replace(' ', "-");
    if let Some((rest, date)) = name.rsplit_once('-')
        && date.len() == 8
        && is_num(date)
    {
        name.truncate(rest.len());
    }
    dot_versions(&name)
}

/// A Bedrock model id without its vendor (`anthropic.`) and version (`-v1`, `-v1:0`):
/// `anthropic.claude-sonnet-4-5-20250929-v1:0` → `claude-sonnet-4-5-20250929`. Only
/// Bedrock ids get the version cut, so a name such as `DeepSeek-V3` keeps its `-V3`.
fn bedrock_name<'a>(model_id: &'a str, prefix: &str) -> &'a str {
    let mut name = &model_id[prefix.len()..];
    if let Some((rest, n)) = name.rsplit_once(':')
        && is_num(n)
    {
        name = rest;
    }
    if let Some((rest, v)) = name.rsplit_once("-v")
        && is_num(v)
    {
        name = rest;
    }
    name
}

/// Every `N-N` of one- or two-digit numbers as `N.N`: `claude-3-5-sonnet` →
/// `claude-3.5-sonnet`, `claude-sonnet-5-5` → `claude-sonnet-5.5`.
pub(super) fn dot_versions(name: &str) -> String {
    let short_num = |t: &str| (1..=2).contains(&t.len()) && t.bytes().all(|b| b.is_ascii_digit());
    let tokens: Vec<&str> = name.split('-').collect();
    let mut out = String::with_capacity(name.len());
    let mut i = 0;
    while i < tokens.len() {
        if i > 0 {
            out.push('-');
        }
        out.push_str(tokens[i]);
        if i + 1 < tokens.len() && short_num(tokens[i]) && short_num(tokens[i + 1]) {
            out.push('.');
            out.push_str(tokens[i + 1]);
            i += 2;
        } else {
            i += 1;
        }
    }
    out
}

/// Bounds each catalog request as a whole, so a stalled one does not hold up building.
static CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("the HTTP client builds")
});

async fn get_json<T: serde::de::DeserializeOwned>(
    url: &str,
    api_key: Option<&str>,
) -> anyhow::Result<T> {
    let mut request = CLIENT.get(url);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await?;
    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        anyhow::bail!("{url} returned {status}: {text}");
    }
    response
        .json()
        .await
        .with_context(|| format!("reading {url}"))
}

/// The models and system inference profiles Bedrock lists in one region.
#[derive(Debug, Default, Deserialize)]
pub(super) struct BedrockCatalog {
    #[serde(default)]
    models: Vec<BedrockModelSummary>,
    #[serde(default)]
    profiles: Vec<BedrockProfile>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockModelSummary {
    model_id: String,
    #[serde(default)]
    model_name: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockProfile {
    inference_profile_id: String,
    #[serde(default)]
    models: Vec<BedrockProfileModel>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockProfileModel {
    model_arn: String,
}

impl BedrockCatalog {
    /// The id to run `name`, made by `vendor` (as Bedrock names it, e.g. `moonshot`), as:
    /// its global inference profile, else another profile this region can use, else its
    /// model id. The model is found by its id, or else by its display name, since Bedrock
    /// spells some ids its own way (`deepseek.v3-v1:0`).
    pub(super) fn find(&self, vendor: &str, name: &str) -> Option<String> {
        let want = normalize(name);
        let prefix = format!("{vendor}.");
        let vendors = || {
            self.models
                .iter()
                .filter(|m| m.model_id.starts_with(&prefix))
        };
        let model = vendors()
            .find(|m| normalize(bedrock_name(&m.model_id, &prefix)) == want)
            .or_else(|| vendors().find(|m| normalize(&m.model_name) == want))?;

        let arn_suffix = format!("/{}", model.model_id);
        let profiles: Vec<&str> = self
            .profiles
            .iter()
            .filter(|p| p.models.iter().any(|m| m.model_arn.ends_with(&arn_suffix)))
            .map(|p| p.inference_profile_id.as_str())
            .collect();
        let id = profiles
            .iter()
            .find(|p| p.starts_with("global."))
            .or(profiles.first())
            .map_or(model.model_id.as_str(), |p| p);
        Some(id.to_owned())
    }

    async fn fetch(region: &str, api_key: &str) -> anyhow::Result<Self> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Models {
            model_summaries: Vec<BedrockModelSummary>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Profiles {
            #[serde(default)]
            inference_profile_summaries: Vec<BedrockProfile>,
            next_token: Option<String>,
        }

        let base = format!("https://bedrock.{region}.amazonaws.com");
        let models: Models = get_json(&format!("{base}/foundation-models"), Some(api_key)).await?;

        let mut profiles = Vec::new();
        let mut next_token: Option<String> = None;
        loop {
            let mut url =
                format!("{base}/inference-profiles?typeEquals=SYSTEM_DEFINED&maxResults=1000");
            if let Some(token) = &next_token {
                url.push_str("&nextToken=");
                url.push_str(&urlencoding::encode(token));
            }
            let page: Profiles = get_json(&url, Some(api_key)).await?;
            profiles.extend(page.inference_profile_summaries);
            next_token = page.next_token;
            if next_token.is_none() {
                break;
            }
        }
        Ok(Self {
            models: models.model_summaries,
            profiles,
        })
    }
}

/// The models OpenRouter lists.
#[derive(Debug, Default, Deserialize)]
pub(super) struct OpenRouterCatalog {
    #[serde(default)]
    data: Vec<OpenRouterEntry>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterEntry {
    id: String,
    #[serde(default)]
    canonical_slug: Option<String>,
    #[serde(default)]
    hugging_face_id: Option<String>,
}

impl OpenRouterCatalog {
    /// The id to run `<org>/<name>` as, made by `vendor` (as OpenRouter names it, e.g.
    /// `z-ai`): the model whose Hugging Face repo is `<org>/<name>`, else the one whose
    /// slug under `vendor/` is `name`. Variants (`:free`, …) and `~` aliases are skipped.
    pub(super) fn find(&self, org: &str, vendor: &str, name: &str) -> Option<String> {
        let key = format!("{org}/{name}");
        let plain = || {
            self.data
                .iter()
                .filter(|e| !e.id.contains(':') && !e.id.starts_with('~'))
        };
        if let Some(e) = plain().find(|e| {
            e.hugging_face_id
                .as_deref()
                .is_some_and(|hf| hf.eq_ignore_ascii_case(&key))
        }) {
            return Some(e.id.clone());
        }

        let prefix = format!("{vendor}/");
        let want = normalize(name);
        let slug_matches = |id: &str| {
            id.strip_prefix(&prefix)
                .is_some_and(|slug| normalize(slug) == want)
        };
        plain()
            .find(|e| slug_matches(&e.id))
            .or_else(|| plain().find(|e| e.canonical_slug.as_deref().is_some_and(slug_matches)))
            .map(|e| e.id.clone())
    }

    async fn fetch() -> anyhow::Result<Self> {
        get_json("https://openrouter.ai/api/v1/models", None).await
    }
}

/// Bedrock's catalog for `region`, or `None` if it could not be fetched.
pub(super) async fn bedrock_catalog(region: &str, api_key: &str) -> Option<Arc<BedrockCatalog>> {
    static CACHE: LazyLock<Mutex<HashMap<String, Option<Arc<BedrockCatalog>>>>> =
        LazyLock::new(Default::default);
    // Held across the fetch so concurrent builds share one.
    let mut cache = CACHE.lock().await;
    if let Some(catalog) = cache.get(region) {
        return catalog.clone();
    }
    let catalog = BedrockCatalog::fetch(region, api_key)
        .await
        .inspect_err(|e| {
            log::warn!("listing Bedrock models in {region} failed, ids are built by rule: {e:#}")
        })
        .ok()
        .map(Arc::new);
    cache.insert(region.to_owned(), catalog.clone());
    catalog
}

/// OpenRouter's catalog, or `None` if it could not be fetched.
pub(super) async fn openrouter_catalog() -> Option<Arc<OpenRouterCatalog>> {
    static CACHE: Mutex<Option<Option<Arc<OpenRouterCatalog>>>> = Mutex::const_new(None);
    let mut cache = CACHE.lock().await;
    if let Some(catalog) = &*cache {
        return catalog.clone();
    }
    let catalog = OpenRouterCatalog::fetch()
        .await
        .inspect_err(|e| {
            log::warn!("listing OpenRouter models failed, ids are built by rule: {e:#}")
        })
        .ok()
        .map(Arc::new);
    *cache = Some(catalog.clone());
    catalog
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_spellings_agree() {
        for name in [
            "claude-sonnet-4-5",
            "claude-sonnet-4-5-20250929",
            "claude-sonnet-4.5",
            "Claude Sonnet 4.5",
        ] {
            assert_eq!(normalize(name), "claude-sonnet-4.5", "{name}");
        }
        assert_eq!(normalize("claude-3-5-sonnet-20241022"), "claude-3.5-sonnet");
        assert_eq!(normalize("gpt-6.1-sol"), "gpt-6.1-sol");
        assert_eq!(normalize("DeepSeek V3.2"), "deepseek-v3.2");
        assert_eq!(normalize("DeepSeek-V3"), "deepseek-v3");
        // Four digits are no version.
        assert_eq!(normalize("Kimi-K2-Instruct-0905"), "kimi-k2-instruct-0905");
    }

    #[test]
    fn test_bedrock_name_cuts_the_version() {
        let name = |id| bedrock_name(id, "anthropic.");
        assert_eq!(
            name("anthropic.claude-sonnet-4-5-20250929-v1:0"),
            "claude-sonnet-4-5-20250929"
        );
        assert_eq!(name("anthropic.claude-opus-4-6-v1"), "claude-opus-4-6");
        assert_eq!(name("anthropic.claude-sonnet-5-5"), "claude-sonnet-5-5");
    }

    fn bedrock() -> BedrockCatalog {
        serde_json::from_value(serde_json::json!({
            "models": [
                {"modelId": "anthropic.claude-sonnet-5-5", "modelName": "Claude Sonnet 5.5"},
                {"modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0", "modelName": "Claude Sonnet 4.5"},
                {"modelId": "openai.gpt-6.1-sol", "modelName": "GPT-6.1 Sol"},
                {"modelId": "deepseek.v3-2-v1:0", "modelName": "DeepSeek V3.2"},
                {"modelId": "qwen.qwen3-32b-v1:0", "modelName": "Qwen3 32B"},
            ],
            "profiles": [
                {"inferenceProfileId": "us.anthropic.claude-sonnet-5-5", "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5-5"}]},
                {"inferenceProfileId": "global.anthropic.claude-sonnet-5-5", "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-5-5"}]},
                {"inferenceProfileId": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0"}]},
                {"inferenceProfileId": "us.openai.gpt-6.1-sol", "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6.1-sol"}]},
            ],
        }))
        .unwrap()
    }

    #[test]
    fn test_bedrock_prefers_the_global_profile() {
        let c = bedrock();
        assert_eq!(
            c.find("anthropic", "claude-sonnet-5-5").as_deref(),
            Some("global.anthropic.claude-sonnet-5-5")
        );
        assert_eq!(
            c.find("anthropic", "claude-sonnet-4-5").as_deref(),
            Some("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        );
        assert_eq!(
            c.find("openai", "gpt-6.1-sol").as_deref(),
            Some("us.openai.gpt-6.1-sol")
        );
    }

    #[test]
    fn test_bedrock_matches_by_display_name_and_falls_back_to_the_model_id() {
        let c = bedrock();
        assert_eq!(
            c.find("deepseek", "DeepSeek-V3.2").as_deref(),
            Some("deepseek.v3-2-v1:0")
        );
        assert_eq!(
            c.find("qwen", "Qwen3-32B").as_deref(),
            Some("qwen.qwen3-32b-v1:0")
        );
        assert_eq!(c.find("anthropic", "claude-unknown-9"), None);
        // The vendor must match too.
        assert_eq!(c.find("openai", "claude-sonnet-5-5"), None);
    }

    fn openrouter() -> OpenRouterCatalog {
        serde_json::from_value(serde_json::json!({"data": [
            {"id": "anthropic/claude-sonnet-5.5", "canonical_slug": "anthropic/claude-sonnet-5.5-20260928"},
            {"id": "~anthropic/claude-sonnet-latest"},
            {"id": "openai/gpt-6.1-sol"},
            {"id": "openai/gpt-oss-120b:free", "hugging_face_id": "openai/gpt-oss-120b"},
            {"id": "openai/gpt-oss-120b", "hugging_face_id": "openai/gpt-oss-120b"},
            {"id": "z-ai/glm-4.6", "hugging_face_id": "zai-org/GLM-4.6"},
            {"id": "qwen/qwen3-235b-a22b-2507", "hugging_face_id": "Qwen/Qwen3-235B-A22B-Instruct-2507"},
        ]}))
        .unwrap()
    }

    #[test]
    fn test_openrouter_matches_by_hugging_face_id_then_slug() {
        let c = openrouter();
        assert_eq!(
            c.find("Qwen", "qwen", "Qwen3-235B-A22B-Instruct-2507")
                .as_deref(),
            Some("qwen/qwen3-235b-a22b-2507")
        );
        assert_eq!(
            c.find("zai-org", "z-ai", "GLM-4.6").as_deref(),
            Some("z-ai/glm-4.6")
        );
        assert_eq!(
            c.find("openai", "openai", "gpt-oss-120b").as_deref(),
            Some("openai/gpt-oss-120b")
        );
        assert_eq!(
            c.find("anthropic", "anthropic", "claude-sonnet-5-5")
                .as_deref(),
            Some("anthropic/claude-sonnet-5.5")
        );
        assert_eq!(
            c.find("openai", "openai", "gpt-6.1-sol").as_deref(),
            Some("openai/gpt-6.1-sol")
        );
        assert_eq!(c.find("anthropic", "anthropic", "claude-unknown-9"), None);
    }
}
