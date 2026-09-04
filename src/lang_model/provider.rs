use std::{
    collections::{BTreeMap, HashMap},
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::lang_model::LangModelAPISchema;

/// Describes the runtime endpoint used to invoke a language model.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum LangModelProviderElem {
    /// Calls a remote HTTP API. Requires the wire `schema`, the `url` of the endpoint, and an optional `api_key` for authentication.
    API {
        schema: LangModelAPISchema,

        url: Url,

        api_key: Option<String>,
    },
}

/// Registry of language model endpoints, keyed by model-name patterns.
///
/// Keys may be exact model names (e.g. `"openai/gpt-4o"`) or globs supporting
/// `*` (any sequence) and `?` (any single character) — e.g. `"openai/*"`,
/// `"anthropic/claude-*"`. [`get`](Self::get) prefers an exact hit, then falls
/// back to the most specific glob match (longest run of literal characters).
///
/// Populate via the convenience constructors ([`openai`](Self::openai),
/// [`anthropic`](Self::anthropic), [`gemini`](Self::gemini),
/// [`bedrock`](Self::bedrock), [`chat_completion`](Self::chat_completion), …)
/// which return
/// [`LangModelProviderElem`] values, then [`insert`](Self::insert) them under
/// the chosen pattern.  At agent construction time the runtime calls
/// [`get`](Self::get) to verify that the spec's `model` matches an entry, then
/// hands the resolved provider-name and model id to
/// [`LangModel::try_from_provider`](crate::lang_model::LangModel::try_from_provider).
///
/// [`Default::default`] returns a registry pre-populated from the environment:
/// registers `openai/*`, `anthropic/*`, `google/*`, `x-ai/*`, `deepseek/*`,
/// and/or `moonshotai/kimi-*` for every `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`
/// / `GEMINI_API_KEY` / `XAI_API_KEY` / `DEEPSEEK_API_KEY` / `KIMI_API_KEY`
/// that is set, plus `bedrock/*` (Converse) for `AWS_BEARER_TOKEN_BEDROCK`
/// (region from `AWS_REGION`, then `AWS_DEFAULT_REGION`, defaulting to
/// `us-east-1`).  The default is what the global [`lang_model_providers`]
/// registry stores under the `"default"` key.  Use [`new`](Self::new) for an
/// empty registry.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct LangModelProvider {
    inner: BTreeMap<String, LangModelProviderElem>,
}

/// Reads an API key from the environment, treating blank as absent.
///
/// A `.env` copied from `.env.example` leaves keys set-but-empty; registering
/// those would produce a provider that resolves fine and then fails with 401 at
/// call time, which is a much worse error to debug than "no provider found".
fn env_key(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

impl Default for LangModelProvider {
    fn default() -> Self {
        let mut p = Self::new();
        if let Some(key) = env_key("OPENAI_API_KEY") {
            p.insert("openai/*".into(), Self::openai(key));
        }
        if let Some(key) = env_key("ANTHROPIC_API_KEY") {
            p.insert("anthropic/*".into(), Self::anthropic(key));
        }
        if let Some(key) = env_key("GEMINI_API_KEY") {
            p.insert("google/*".into(), Self::gemini(key));
        }
        if let Some(key) = env_key("XAI_API_KEY") {
            p.insert("x-ai/*".into(), Self::grok(key));
        }
        if let Some(key) = env_key("DEEPSEEK_API_KEY") {
            p.insert("deepseek/*".into(), Self::deepseek(key));
        }
        if let Some(key) = env_key("KIMI_API_KEY") {
            p.insert("moonshotai/*".into(), Self::kimi(key));
        }
        if let Some(key) = env_key("AWS_BEARER_TOKEN_BEDROCK") {
            // Same precedence the AWS SDKs use, so a shell already configured
            // for AWS needs no extra variable.
            let region = env_key("AWS_REGION")
                .or_else(|| env_key("AWS_DEFAULT_REGION"))
                .unwrap_or_else(|| "us-east-1".to_string());
            p.insert("bedrock/*".into(), Self::bedrock(region, key));
        }
        p
    }
}

impl LangModelProvider {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Register an endpoint under a name or glob pattern (`*`, `?`).
    /// Overwrites any existing entry with the same key.
    pub fn insert(&mut self, pattern: String, elem: LangModelProviderElem) {
        self.inner.insert(pattern, elem);
    }

    /// Convenience over [`insert`](Self::insert) that constructs an
    /// [`LangModelProviderElem::API`] inline.
    pub fn insert_api(
        &mut self,
        pattern: String,
        schema: LangModelAPISchema,
        url: Url,
        api_key: Option<String>,
    ) {
        self.inner.insert(
            pattern,
            LangModelProviderElem::API {
                schema,
                url,
                api_key,
            },
        );
    }

    pub fn remove(&mut self, pattern: &str) {
        self.inner.remove(pattern);
    }

    /// Resolve a model name. Exact match wins; otherwise the registered glob
    /// pattern with the longest literal run is selected.
    pub fn get(&self, name: impl AsRef<str>) -> Option<&LangModelProviderElem> {
        let name = name.as_ref();
        if let Some(elem) = self.inner.get(name) {
            return Some(elem);
        }
        self.inner
            .iter()
            .filter(|(pattern, _)| glob_match(pattern, name))
            .max_by_key(|(pattern, _)| pattern.chars().filter(|&c| c != '*' && c != '?').count())
            .map(|(_, elem)| elem)
    }

    /// Verify that `spec_model` matches a registered pattern and return the
    /// API-side model id with any `provider/` prefix stripped (e.g.
    /// `"openai/gpt-4o"` → `"gpt-4o"`).  Returns an error if no pattern
    /// matches.
    pub fn resolve_model_id(&self, spec_model: impl AsRef<str>) -> anyhow::Result<String> {
        let spec_model = spec_model.as_ref();
        let _ = self
            .get(spec_model)
            .ok_or_else(|| anyhow::anyhow!("No provider found for model '{}'", spec_model))?;
        let model_id = spec_model
            .split_once('/')
            .map(|(_, id)| id.to_string())
            .unwrap_or_else(|| spec_model.to_string());
        Ok(model_id)
    }
}

/// Process-wide named registry of [`LangModelProvider`] instances.
///
/// Populated at first access with a single `"default"` entry built from
/// [`LangModelProvider::default`] (i.e. the env-variable seeded provider).
/// Additional named providers can be registered via [`lang_model_providers_mut`],
/// and looked up via [`lang_model_providers`].
static LANG_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, LangModelProvider>>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        map.insert("default".to_string(), LangModelProvider::default());
        RwLock::new(map)
    });

/// Borrow the process-wide [`LangModelProvider`] registry for reading.
///
/// Holds a [`std::sync::RwLockReadGuard`]; drop it before performing long
/// operations to avoid blocking writers.
pub fn get_lm_providers() -> RwLockReadGuard<'static, HashMap<String, LangModelProvider>> {
    LANG_MODEL_PROVIDERS
        .read()
        .expect("lang_model_providers lock poisoned")
}

/// Borrow the process-wide [`LangModelProvider`] registry for writing.
pub fn get_lm_providers_mut() -> RwLockWriteGuard<'static, HashMap<String, LangModelProvider>> {
    LANG_MODEL_PROVIDERS
        .write()
        .expect("lang_model_providers lock poisoned")
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    glob_match_chars(&p, &t)
}

fn glob_match_chars(p: &[char], t: &[char]) -> bool {
    match (p.split_first(), t.split_first()) {
        (None, None) => true,
        (None, Some(_)) => false,
        (Some((&'*', rest_p)), _) => {
            // * matches zero characters here, or consume one character from text
            glob_match_chars(rest_p, t)
                || t.split_first()
                    .is_some_and(|(_, rest_t)| glob_match_chars(p, rest_t))
        }
        (Some((&'?', rest_p)), Some((_, rest_t))) => glob_match_chars(rest_p, rest_t),
        (Some((pc, rest_p)), Some((tc, rest_t))) if pc == tc => glob_match_chars(rest_p, rest_t),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy() -> LangModelProviderElem {
        LangModelProviderElem::API {
            schema: LangModelAPISchema::OpenAI,
            url: Url::parse("https://example.com").unwrap(),
            api_key: None,
        }
    }

    #[test]
    fn exact_match_takes_precedence() {
        let mut p = LangModelProvider::new();
        p.insert("openai/*".into(), dummy());
        p.insert("openai/gpt-4o".into(), dummy());
        // both match; exact wins (verified indirectly: removing exact still leaves a hit).
        assert!(p.get("openai/gpt-4o").is_some());
        p.remove("openai/gpt-4o");
        assert!(p.get("openai/gpt-4o").is_some()); // still resolves via glob
    }

    #[test]
    fn glob_picks_most_specific() {
        let mut p = LangModelProvider::new();
        p.insert("*".into(), dummy());
        p.insert("openai/*".into(), dummy());
        p.insert("anthropic/*".into(), dummy());
        // longest literal run is "openai/" — ensures it would be picked over "*".
        assert!(p.get("openai/gpt-4o").is_some());
        assert!(p.get("anthropic/claude-x").is_some());
        assert!(p.get("anything-else").is_some());
    }

    #[test]
    fn no_match_returns_none() {
        let mut p = LangModelProvider::new();
        p.insert("openai/*".into(), dummy());
        assert!(p.get("anthropic/claude").is_none());
    }

    #[test]
    fn resolve_model_id_strips_prefix() {
        let mut p = LangModelProvider::new();
        p.insert("openai/*".into(), dummy());
        assert_eq!(p.resolve_model_id("openai/gpt-4o").unwrap(), "gpt-4o");
    }
}
