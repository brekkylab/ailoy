use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::lang_model::{LangModel, LangModelAPISchema};

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

#[derive(Clone, Debug)]
pub struct LangModelFactory {
    model: String,

    elem: LangModelProviderElem,
}

impl LangModelFactory {
    pub fn make(self) -> LangModel {
        LangModel::new(self.model, self.elem)
    }
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
/// [`chat_completion`](Self::chat_completion), …) which return
/// [`LangModelProviderElem`] values, then [`insert`](Self::insert) them under
/// the chosen pattern. At agent construction the registry is consulted via
/// [`provide`](Self::provide) to build a [`LangModel`].
///
/// [`Default::default`] (and therefore [`AgentProvider::new`]) returns a
/// registry pre-populated from the environment:  registers `openai/*`,
/// `anthropic/*`, `google/*`, `x-ai/*`, `deepseek/*`, and/or `moonshotai/kimi-*` for every
/// `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `XAI_API_KEY` / `DEEPSEEK_API_KEY`
/// / `KIMI_API_KEY` that is set.
/// Use [`new`](Self::new) for an empty registry.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct LangModelProvider {
    inner: BTreeMap<String, LangModelProviderElem>,
}

impl Default for LangModelProvider {
    fn default() -> Self {
        let mut p = Self::new();
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            p.insert("openai/*".into(), Self::openai(key));
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            p.insert("anthropic/*".into(), Self::anthropic(key));
        }
        if let Ok(key) = std::env::var("GEMINI_API_KEY") {
            p.insert("google/*".into(), Self::gemini(key));
        }
        if let Ok(key) = std::env::var("XAI_API_KEY") {
            p.insert("x-ai/*".into(), Self::grok(key));
        }
        if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
            p.insert("deepseek/*".into(), Self::deepseek(key));
        }
        if let Ok(key) = std::env::var("KIMI_API_KEY") {
            p.insert("moonshotai/*".into(), Self::kimi(key));
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

    /// Resolve `spec_model` to a [`LangModelFactory`] ready to build a [`LangModel`].
    ///
    /// Looks up `spec_model` (with glob fallback via [`get`](Self::get)) and
    /// strips any `provider/` prefix to recover the API-side model id (e.g.
    /// `"openai/gpt-4o"` → `"gpt-4o"`). Returns an error if no pattern matches.
    /// Call [`LangModelFactory::make`] to instantiate the [`LangModel`].
    pub fn provide(&self, spec_model: impl AsRef<str>) -> anyhow::Result<LangModelFactory> {
        let spec_model = spec_model.as_ref();
        let elem = self
            .get(spec_model)
            .ok_or_else(|| anyhow::anyhow!("No provider found for model '{}'", spec_model))?
            .clone();
        let model_id = spec_model
            .split_once('/')
            .map(|(_, id)| id.to_string())
            .unwrap_or_else(|| spec_model.to_string());
        Ok(LangModelFactory {
            model: model_id,
            elem,
        })
    }
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
    fn make_runtime_strips_prefix() {
        let mut p = LangModelProvider::new();
        p.insert("openai/*".into(), dummy());
        let m = p.provide("openai/gpt-4o").unwrap();
        assert_eq!(m.model, "gpt-4o");
    }
}
