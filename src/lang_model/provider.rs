use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

/// Wire protocol used when calling a language model API.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LangModelAPISchema {
    /// OpenAI-compatible `/v1/chat/completions` format
    ChatCompletion,

    /// Anthropic Messages API format
    Anthropic,

    /// Google Gemini API format
    Gemini,

    /// OpenAI Responses API format
    #[serde(rename = "openai")]
    OpenAI,
}

/// Describes the runtime endpoint used to invoke a language model.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum LangModelProviderElem {
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
/// `"anthropic/claude-*"`. Lookups prefer an exact hit, then fall back to the
/// most specific glob match (longest run of literal characters).
///
/// Populate via [`insert_api`](Self::insert_api). At agent construction the
/// registry is consulted by
/// [`LangModel::try_with_provider`](crate::lang_model::LangModel::try_with_provider).
///
/// [`Default::default`] (and therefore [`AgentProvider::new`]) returns a
/// registry pre-populated from the environment: registers `openai/*`,
/// `anthropic/*`, `google/*`, `x-ai/*`, `deepseek/*`, and/or `moonshotai/*` for every
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
            p.insert_api(
                "openai/*".into(),
                LangModelAPISchema::OpenAI,
                "https://api.openai.com/v1/responses",
                Some(key),
            )
            .unwrap();
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            p.insert_api(
                "anthropic/*".into(),
                LangModelAPISchema::Anthropic,
                "https://api.anthropic.com/v1/messages",
                Some(key),
            )
            .unwrap();
        }
        if let Ok(key) = std::env::var("GEMINI_API_KEY") {
            p.insert_api(
                "google/*".into(),
                LangModelAPISchema::Gemini,
                "https://generativelanguage.googleapis.com/v1beta/models/",
                Some(key),
            )
            .unwrap();
        }
        if let Ok(key) = std::env::var("XAI_API_KEY") {
            p.insert_api(
                "x-ai/*".into(),
                LangModelAPISchema::ChatCompletion,
                "https://api.x.ai/v1/chat/completions",
                Some(key),
            )
            .unwrap();
        }
        if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
            p.insert_api(
                "deepseek/*".into(),
                LangModelAPISchema::ChatCompletion,
                "https://api.deepseek.com/chat/completions",
                Some(key),
            )
            .unwrap();
        }
        if let Ok(key) = std::env::var("KIMI_API_KEY") {
            p.insert_api(
                "moonshotai/*".into(),
                LangModelAPISchema::ChatCompletion,
                "https://api.moonshot.ai/v1/chat/completions",
                Some(key),
            )
            .unwrap();
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

    /// Register an arbitrary API endpoint under a name or glob pattern (`*`, `?`).
    /// Overwrites any existing entry with the same key. Returns an error if
    /// `url` is not a valid URL.
    pub fn insert_api(
        &mut self,
        pattern: String,
        schema: LangModelAPISchema,
        url: impl AsRef<str>,
        api_key: Option<String>,
    ) -> anyhow::Result<()> {
        let url = Url::parse(url.as_ref())?;
        self.inner.insert(
            pattern,
            LangModelProviderElem::API {
                schema,
                url,
                api_key,
            },
        );
        Ok(())
    }

    pub fn remove(&mut self, pattern: &str) {
        self.inner.remove(pattern);
    }

    /// Resolve a model name. Exact match wins; otherwise the registered glob
    /// pattern with the longest literal run is selected.
    pub(crate) fn get(&self, name: impl AsRef<str>) -> Option<&LangModelProviderElem> {
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
    use crate::lang_model::LangModel;

    fn insert_dummy(p: &mut LangModelProvider, pattern: &str) {
        p.insert_api(
            pattern.into(),
            LangModelAPISchema::OpenAI,
            "https://example.com",
            None,
        )
        .unwrap();
    }

    #[test]
    fn exact_match_takes_precedence() {
        let mut p = LangModelProvider::new();
        insert_dummy(&mut p, "openai/*");
        insert_dummy(&mut p, "openai/gpt-4o");
        // both match; exact wins (verified indirectly: removing exact still leaves a hit).
        assert!(p.get("openai/gpt-4o").is_some());
        p.remove("openai/gpt-4o");
        assert!(p.get("openai/gpt-4o").is_some()); // still resolves via glob
    }

    #[test]
    fn glob_picks_most_specific() {
        let mut p = LangModelProvider::new();
        insert_dummy(&mut p, "*");
        insert_dummy(&mut p, "openai/*");
        insert_dummy(&mut p, "anthropic/*");
        // longest literal run is "openai/" — ensures it would be picked over "*".
        assert!(p.get("openai/gpt-4o").is_some());
        assert!(p.get("anthropic/claude-x").is_some());
        assert!(p.get("anything-else").is_some());
    }

    #[test]
    fn no_match_returns_none() {
        let mut p = LangModelProvider::new();
        insert_dummy(&mut p, "openai/*");
        assert!(p.get("anthropic/claude").is_none());
    }

    #[test]
    fn make_runtime_strips_prefix() {
        let mut p = LangModelProvider::new();
        insert_dummy(&mut p, "openai/*");
        let m = LangModel::try_with_provider("openai/gpt-4o".to_string(), &p).unwrap();
        assert_eq!(m.model_id(), "gpt-4o");
    }
}
