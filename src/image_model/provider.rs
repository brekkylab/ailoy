use std::{
    collections::{BTreeMap, HashMap},
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::image_model::ImageModelAPISchema;

/// Describes the runtime endpoint used to invoke an image generation model.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ImageModelProviderElem {
    /// Calls a remote HTTP API. Requires the wire `schema`, the `url` of the endpoint, and an optional `api_key` for authentication.
    API {
        schema: ImageModelAPISchema,

        url: Url,

        api_key: Option<String>,
    },
}

/// Registry of image model endpoints, keyed by model-name patterns.
///
/// The twin of [`LangModelProvider`](crate::lang_model::LangModelProvider), and
/// deliberately a separate registry: image generation has its own endpoints,
/// and a text model registered for chat must not resolve for
/// [`ImageModel`](crate::image_model::ImageModel).
///
/// Keys may be exact model names (e.g. `"openai/gpt-image-1"`) or globs
/// supporting `*` (any sequence) and `?` (any single character) — e.g.
/// `"openai/*"`, `"google/gemini-*-image"`. [`get`](Self::get) prefers an exact
/// hit, then falls back to the most specific glob match (longest run of literal
/// characters).
///
/// Populate via the convenience constructors ([`openai`](Self::openai),
/// [`gemini`](Self::gemini)) which return [`ImageModelProviderElem`] values,
/// then [`insert`](Self::insert) them under the chosen pattern.
///
/// [`Default::default`] returns a registry pre-populated from the environment:
/// registers `openai/*` for `OPENAI_API_KEY` and `google/*` for
/// `GEMINI_API_KEY`, skipping any that is unset or blank.  The default is what
/// the global registry stores under the `"default"` key — see
/// [`get_im_providers`].  Use [`new`](Self::new) for an empty registry.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
#[schemars(transparent)]
pub struct ImageModelProvider {
    inner: BTreeMap<String, ImageModelProviderElem>,
}

impl Default for ImageModelProvider {
    fn default() -> Self {
        Self::from_keys(|name| std::env::var(name).ok())
    }
}

impl ImageModelProvider {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    /// Seed a registry from a key lookup, the way [`Default`] does from the
    /// environment.  Blank values count as absent: a `.env` copied from
    /// `.env.example` leaves keys set-but-empty, and registering those would
    /// produce a provider that resolves fine and then fails with 401 at call
    /// time, which is a much worse error to debug than "no provider found".
    ///
    /// Split out from [`Default`] so the seeding rules are testable without
    /// mutating the process environment.
    fn from_keys(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let key = |name: &str| lookup(name).filter(|v| !v.trim().is_empty());
        let mut p = Self::new();
        if let Some(k) = key("OPENAI_API_KEY") {
            p.insert("openai/*".into(), Self::openai(k));
        }
        if let Some(k) = key("GEMINI_API_KEY") {
            p.insert("google/*".into(), Self::gemini(k));
        }
        p
    }

    /// Register an endpoint under a name or glob pattern (`*`, `?`).
    /// Overwrites any existing entry with the same key.
    pub fn insert(&mut self, pattern: String, elem: ImageModelProviderElem) {
        self.inner.insert(pattern, elem);
    }

    /// Convenience over [`insert`](Self::insert) that constructs an
    /// [`ImageModelProviderElem::API`] inline.
    pub fn insert_api(
        &mut self,
        pattern: String,
        schema: ImageModelAPISchema,
        url: Url,
        api_key: Option<String>,
    ) {
        self.inner.insert(
            pattern,
            ImageModelProviderElem::API {
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
    pub fn get(&self, name: impl AsRef<str>) -> Option<&ImageModelProviderElem> {
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
    /// `"openai/gpt-image-1"` → `"gpt-image-1"`).  Returns an error if no
    /// pattern matches.
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

/// Process-wide named registry of [`ImageModelProvider`] instances.
///
/// Populated at first access with a single `"default"` entry built from
/// [`ImageModelProvider::default`] (i.e. the env-variable seeded provider).
/// Additional named providers can be registered via [`get_im_providers_mut`],
/// and looked up via [`get_im_providers`].
static IMAGE_MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, ImageModelProvider>>> =
    LazyLock::new(|| {
        let mut map = HashMap::new();
        map.insert("default".to_string(), ImageModelProvider::default());
        RwLock::new(map)
    });

/// Borrow the process-wide [`ImageModelProvider`] registry for reading.
///
/// Holds a [`std::sync::RwLockReadGuard`]; drop it before performing long
/// operations to avoid blocking writers.
pub fn get_im_providers() -> RwLockReadGuard<'static, HashMap<String, ImageModelProvider>> {
    IMAGE_MODEL_PROVIDERS
        .read()
        .expect("image_model_providers lock poisoned")
}

/// Borrow the process-wide [`ImageModelProvider`] registry for writing.
pub fn get_im_providers_mut() -> RwLockWriteGuard<'static, HashMap<String, ImageModelProvider>> {
    IMAGE_MODEL_PROVIDERS
        .write()
        .expect("image_model_providers lock poisoned")
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

    /// An entry tagged by its url, so a lookup's *result* can be identified —
    /// asserting `is_some()` on interchangeable entries would pass even if
    /// precedence were resolved backwards.
    fn at(url: &str) -> ImageModelProviderElem {
        ImageModelProviderElem::API {
            schema: ImageModelAPISchema::OpenAI,
            url: Url::parse(url).unwrap(),
            api_key: None,
        }
    }

    fn url_of(elem: &ImageModelProviderElem) -> String {
        let ImageModelProviderElem::API { url, .. } = elem;
        url.to_string()
    }

    fn dummy() -> ImageModelProviderElem {
        at("https://example.com")
    }

    #[test]
    fn exact_match_takes_precedence() {
        let mut p = ImageModelProvider::new();
        p.insert("openai/*".into(), at("https://example.com/glob"));
        p.insert("openai/gpt-image-1".into(), at("https://example.com/exact"));
        assert_eq!(
            url_of(p.get("openai/gpt-image-1").expect("exact entry")),
            "https://example.com/exact"
        );

        p.remove("openai/gpt-image-1");
        assert_eq!(
            url_of(p.get("openai/gpt-image-1").expect("glob fallback")),
            "https://example.com/glob",
            "with the exact entry gone the glob must take over"
        );
    }

    #[test]
    fn glob_picks_most_specific() {
        let mut p = ImageModelProvider::new();
        p.insert("*".into(), at("https://example.com/any"));
        p.insert("openai/*".into(), at("https://example.com/openai"));
        p.insert(
            "google/gemini-?.?-flash-image".into(),
            at("https://example.com/gemini"),
        );
        assert_eq!(
            url_of(p.get("openai/gpt-image-1").expect("openai/* matches")),
            "https://example.com/openai",
            "the longer literal run must beat the bare `*`"
        );
        assert_eq!(
            url_of(
                p.get("google/gemini-3.1-flash-image")
                    .expect("the `?` pattern matches")
            ),
            "https://example.com/gemini"
        );
        assert_eq!(
            url_of(p.get("anything-else").expect("`*` matches everything")),
            "https://example.com/any"
        );
    }

    #[test]
    fn no_match_returns_none() {
        let mut p = ImageModelProvider::new();
        p.insert("openai/*".into(), dummy());
        assert!(p.get("google/gemini-3.1-flash-image").is_none());
    }

    #[test]
    fn resolve_model_id_strips_prefix() {
        let mut p = ImageModelProvider::new();
        p.insert("openai/*".into(), dummy());
        assert_eq!(
            p.resolve_model_id("openai/gpt-image-1").unwrap(),
            "gpt-image-1"
        );
        assert!(p.resolve_model_id("google/gemini-3.1-flash-image").is_err());
    }

    #[test]
    fn default_seeding_registers_env_backed_patterns_and_ignores_blanks() {
        let p = ImageModelProvider::from_keys(|name| match name {
            "OPENAI_API_KEY" => Some("sk-test".to_string()),
            "GEMINI_API_KEY" => Some("AIza-test".to_string()),
            _ => None,
        });
        let ImageModelProviderElem::API {
            schema,
            url,
            api_key,
        } = p.get("openai/gpt-image-1").expect("openai/* registered");
        assert!(matches!(schema, ImageModelAPISchema::OpenAI));
        assert_eq!(url.as_str(), "https://api.openai.com/v1/images/generations");
        assert_eq!(api_key.as_deref(), Some("sk-test"));
        let ImageModelProviderElem::API { schema, url, .. } = p
            .get("google/gemini-3.1-flash-image")
            .expect("google/* registered");
        assert!(matches!(schema, ImageModelAPISchema::Gemini));
        assert_eq!(
            url.as_str(),
            "https://generativelanguage.googleapis.com/v1beta/models/"
        );

        // A blank key (a `.env` copied from an example) registers nothing.
        let p = ImageModelProvider::from_keys(|name| match name {
            "GEMINI_API_KEY" => Some("   ".to_string()),
            _ => None,
        });
        assert!(p.get("google/gemini-3.1-flash-image").is_none());
        assert!(p.get("openai/gpt-image-1").is_none());
    }
}
