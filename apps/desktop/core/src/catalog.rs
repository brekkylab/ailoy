//! Model metadata — context window, output cap, prices, capabilities — from models.dev.
//!
//! The list lives in the app's cache directory (`<data dir>/cache/models.json`), refreshed
//! from `https://models.dev/api.json` in the background (see `Engine::start`). A release
//! build also embeds the snapshot `npm run catalog` fetched just before it was built, for the
//! one start that has no cache and cannot reach models.dev — a network that lets the
//! provider APIs through but not a third-party site. Nothing is committed: a clone or a dev
//! build embeds nothing (see `build.rs`) and starts with an empty list until the first
//! refresh lands.
//!
//! Of the two, whichever was fetched later is used, so an update that ships a fresher
//! snapshot beats a cache an install last wrote months ago.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::RwLock,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::sync::watch;

use crate::types::{CatalogStatus, ModelCost, RegionRouting};

pub const MODELS_DEV_URL: &str = "https://models.dev/api.json";

/// The shape of a stored catalog, which is [`filter_models_dev`]'s output and not models.dev's
/// own. Bump it when that filter changes what it keeps: a cache written by an older filter
/// is otherwise fresh by its timestamp and would outlive the update that fixed it.
pub const FORMAT: u32 = 1;

/// How old the list may get before the background loop fetches it again.
pub const REFRESH_EVERY: Duration = Duration::from_secs(6 * 60 * 60);

/// How long the loop waits after a failed fetch. Short against `REFRESH_EVERY`, because a
/// failure on the first start leaves nothing to pick until one succeeds.
pub const RETRY_AFTER: Duration = Duration::from_secs(5 * 60);

/// The release build's snapshot, or an empty string when it was built without one.
const EMBEDDED: &str = include_str!(concat!(env!("OUT_DIR"), "/models.json"));

/// ailoy model-id prefix → models.dev provider id.
pub const PROVIDER_MAP: &[(&str, &str)] = &[
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("google", "google"),
    ("x-ai", "xai"),
    ("deepseek", "deepseek"),
    ("moonshotai", "moonshotai"),
    ("bedrock", "amazon-bedrock"),
];

pub fn models_dev_provider(ailoy_prefix: &str) -> Option<&'static str> {
    PROVIDER_MAP
        .iter()
        .find(|(p, _)| *p == ailoy_prefix)
        .map(|(_, id)| *id)
}

/// `"anthropic/claude-opus-5"` → `("anthropic", "claude-opus-5")`.
pub fn split_model_id(ailoy_model: &str) -> Option<(&str, &str)> {
    ailoy_model
        .split_once('/')
        .filter(|(p, m)| !p.is_empty() && !m.is_empty())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default)]
    pub tool_call: bool,
    pub context: Option<u64>,
    pub output: Option<u64>,
    pub cost: Option<ModelCost>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogProvider {
    pub name: String,
    pub models: BTreeMap<String, CatalogModel>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CatalogData {
    /// [`FORMAT`] when written. Missing reads as 0, which no current build accepts.
    #[serde(default)]
    pub format: u32,
    /// When this was fetched from models.dev, in Unix milliseconds — our clock, not a
    /// timestamp models.dev publishes. Missing reads as older than anything.
    #[serde(default)]
    pub fetched_at: Option<i64>,
    pub providers: BTreeMap<String, CatalogProvider>,
}

impl CatalogData {
    fn model_count(&self) -> usize {
        self.providers.values().map(|p| p.models.len()).sum()
    }

    /// A stored catalog this build can use: its own format, and something in it.
    fn usable(self) -> Option<CatalogData> {
        (self.format == FORMAT && !self.providers.is_empty()).then_some(self)
    }
}

/// models.dev, filtered and stamped: what the cache and the release snapshot both hold.
pub async fn fetch_models_dev() -> anyhow::Result<CatalogData> {
    let full: serde_json::Value = reqwest::Client::new()
        .get(MODELS_DEV_URL)
        .timeout(Duration::from_secs(20))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let mut data = filter_models_dev(&full);
    anyhow::ensure!(
        !data.providers.is_empty(),
        "models.dev answered with none of our providers"
    );
    data.format = FORMAT;
    data.fetched_at = Some(chrono::Utc::now().timestamp_millis());
    Ok(data)
}

/// The raw models.dev shape, only the fields read here. Everything is optional so a new
/// field upstream never breaks the parse.
#[derive(Deserialize)]
struct RawProvider {
    #[serde(default)]
    name: String,
    #[serde(default)]
    models: BTreeMap<String, RawModel>,
}

#[derive(Deserialize, Default)]
struct RawModel {
    #[serde(default)]
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    modalities: RawModalities,
    #[serde(default)]
    limit: RawLimit,
    cost: Option<ModelCost>,
}

#[derive(Deserialize, Default)]
struct RawModalities {
    #[serde(default)]
    output: Vec<String>,
}

#[derive(Deserialize, Default)]
struct RawLimit {
    context: Option<u64>,
    output: Option<u64>,
}

/// Keep the providers ailoy can call and the models an agent can drive: tool-calling, text
/// out, a known context window.
pub fn filter_models_dev(full: &serde_json::Value) -> CatalogData {
    let mut data = CatalogData::default();
    for (_, md_id) in PROVIDER_MAP {
        let Some(raw) = full.get(*md_id) else {
            continue;
        };
        let Ok(provider) = serde_json::from_value::<RawProvider>(raw.clone()) else {
            continue;
        };
        let models: BTreeMap<String, CatalogModel> = provider
            .models
            .into_iter()
            .filter(|(_, m)| {
                m.tool_call
                    && m.modalities.output.iter().any(|o| o == "text")
                    && m.limit.context.is_some()
            })
            .map(|(key, m)| {
                let id = if m.id.is_empty() { key.clone() } else { m.id };
                let name = if m.name.is_empty() {
                    id.clone()
                } else {
                    m.name
                };
                (
                    key,
                    CatalogModel {
                        id,
                        name,
                        reasoning: m.reasoning,
                        tool_call: m.tool_call,
                        context: m.limit.context,
                        output: m.limit.output,
                        cost: m.cost,
                    },
                )
            })
            .collect();
        if !models.is_empty() {
            data.providers.insert(
                md_id.to_string(),
                CatalogProvider {
                    name: provider.name,
                    models,
                },
            );
        }
    }
    data
}

/// models.dev's marker for a row that is an alias rather than a snapshot of its own.
const LATEST_MARKER: &str = " (latest)";

/// Whether `id` is `alias` plus a date: `claude-sonnet-4-5-20250929` behind
/// `claude-sonnet-4-5`, and not `gpt-4o-mini` behind `gpt-4o`.
fn is_dated_snapshot_of(id: &str, alias: &str) -> bool {
    id.strip_prefix(alias)
        .and_then(|rest| rest.strip_prefix('-'))
        .is_some_and(|date| {
            !date.is_empty() && date.chars().all(|c| c.is_ascii_digit() || c == '-')
        })
}

/// One provider's models as a picker should offer them.
///
/// models.dev lists a model once per API id, and for Claude's 4.5 generation that means
/// twice: `claude-sonnet-4-5`, which Anthropic documents as "a convenience pointer that
/// resolves to the dated ID", and `claude-sonnet-4-5-20250929`, the snapshot it resolves
/// to. Same context, same prices, same capabilities — models.dev tells the two apart by
/// naming the alias `Claude Sonnet 4.5 (latest)`, so the picker offered one model on two
/// rows, one of them decorated. From the 4.6 generation on the dateless id *is* its own
/// pinned snapshot and there is nothing to collapse.
///
/// So an alias row swallows the dated rows behind it: an id that is the alias's plus a
/// date, carrying the alias's own name once the marker is off. Nothing else is touched —
/// OpenAI names its snapshots `GPT-4o (2024-08-06)`, which is a row that says something
/// the alias does not and keeps its place.
///
/// Only the listing collapses. The catalog still holds every id, and `lookup` still
/// answers for a session pinned to a dated snapshot — which is where that session's
/// context window and prices come from.
pub fn listed_models(models: &BTreeMap<String, CatalogModel>) -> Vec<CatalogModel> {
    let mut shadowed: BTreeSet<&str> = BTreeSet::new();
    for alias in models.values() {
        let Some(bare) = alias.name.strip_suffix(LATEST_MARKER) else {
            continue;
        };
        for m in models.values() {
            if is_dated_snapshot_of(&m.id, &alias.id) && m.name == bare {
                shadowed.insert(m.id.as_str());
            }
        }
    }

    let mut out: Vec<CatalogModel> = models
        .values()
        .filter(|m| !shadowed.contains(m.id.as_str()))
        .cloned()
        .collect();
    // With the row it was distinguishing itself from gone, the marker is noise. Left on
    // where some other listed row already carries the bare name, so that collapsing ids
    // never ends in two rows that read the same.
    let taken: BTreeSet<String> = out.iter().map(|m| m.name.clone()).collect();
    for m in &mut out {
        if let Some(bare) = m.name.strip_suffix(LATEST_MARKER)
            && !taken.contains(bare)
        {
            m.name = bare.to_string();
        }
    }
    out
}

/// Bedrock's inference-profile prefixes, as they are spelled at the front of a model id.
///
/// A closed list because it is what tells `us.anthropic.claude-…` — one model reached
/// through the US profile — from `anthropic.claude-…`, the plain on-demand id, without
/// having to guess that `anthropic` is a vendor and `us` is not. A prefix models.dev starts
/// using that is missing from here simply keeps its own row, which is the safe way to be
/// wrong: nothing is merged that should not be.
pub const REGION_PREFIXES: &[&str] = &[
    "global", "us", "us-gov", "eu", "apac", "au", "jp", "in", "ca", "sa",
];

/// `"us.anthropic.claude-opus-5"` → `("us", "anthropic.claude-opus-5")`, and `None` for an
/// id that is not reached through a profile.
///
/// The remainder has to carry a dot of its own: every Bedrock id is vendor-qualified, so
/// `us.anthropic.…` splits and a hypothetical bare `global-something` does not.
pub fn region_prefix(id: &str) -> Option<(&str, &str)> {
    let (head, rest) = id.split_once('.')?;
    (REGION_PREFIXES.contains(&head) && rest.contains('.')).then_some((head, rest))
}

/// The same model, whichever profile it is reached through.
fn base_id(id: &str) -> &str {
    region_prefix(id).map_or(id, |(_, base)| base)
}

/// The ways this provider will route a call, in the order a menu should show them.
///
/// Derived from the ids rather than declared: the label is the parenthesis models.dev puts
/// on the name — `(GovCloud)` for `us-gov`, `(India)` for `in` — which is neither the
/// prefix nor a mechanical uppercasing of it, and is not ours to invent. A profile whose
/// models are all named without one falls back to the prefix itself.
pub fn region_routings(models: &[CatalogModel]) -> Vec<RegionRouting> {
    let mut out: Vec<RegionRouting> = Vec::new();
    for m in models {
        let Some((prefix, _)) = region_prefix(&m.id) else {
            continue;
        };
        if out.iter().any(|r| r.id == prefix) {
            continue;
        }
        let label = m
            .name
            .rsplit_once(" (")
            .and_then(|(_, tail)| tail.strip_suffix(')'))
            .map(str::to_string)
            .unwrap_or_else(|| prefix.to_string());
        out.push(RegionRouting {
            id: prefix.to_string(),
            label,
        });
    }
    // `REGION_PREFIXES` order, so the menu reads the same whatever order the catalog is in.
    out.sort_by_key(|r| {
        REGION_PREFIXES
            .iter()
            .position(|p| *p == r.id)
            .unwrap_or(usize::MAX)
    });
    out
}

/// One row per model, reached through the profile the user asked for.
///
/// Bedrock offers a model once per inference profile — global for dynamic routing, a
/// regional one for guaranteed data routing — and models.dev lists each as its own model:
/// 158 of them, most of which are `Nova Pro (US)`, `Nova Pro (EU)`, `Nova Pro (APAC)`.
/// That is the provider's catalogue, not a menu.
///
/// So a model appears once, and `routing` decides which profile it is reached through.
/// Failing that profile it is the plain id, which Bedrock serves in whatever region the
/// client is pointed at; failing both, every profile there is, each keeping the region in
/// its name — a model offered only in GovCloud is not silently a global one.
///
/// The name comes from the plain id where there is one, because the parenthesis models.dev
/// adds is about the profile and not about the model, and `Pixtral Large (25.02)` shows
/// that not every parenthesis is.
pub fn fold_region_profiles(models: &[CatalogModel], routing: &str) -> Vec<CatalogModel> {
    let mut bases: Vec<&str> = Vec::new();
    for m in models {
        let base = base_id(&m.id);
        if !bases.contains(&base) {
            bases.push(base);
        }
    }

    let mut out = Vec::with_capacity(bases.len());
    for base in bases {
        let group: Vec<&CatalogModel> = models.iter().filter(|m| base_id(&m.id) == base).collect();
        let plain = group.iter().find(|m| region_prefix(&m.id).is_none());
        let routed = group
            .iter()
            .find(|m| region_prefix(&m.id).is_some_and(|(p, _)| p == routing));
        match (routed, plain) {
            (Some(r), Some(p)) => out.push(CatalogModel {
                name: p.name.clone(),
                ..(*r).clone()
            }),
            (Some(r), None) => out.push((*r).clone()),
            (None, Some(p)) => out.push((*p).clone()),
            (None, None) => out.extend(group.into_iter().cloned()),
        }
    }
    out
}

pub struct Catalog {
    data: RwLock<CatalogData>,
    /// What the window is told: every change to the list, and to whether one is on its way.
    status: watch::Sender<CatalogStatus>,
    /// One fetch at a time. The background loop and a Refresh click can land together, and
    /// two writers racing on the cache file is how it ends up holding the older answer.
    fetching: tokio::sync::Mutex<()>,
}

impl Catalog {
    pub fn from_data(data: CatalogData) -> Catalog {
        let status = CatalogStatus {
            fetched_at: data.fetched_at,
            models: data.model_count(),
            ..CatalogStatus::default()
        };
        Catalog {
            data: RwLock::new(data),
            status: watch::channel(status).0,
            fetching: tokio::sync::Mutex::new(()),
        }
    }

    /// The release snapshot or the cache, whichever was fetched later; an empty catalog when
    /// neither is there.
    pub fn load(cache: Option<&Path>) -> Catalog {
        Catalog::load_from(EMBEDDED, cache)
    }

    fn load_from(embedded: &str, cache: Option<&Path>) -> Catalog {
        let embedded = parse_stored(embedded, "the embedded catalog");
        let cached = cache
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| parse_stored(&s, "the cached catalog"));
        Catalog::from_data(newer(embedded, cached).unwrap_or_default())
    }

    pub fn status(&self) -> CatalogStatus {
        self.status.borrow().clone()
    }

    pub fn subscribe(&self) -> watch::Receiver<CatalogStatus> {
        self.status.subscribe()
    }

    /// How long ago the list in use was fetched. `None` when there is no list, and when its
    /// timestamp is ahead of this clock — a clock that moved back — which is as good as not
    /// knowing.
    pub fn age(&self) -> Option<Duration> {
        let at = self.status.borrow().fetched_at?;
        let ms = chrono::Utc::now().timestamp_millis().checked_sub(at)?;
        u64::try_from(ms).ok().map(Duration::from_millis)
    }

    pub fn lookup(&self, ailoy_model: &str) -> Option<CatalogModel> {
        let (prefix, model) = split_model_id(ailoy_model)?;
        let md = models_dev_provider(prefix)?;
        self.data
            .read()
            .ok()?
            .providers
            .get(md)?
            .models
            .get(model)
            .cloned()
    }

    /// What a picker offers for this provider. See [`listed_models`]: an alias and the
    /// dated snapshot it points at are one model, and one row.
    pub fn models_for(&self, ailoy_prefix: &str) -> Vec<CatalogModel> {
        let Some(md) = models_dev_provider(ailoy_prefix) else {
            return vec![];
        };
        self.data
            .read()
            .ok()
            .and_then(|d| d.providers.get(md).map(|p| listed_models(&p.models)))
            .unwrap_or_default()
    }

    /// Fetch a fresh list, keep it, and say so. A failure leaves the list as it was, and
    /// the status carries why until a later fetch succeeds.
    ///
    /// A cache that cannot be written is not a failure: the list is still good for as long
    /// as the app runs, and the next start fetches again.
    pub async fn refresh(&self, cache: &Path) -> anyhow::Result<()> {
        self.refresh_from(cache, fetch_models_dev()).await
    }

    /// [`Catalog::refresh`] over any fetch, so the tests can fail one without a network.
    async fn refresh_from(
        &self,
        cache: &Path,
        fetch: impl Future<Output = anyhow::Result<CatalogData>>,
    ) -> anyhow::Result<()> {
        let _one = self.fetching.lock().await;
        self.status.send_modify(|s| s.refreshing = true);
        let outcome = self.keep(cache, fetch).await;
        self.status.send_modify(|s| {
            s.refreshing = false;
            match &outcome {
                Ok((fetched_at, models)) => {
                    s.fetched_at = *fetched_at;
                    s.models = *models;
                    s.error = None;
                }
                Err(e) => s.error = Some(format!("{e:#}")),
            }
        });
        outcome.map(|_| ())
    }

    async fn keep(
        &self,
        cache: &Path,
        fetch: impl Future<Output = anyhow::Result<CatalogData>>,
    ) -> anyhow::Result<(Option<i64>, usize)> {
        let data = fetch.await?;
        if let Err(e) = write_cache(cache, &data).await {
            tracing::warn!(
                "the model catalog is kept in memory only: writing {}: {e:#}",
                cache.display()
            );
        }
        let kept = (data.fetched_at, data.model_count());
        *self
            .data
            .write()
            .map_err(|_| anyhow::anyhow!("catalog lock poisoned"))? = data;
        Ok(kept)
    }
}

async fn write_cache(cache: &Path, data: &CatalogData) -> anyhow::Result<()> {
    if let Some(parent) = cache.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(cache, serde_json::to_vec_pretty(data)?).await?;
    Ok(())
}

/// A stored catalog, or `None` for one this build cannot use — said in the log, since a
/// list that silently stops loading looks like models.dev dropping every model.
fn parse_stored(text: &str, what: &str) -> Option<CatalogData> {
    if text.trim().is_empty() {
        return None;
    }
    let data = match serde_json::from_str::<CatalogData>(text) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("ignoring {what}: {e}");
            return None;
        }
    };
    let format = data.format;
    let usable = data.usable();
    if usable.is_none() {
        tracing::info!("ignoring {what}: format {format}, empty or not this build's {FORMAT}");
    }
    usable
}

/// The later-fetched of two catalogs. A tie goes to the cache, which is what this install
/// fetched itself.
fn newer(embedded: Option<CatalogData>, cached: Option<CatalogData>) -> Option<CatalogData> {
    match (embedded, cached) {
        (Some(e), Some(c)) => Some(if e.fetched_at > c.fetched_at { e } else { c }),
        (e, c) => c.or(e),
    }
}

/// Whether a list this old should be fetched again. No list at all always should.
pub fn is_due(age: Option<Duration>) -> bool {
    age.is_none_or(|a| a >= REFRESH_EVERY)
}

/// How long the background loop sleeps before it looks again, given what it just did.
/// `None` is until something wakes it: with refreshing turned off, it fetches only when
/// the setting is turned back on.
pub fn next_check(enabled: bool, age: Option<Duration>, just_failed: bool) -> Option<Duration> {
    if !enabled {
        return None;
    }
    match age {
        Some(a) if !just_failed => Some(REFRESH_EVERY.saturating_sub(a)),
        _ => Some(RETRY_AFTER),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> serde_json::Value {
        serde_json::json!({
            "anthropic": { "id": "anthropic", "name": "Anthropic", "models": {
                "claude-opus-5": { "id": "claude-opus-5", "name": "Claude Opus 5", "reasoning": true, "tool_call": true,
                    "modalities": {"input": ["text","image"], "output": ["text"]},
                    "limit": {"context": 1000000, "output": 128000},
                    "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25} },
                "claude-3-embed": { "id": "claude-3-embed", "name": "Embed", "tool_call": false,
                    "modalities": {"input": ["text"], "output": ["embedding"]}, "limit": {"context": 8000} }
            }},
            "xai": { "id": "xai", "name": "xAI", "models": {
                "grok-4.6": { "id": "grok-4.6", "name": "Grok 4.6", "tool_call": true,
                    "modalities": {"input": ["text"], "output": ["text"]}, "limit": {"context": 256000, "output": 32000}, "cost": {"input": 3, "output": 15} }
            }},
            "someone-else": { "id": "someone-else", "name": "X", "models": {
                "m": { "id": "m", "name": "m", "tool_call": true, "modalities": {"input":["text"],"output":["text"]}, "limit": {"context": 1} }
            }}
        })
    }

    #[test]
    fn filter_keeps_mapped_providers_and_chat_tool_models_only() {
        let data = filter_models_dev(&sample());
        assert!(data.providers.contains_key("anthropic"));
        assert!(data.providers.contains_key("xai"));
        assert!(!data.providers.contains_key("someone-else"));
        let a = &data.providers["anthropic"];
        assert_eq!(a.name, "Anthropic");
        assert!(a.models.contains_key("claude-opus-5"));
        assert!(
            !a.models.contains_key("claude-3-embed"),
            "no tool_call / non-text output"
        );
        let m = &a.models["claude-opus-5"];
        assert_eq!(m.context, Some(1_000_000));
        assert_eq!(m.cost.as_ref().unwrap().cache_write, Some(6.25));
    }

    #[test]
    fn lookup_maps_ailoy_ids_to_models_dev_providers() {
        let cat = Catalog::from_data(filter_models_dev(&sample()));
        assert_eq!(
            cat.lookup("anthropic/claude-opus-5").unwrap().name,
            "Claude Opus 5"
        );
        assert_eq!(cat.lookup("x-ai/grok-4.6").unwrap().context, Some(256_000));
        assert!(cat.lookup("google/gemini-2.5-pro").is_none());
        assert!(cat.lookup("no-slash").is_none());
        assert_eq!(cat.models_for("anthropic").len(), 1);
        assert_eq!(
            split_model_id("bedrock/anthropic.claude-opus-5"),
            Some(("bedrock", "anthropic.claude-opus-5"))
        );
        assert_eq!(models_dev_provider("bedrock"), Some("amazon-bedrock"));
    }

    fn model(id: &str, name: &str) -> (String, CatalogModel) {
        (
            id.to_string(),
            CatalogModel {
                id: id.to_string(),
                name: name.to_string(),
                reasoning: true,
                tool_call: true,
                context: Some(200_000),
                output: Some(64_000),
                cost: None,
            },
        )
    }

    #[test]
    fn an_alias_and_the_snapshot_it_points_at_are_one_row() {
        let models: BTreeMap<String, CatalogModel> = [
            model("claude-sonnet-4-5", "Claude Sonnet 4.5 (latest)"),
            model("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
            model("claude-opus-5", "Claude Opus 5"),
        ]
        .into_iter()
        .collect();

        let listed = listed_models(&models);
        assert_eq!(
            listed.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["claude-opus-5", "claude-sonnet-4-5"]
        );
        // And the marker comes off with the row it was distinguishing from.
        assert_eq!(
            listed
                .iter()
                .find(|m| m.id == "claude-sonnet-4-5")
                .unwrap()
                .name,
            "Claude Sonnet 4.5"
        );
    }

    #[test]
    fn a_snapshot_that_says_something_the_alias_does_not_keeps_its_row() {
        let models: BTreeMap<String, CatalogModel> = [
            // OpenAI's naming: the snapshot carries its date, so the two rows do not read
            // as the same model and both stay.
            model("gpt-4o", "GPT-4o"),
            model("gpt-4o-2024-08-06", "GPT-4o (2024-08-06)"),
            // Nor is a smaller model a snapshot of the one whose id it starts with.
            model("gpt-4o-mini", "GPT-4o mini"),
        ]
        .into_iter()
        .collect();

        assert_eq!(listed_models(&models).len(), 3);
        assert!(!is_dated_snapshot_of("gpt-4o-mini", "gpt-4o"));
        assert!(is_dated_snapshot_of("gpt-4o-2024-08-06", "gpt-4o"));
        assert!(is_dated_snapshot_of(
            "claude-haiku-4-5-20251001",
            "claude-haiku-4-5"
        ));
        assert!(!is_dated_snapshot_of(
            "claude-haiku-4-5",
            "claude-haiku-4-5"
        ));
    }

    #[test]
    fn a_marker_stays_when_dropping_it_would_double_a_name() {
        // Nothing was collapsed here — the ids are unrelated — so taking the marker off
        // would leave the list with two rows reading `Gemini Flash`.
        let models: BTreeMap<String, CatalogModel> = [
            model("gemini-flash-latest", "Gemini Flash (latest)"),
            model("gemini-3.5-flash", "Gemini Flash"),
        ]
        .into_iter()
        .collect();

        let listed = listed_models(&models);
        assert_eq!(listed.len(), 2);
        assert!(listed.iter().any(|m| m.name == "Gemini Flash (latest)"));
    }

    #[test]
    fn the_catalog_still_answers_for_a_snapshot_that_is_not_listed() {
        // A session pinned to the dated id keeps its context window and its prices.
        let mut data = CatalogData::default();
        data.providers.insert(
            "anthropic".into(),
            CatalogProvider {
                name: "Anthropic".into(),
                models: [
                    model("claude-sonnet-4-5", "Claude Sonnet 4.5 (latest)"),
                    model("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
                ]
                .into_iter()
                .collect(),
            },
        );
        let cat = Catalog::from_data(data);

        assert_eq!(cat.models_for("anthropic").len(), 1);
        assert_eq!(
            cat.lookup("anthropic/claude-sonnet-4-5-20250929")
                .unwrap()
                .context,
            Some(200_000)
        );
    }

    /// A Bedrock group: the plain id, then the profiles that carry it.
    fn bedrock_group(base: &str, name: &str, prefixes: &[&str]) -> Vec<(String, CatalogModel)> {
        let mut out = vec![model(base, name)];
        for p in prefixes {
            let label = match *p {
                "us-gov" => "GovCloud".to_string(),
                "in" => "India".to_string(),
                "global" => "Global".to_string(),
                other => other.to_uppercase(),
            };
            out.push(model(&format!("{p}.{base}"), &format!("{name} ({label})")));
        }
        out
    }

    #[test]
    fn a_model_is_one_row_reached_through_the_routing_that_was_asked_for() {
        let models: Vec<CatalogModel> = bedrock_group(
            "anthropic.claude-opus-5",
            "Claude Opus 5",
            &["us", "eu", "global"],
        )
        .into_iter()
        .map(|(_, m)| m)
        .collect();

        let global = fold_region_profiles(&models, "global");
        assert_eq!(global.len(), 1);
        assert_eq!(global[0].id, "global.anthropic.claude-opus-5");
        // The parenthesis was about the profile, so the row takes the plain id's name.
        assert_eq!(global[0].name, "Claude Opus 5");

        let eu = fold_region_profiles(&models, "eu");
        assert_eq!(eu[0].id, "eu.anthropic.claude-opus-5");
        assert_eq!(eu[0].name, "Claude Opus 5");
    }

    #[test]
    fn a_routing_the_model_does_not_offer_falls_back_to_the_plain_id() {
        let models: Vec<CatalogModel> =
            bedrock_group("amazon.nova-lite-v1:0", "Nova Lite", &["us", "eu", "apac"])
                .into_iter()
                .map(|(_, m)| m)
                .collect();

        let folded = fold_region_profiles(&models, "global");
        assert_eq!(folded.len(), 1);
        assert_eq!(
            folded[0].id, "amazon.nova-lite-v1:0",
            "Bedrock's own region serves it"
        );
        assert_eq!(folded[0].name, "Nova Lite");
    }

    #[test]
    fn a_model_offered_in_one_region_only_keeps_that_region_in_its_name() {
        // Neither the routing nor a plain id: hiding the row would hide the model, and
        // renaming it would call a US-only profile a global one.
        let models = vec![
            model("us.amazon.nova-premier-v1:0", "Nova Premier (US)").1,
            model("anthropic.claude-opus-5", "Claude Opus 5").1,
        ];
        let folded = fold_region_profiles(&models, "global");
        assert_eq!(folded.len(), 2);
        assert!(
            folded
                .iter()
                .any(|m| m.id == "us.amazon.nova-premier-v1:0" && m.name == "Nova Premier (US)")
        );
    }

    #[test]
    fn a_profile_prefix_is_told_from_a_vendor_by_the_closed_list() {
        assert_eq!(
            region_prefix("us.anthropic.claude-opus-5"),
            Some(("us", "anthropic.claude-opus-5"))
        );
        assert_eq!(
            region_prefix("us-gov.anthropic.claude-opus-5"),
            Some(("us-gov", "anthropic.claude-opus-5"))
        );
        // A vendor-qualified id is not a profile, and neither is a lone token.
        assert_eq!(region_prefix("anthropic.claude-opus-5"), None);
        assert_eq!(region_prefix("global-something"), None);
        assert_eq!(
            region_prefix("us.something"),
            None,
            "no vendor behind the prefix"
        );
    }

    #[test]
    fn the_routings_on_offer_are_named_the_way_the_catalog_names_them() {
        let models: Vec<CatalogModel> = bedrock_group(
            "amazon.nova-pro-v1:0",
            "Nova Pro",
            &["us", "in", "us-gov", "global"],
        )
        .into_iter()
        .map(|(_, m)| m)
        .collect();

        let routings = region_routings(&models);
        assert_eq!(
            routings
                .iter()
                .map(|r| (r.id.as_str(), r.label.as_str()))
                .collect::<Vec<_>>(),
            // `REGION_PREFIXES` order, and the label is the provider's own word for it.
            [
                ("global", "Global"),
                ("us", "US"),
                ("us-gov", "GovCloud"),
                ("in", "India"),
            ]
        );
    }

    /// A slice of a real models.dev fetch (2026-09-18): every Anthropic model, and six
    /// Bedrock models through each profile models.dev lists them under. Frozen on purpose —
    /// these tests pin how the folds treat real data, not what models.dev says today.
    const FIXTURE: &str = include_str!("../testdata/catalog.json");

    fn fixture() -> Catalog {
        Catalog::load_from(FIXTURE, None)
    }

    fn stored(fetched_at: Option<i64>, model: &str) -> CatalogData {
        let mut data = filter_models_dev(&sample());
        data.format = FORMAT;
        data.fetched_at = fetched_at;
        data.providers
            .get_mut("anthropic")
            .unwrap()
            .models
            .retain(|k, _| k == model);
        data
    }

    fn write(dir: &Path, data: &CatalogData) -> std::path::PathBuf {
        let p = dir.join("models.json");
        std::fs::write(&p, serde_json::to_vec(data).unwrap()).unwrap();
        p
    }

    #[test]
    fn the_later_fetch_wins_whichever_side_it_is_on() {
        let dir = tempfile::tempdir().unwrap();
        let old = serde_json::to_string(&stored(Some(1_000), "claude-opus-5")).unwrap();
        let new = stored(Some(2_000), "claude-3-embed");
        let cache = write(dir.path(), &new);
        // The cache is newer: an install that fetched after its build.
        let cat = Catalog::load_from(&old, Some(&cache));
        assert_eq!(cat.status().fetched_at, Some(2_000));
        // The build is newer: an update over a cache nothing has refreshed in months.
        let cache = write(dir.path(), &stored(Some(500), "claude-3-embed"));
        let cat = Catalog::load_from(&old, Some(&cache));
        assert_eq!(cat.status().fetched_at, Some(1_000));
        assert!(cat.lookup("anthropic/claude-opus-5").is_some());
    }

    #[test]
    fn a_catalog_from_another_format_or_with_nothing_in_it_is_passed_over() {
        let dir = tempfile::tempdir().unwrap();
        let mut foreign = stored(Some(9_000), "claude-opus-5");
        foreign.format = FORMAT + 1;
        let cache = write(dir.path(), &foreign);
        assert_eq!(Catalog::load_from("", Some(&cache)).status().models, 0);

        // A cache from before timestamps and formats existed reads as format 0.
        std::fs::write(
            &cache,
            r#"{"providers":{"anthropic":{"name":"A","models":{}}}}"#,
        )
        .unwrap();
        assert_eq!(
            Catalog::load_from("", Some(&cache)).status(),
            CatalogStatus::default()
        );

        let empty = stored(Some(9_000), "none-of-them");
        let cache = write(
            dir.path(),
            &CatalogData {
                providers: BTreeMap::new(),
                ..empty
            },
        );
        let embedded = serde_json::to_string(&stored(Some(1), "claude-opus-5")).unwrap();
        assert_eq!(
            Catalog::load_from(&embedded, Some(&cache))
                .status()
                .fetched_at,
            Some(1)
        );
    }

    #[test]
    fn a_build_without_a_snapshot_starts_empty() {
        let cat = Catalog::load_from("", None);
        assert_eq!(cat.status(), CatalogStatus::default());
        assert!(cat.models_for("anthropic").is_empty());
        assert_eq!(cat.age(), None);
    }

    #[test]
    fn the_embedded_snapshot_when_there_is_one_is_this_builds_format() {
        // Empty in a clone and in CI; `npm run catalog` fills it before a release build. A
        // leftover from before `FORMAT` would be ignored at runtime, and this says so here.
        if !EMBEDDED.trim().is_empty() {
            let data: CatalogData = serde_json::from_str(EMBEDDED).expect("gen-catalog writes it");
            assert_eq!(
                data.format, FORMAT,
                "rerun `npm run catalog` in apps/desktop"
            );
            assert!(data.fetched_at.is_some());
        }
    }

    #[test]
    fn the_loop_fetches_when_the_list_is_old_and_retries_sooner_after_a_failure() {
        let hour = Duration::from_secs(3600);
        assert!(is_due(None));
        assert!(!is_due(Some(hour)));
        assert!(is_due(Some(REFRESH_EVERY)));
        assert_eq!(
            next_check(true, Some(hour), false),
            Some(REFRESH_EVERY - hour)
        );
        assert_eq!(next_check(true, Some(hour), true), Some(RETRY_AFTER));
        // Nothing fetched yet and nothing just failed can only be a list that never loaded.
        assert_eq!(next_check(true, None, false), Some(RETRY_AFTER));
        assert_eq!(next_check(false, None, true), None);
    }

    #[tokio::test]
    async fn a_failed_refresh_keeps_the_list_and_says_why() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache/models.json");
        let cat = fixture();
        let before = cat.status();
        let rx = cat.subscribe();
        let failed = cat
            .refresh_from(&cache, async { anyhow::bail!("models.dev is unreachable") })
            .await;
        assert!(failed.is_err());
        let after = cat.status();
        assert_eq!(after.error.as_deref(), Some("models.dev is unreachable"));
        assert!(!after.refreshing);
        assert_eq!(
            (after.fetched_at, after.models),
            (before.fetched_at, before.models)
        );
        assert!(cat.lookup("anthropic/claude-opus-5").is_some());
        assert!(rx.has_changed().unwrap(), "the window hears about it");
        assert!(!cache.exists());

        // The next success replaces the list, writes the cache, and clears the error.
        let fresh = stored(Some(before.fetched_at.unwrap() + 1), "claude-3-embed");
        cat.refresh_from(&cache, async { Ok(fresh) }).await.unwrap();
        let after = cat.status();
        assert_eq!(after.error, None);
        assert_eq!(after.models, 1);
        assert!(cat.lookup("anthropic/claude-opus-5").is_none());
        assert_eq!(Catalog::load_from("", Some(&cache)).status().models, 1);
    }

    #[test]
    fn the_fixture_offers_each_bedrock_model_once() {
        let all = fixture().models_for("bedrock");
        let folded = fold_region_profiles(&all, "global");
        assert!(
            all.len() > folded.len() * 2,
            "the snapshot lists {} bedrock rows; folding left {}",
            all.len(),
            folded.len()
        );
        // One row per model, so no two rows are the same model through two profiles.
        let mut bases: Vec<&str> = folded.iter().map(|m| base_id(&m.id)).collect();
        bases.sort_unstable();
        let before = bases.len();
        bases.dedup();
        assert_eq!(bases.len(), before);
        assert!(
            !region_routings(&all).is_empty(),
            "and the pane has profiles to offer"
        );
    }

    #[test]
    fn the_fixture_offers_each_claude_4_5_once() {
        let listed = fixture().models_for("anthropic");
        let dated: Vec<&str> = listed
            .iter()
            .map(|m| m.id.as_str())
            .filter(|id| {
                id.starts_with("claude-")
                    && id
                        .rsplit('-')
                        .next()
                        .is_some_and(|t| t.len() == 8 && t.chars().all(|c| c.is_ascii_digit()))
            })
            .collect();
        assert!(
            dated.is_empty(),
            "alias rows should have swallowed {dated:?}"
        );
        assert!(
            listed.iter().all(|m| !m.name.ends_with(LATEST_MARKER)),
            "no row should still be marked as the latest of something"
        );
    }

    #[test]
    fn the_fixture_parses_and_has_anthropic() {
        let cat = fixture();
        assert!(cat.lookup("anthropic/claude-opus-5").is_some());
        assert!(cat.status().fetched_at.is_some());
    }
}
