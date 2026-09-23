//! Settings, and the one side effect they have: which model providers ailoy can call.

use ailoy::lang_model::{BedrockRegion, LangModelProvider, get_lm_providers_mut};

use crate::{
    catalog::split_model_id,
    error::{EngineError, Result},
    store::Store,
    types::{ProviderSetting, Settings, SettingsPatch},
};

/// Every test in this crate that reads or mutates ailoy's process-wide `"default"`
/// registry holds this while it does, so tests in the same binary do not race on it.
#[cfg(test)]
pub(crate) static REGISTRY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct ProviderDef {
    pub key: &'static str,
    pub label: &'static str,
    pub ailoy_prefix: &'static str,
    pub pattern: &'static str,
}

pub const PROVIDERS: &[ProviderDef] = &[
    ProviderDef {
        key: "anthropic",
        label: "Anthropic",
        ailoy_prefix: "anthropic",
        pattern: "anthropic/*",
    },
    ProviderDef {
        key: "openai",
        label: "OpenAI",
        ailoy_prefix: "openai",
        pattern: "openai/*",
    },
    ProviderDef {
        key: "google",
        label: "Google Gemini",
        ailoy_prefix: "google",
        pattern: "google/*",
    },
    ProviderDef {
        key: "xai",
        label: "xAI",
        ailoy_prefix: "x-ai",
        pattern: "x-ai/*",
    },
    ProviderDef {
        key: "deepseek",
        label: "DeepSeek",
        ailoy_prefix: "deepseek",
        pattern: "deepseek/*",
    },
    ProviderDef {
        key: "moonshotai",
        label: "Moonshot Kimi",
        ailoy_prefix: "moonshotai",
        pattern: "moonshotai/*",
    },
    ProviderDef {
        key: "bedrock",
        label: "Amazon Bedrock",
        ailoy_prefix: "bedrock",
        pattern: "bedrock/*",
    },
];

pub const BEDROCK_REGION_KEY: &str = "provider.bedrock.region";
pub const BEDROCK_ROUTING_KEY: &str = "provider.bedrock.routing";
/// Where a Bedrock call goes when the user has not said. `us-east-1` has every model
/// Bedrock offers on day one, which is the only defensible guess to make for someone.
pub const DEFAULT_BEDROCK_REGION: &str = "us-east-1";
/// Global routing by default: Bedrock sends the call wherever there is capacity, which is
/// the choice a user who has not thought about data residency wants. Picking a region in
/// settings is what says otherwise.
pub const DEFAULT_BEDROCK_ROUTING: &str = "global";
pub const DEFAULT_MODEL: &str = "anthropic/claude-opus-5";
pub const DEFAULT_MAX_TOKENS: u64 = 32_000;
pub const DEFAULT_MAX_TURNS: u32 = 50;

pub fn setting_key(provider_key: &str) -> String {
    format!("provider.{provider_key}.api_key")
}

pub fn key_hint(key: &str) -> String {
    let tail: String = key
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("…{tail}")
}

fn provider(key: &str) -> Option<&'static ProviderDef> {
    PROVIDERS.iter().find(|p| p.key == key)
}

/// Register every provider that has a key, drop every one that does not, in ailoy's
/// process-wide `"default"` registry. Returns the keys now active.
pub fn apply(store: &Store) -> Result<Vec<&'static str>> {
    // Every SQLite read happens here, before the registry guard exists: `Store::with`
    // takes the store mutex, so reading under the write guard would pin the order
    // LM_REGISTRY(write) → STORE_MUTEX on every caller of `apply`.
    let mut stored: Vec<(&'static ProviderDef, Option<String>)> =
        Vec::with_capacity(PROVIDERS.len());
    for def in PROVIDERS {
        let key = store
            .setting_get(&setting_key(def.key))?
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty());
        stored.push((def, key));
    }
    // Read and parsed only when a Bedrock key is actually stored, and still before the
    // registry guard exists.
    //
    // Before the loop, because `write_settings` has rejected an unusable region since the B4
    // fix but a row written before it — or by an older build that knew a region this one does
    // not — would otherwise fail `apply` half-way through, leaving the registry holding
    // whichever providers happened to come before "bedrock".
    //
    // Only when there is a key, because the region is an input to nothing else: a legacy row
    // naming a region this build cannot parse would otherwise make every unrelated
    // `settings_set` — an OpenAI key, a max-turns change — fail on a provider the user never
    // configured, with no way to reach the setting that would clear it.
    let bedrock_key = stored
        .iter()
        .find(|(def, _)| def.key == "bedrock")
        .and_then(|(_, key)| key.as_ref());
    let region: Option<BedrockRegion> = match bedrock_key {
        None => None,
        Some(_) => {
            let stored_region = store
                .setting_get(BEDROCK_REGION_KEY)?
                .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string());
            Some(stored_region.parse().map_err(|_| {
                EngineError::Invalid(format!("Unsupported Bedrock region: {stored_region}"))
            })?)
        }
    };

    let mut active = Vec::new();
    let mut registry = get_lm_providers_mut();
    // `contains_key` + `insert` rather than `entry().or_insert_with(..)`: clippy reads the
    // latter as `or_default()`, and here the two are not the same call —
    // `LangModelProvider::default()` reads `OPENAI_API_KEY` & co. out of the environment,
    // which would register providers this store never asked for.
    if !registry.contains_key("default") {
        registry.insert("default".to_string(), LangModelProvider::new());
    }
    let default = registry.get_mut("default").expect("just inserted");
    for (def, key) in stored {
        match key {
            None => default.remove(def.pattern),
            Some(k) => {
                let elem = match def.key {
                    "anthropic" => LangModelProvider::anthropic(k),
                    "openai" => LangModelProvider::openai(k),
                    "google" => LangModelProvider::gemini(k),
                    "xai" => LangModelProvider::grok(k),
                    "deepseek" => LangModelProvider::deepseek(k),
                    "moonshotai" => LangModelProvider::kimi(k),
                    // `region` is the `BedrockRegion` parsed above, not the stored string.
                    // It is `Some` exactly when this arm is reachable: the same key that
                    // puts "bedrock" in `stored` with a value is what made it parse.
                    "bedrock" => LangModelProvider::bedrock(
                        region.expect("a stored bedrock key means the region was parsed"),
                        k,
                    ),
                    _ => unreachable!("PROVIDERS is the closed list above"),
                };
                default.insert(def.pattern.to_string(), elem);
                active.push(def.key);
            }
        }
    }
    drop(registry);
    Ok(active)
}

pub fn read_settings(store: &Store) -> Result<Settings> {
    let mut providers = Vec::with_capacity(PROVIDERS.len());
    for def in PROVIDERS {
        let key = store
            .setting_get(&setting_key(def.key))?
            .filter(|k| !k.trim().is_empty());
        providers.push(ProviderSetting {
            key: def.key.into(),
            label: def.label.into(),
            has_key: key.is_some(),
            key_hint: key.as_deref().map(key_hint).unwrap_or_default(),
            // The effective value, not the stored one: unset means `apply` will use the
            // default, and a pane that showed nothing there would be showing something
            // other than where the call is going.
            region: if def.key == "bedrock" {
                Some(
                    store
                        .setting_get(BEDROCK_REGION_KEY)?
                        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.into()),
                )
            } else {
                None
            },
            regions: if def.key == "bedrock" {
                BedrockRegion::ids().iter().map(|r| r.to_string()).collect()
            } else {
                Vec::new()
            },
            routing: if def.key == "bedrock" {
                Some(
                    store
                        .setting_get(BEDROCK_ROUTING_KEY)?
                        .unwrap_or_else(|| DEFAULT_BEDROCK_ROUTING.into()),
                )
            } else {
                None
            },
            // The catalog's, not the store's: `Engine` fills these in, where it has one.
            routings: Vec::new(),
        });
    }
    Ok(Settings {
        providers,
        default_model: store
            .setting_get("default_model")?
            .unwrap_or_else(|| DEFAULT_MODEL.into()),
        max_tokens: store
            .setting_get("max_tokens")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_TOKENS),
        max_turns: store
            .setting_get("max_turns")?
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MAX_TURNS),
        catalog_refresh: store
            .setting_get("catalog_refresh")?
            .map(|v| v == "true")
            .unwrap_or(true),
    })
}

/// Validate the whole patch, then write it. A patch that fails any check leaves the
/// store exactly as it was — half-applied settings are worse than a rejected save.
pub fn write_settings(store: &Store, patch: &SettingsPatch) -> Result<()> {
    // ── validate ────────────────────────────────────────────────────────────
    for key in patch.provider_keys.keys() {
        if provider(key).is_none() {
            return Err(EngineError::Invalid(format!("Unknown provider: {key}")));
        }
    }
    // An empty region (or model) means "clear it"; anything else has to be a value the
    // engine can actually use later.
    let bedrock_region = patch.bedrock_region.as_deref().map(str::trim);
    if let Some(r) = bedrock_region
        && !r.is_empty()
        && r.parse::<BedrockRegion>().is_err()
    {
        return Err(EngineError::Invalid(format!(
            "Unsupported Bedrock region: {r}"
        )));
    }
    // A routing has to be a profile prefix the catalog could actually put in a model id;
    // which of them Bedrock offers for a given model is the catalog's business, and a
    // routing no model has simply falls back — see `catalog::fold_region_profiles`.
    let bedrock_routing = patch.bedrock_routing.as_deref().map(str::trim);
    if let Some(r) = bedrock_routing
        && !r.is_empty()
        && !crate::catalog::REGION_PREFIXES.contains(&r)
    {
        return Err(EngineError::Invalid(format!(
            "Unknown Bedrock routing: {r}"
        )));
    }
    let default_model = patch.default_model.as_deref().map(str::trim);
    if let Some(m) = default_model
        && !m.is_empty()
        && split_model_id(m).is_none()
    {
        return Err(EngineError::Invalid(format!(
            "A model id looks like provider/model: {m}"
        )));
    }

    // ── write ───────────────────────────────────────────────────────────────
    for (key, value) in &patch.provider_keys {
        match value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            Some(v) => store.setting_set(&setting_key(key), v)?,
            None => store.setting_delete(&setting_key(key))?,
        }
    }
    if let Some(r) = bedrock_region {
        if r.is_empty() {
            store.setting_delete(BEDROCK_REGION_KEY)?;
        } else {
            store.setting_set(BEDROCK_REGION_KEY, r)?;
        }
    }
    if let Some(r) = bedrock_routing {
        if r.is_empty() {
            store.setting_delete(BEDROCK_ROUTING_KEY)?;
        } else {
            store.setting_set(BEDROCK_ROUTING_KEY, r)?;
        }
    }
    if let Some(m) = default_model {
        if m.is_empty() {
            store.setting_delete("default_model")?;
        } else {
            store.setting_set("default_model", m)?;
        }
    }
    if let Some(t) = patch.max_tokens {
        store.setting_set("max_tokens", &t.to_string())?;
    }
    if let Some(t) = patch.max_turns {
        store.setting_set("max_turns", &t.to_string())?;
    }
    if let Some(c) = patch.catalog_refresh {
        store.setting_set("catalog_refresh", if c { "true" } else { "false" })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Store;

    #[test]
    fn settings_default_then_patch() {
        let s = Store::open_in_memory().unwrap();
        let settings = read_settings(&s).unwrap();
        assert_eq!(settings.default_model, DEFAULT_MODEL);
        assert_eq!(settings.max_turns, DEFAULT_MAX_TURNS);
        assert!(settings.providers.iter().all(|p| !p.has_key));

        let mut patch = SettingsPatch::default();
        patch
            .provider_keys
            .insert("anthropic".into(), Some("sk-ant-abcdefgh1234".into()));
        patch.default_model = Some("anthropic/claude-sonnet-5".into());
        patch.max_turns = Some(10);
        write_settings(&s, &patch).unwrap();

        let settings = read_settings(&s).unwrap();
        let a = settings
            .providers
            .iter()
            .find(|p| p.key == "anthropic")
            .unwrap();
        assert!(a.has_key);
        assert_eq!(a.key_hint, "…1234");
        assert_eq!(settings.default_model, "anthropic/claude-sonnet-5");
        assert_eq!(settings.max_turns, 10);

        let mut patch = SettingsPatch::default();
        patch.provider_keys.insert("anthropic".into(), None);
        write_settings(&s, &patch).unwrap();
        assert!(
            !read_settings(&s)
                .unwrap()
                .providers
                .iter()
                .find(|p| p.key == "anthropic")
                .unwrap()
                .has_key
        );
    }

    #[test]
    fn unknown_provider_rejects_the_whole_patch() {
        let s = Store::open_in_memory().unwrap();
        let mut patch = SettingsPatch::default();
        // BTreeMap order puts "anthropic" first, so a patch that validates as it writes
        // would have stored this key before reaching the bad one.
        patch
            .provider_keys
            .insert("anthropic".into(), Some("sk-ant-abcdefgh1234".into()));
        patch
            .provider_keys
            .insert("zzz".into(), Some("sk-zzz-0000".into()));
        let err = write_settings(&s, &patch).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(s.setting_get(&setting_key("anthropic")).unwrap(), None);
    }

    fn region_patch(region: &str) -> SettingsPatch {
        SettingsPatch {
            bedrock_region: Some(region.into()),
            ..SettingsPatch::default()
        }
    }

    fn model_patch(model: &str) -> SettingsPatch {
        SettingsPatch {
            default_model: Some(model.into()),
            ..SettingsPatch::default()
        }
    }

    #[test]
    fn bedrock_region_is_validated_then_round_trips() {
        let s = Store::open_in_memory().unwrap();

        write_settings(&s, &region_patch("us-east-1")).unwrap();
        let settings = read_settings(&s).unwrap();
        let bedrock = settings
            .providers
            .iter()
            .find(|p| p.key == "bedrock")
            .unwrap();
        assert_eq!(bedrock.region.as_deref(), Some("us-east-1"));

        let err = write_settings(&s, &region_patch("nowhere-1")).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(
            s.setting_get(BEDROCK_REGION_KEY).unwrap().as_deref(),
            Some("us-east-1"),
            "a rejected patch must not touch the stored region"
        );

        write_settings(&s, &region_patch("  ")).unwrap();
        assert_eq!(s.setting_get(BEDROCK_REGION_KEY).unwrap(), None);
    }

    #[test]
    fn bedrock_routing_defaults_to_global_then_round_trips() {
        let s = Store::open_in_memory().unwrap();
        let bedrock = |s: &Store| {
            read_settings(s)
                .unwrap()
                .providers
                .into_iter()
                .find(|p| p.key == "bedrock")
                .unwrap()
        };

        let b = bedrock(&s);
        assert_eq!(b.routing.as_deref(), Some(DEFAULT_BEDROCK_ROUTING));
        assert_eq!(b.region.as_deref(), Some(DEFAULT_BEDROCK_REGION));
        assert!(
            b.regions.iter().any(|r| r == "us-east-1"),
            "the pane offers the regions ailoy knows"
        );
        // The profiles on offer come from the catalog, which `Engine` holds and this does not.
        assert!(b.routings.is_empty());
        assert!(
            read_settings(&s)
                .unwrap()
                .providers
                .iter()
                .all(|p| p.key == "bedrock" || (p.routing.is_none() && p.regions.is_empty())),
            "one endpoint, nothing to choose"
        );

        let routing = |r: &str| SettingsPatch {
            bedrock_routing: Some(r.into()),
            ..SettingsPatch::default()
        };
        write_settings(&s, &routing("eu")).unwrap();
        assert_eq!(bedrock(&s).routing.as_deref(), Some("eu"));

        let err = write_settings(&s, &routing("mars")).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(
            s.setting_get(BEDROCK_ROUTING_KEY).unwrap().as_deref(),
            Some("eu"),
            "a rejected patch must not touch the stored routing"
        );

        write_settings(&s, &routing("")).unwrap();
        assert_eq!(
            bedrock(&s).routing.as_deref(),
            Some(DEFAULT_BEDROCK_ROUTING)
        );
    }

    #[test]
    fn default_model_is_validated_then_clearable() {
        let s = Store::open_in_memory().unwrap();

        let err = write_settings(&s, &model_patch("no-slash")).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        assert_eq!(s.setting_get("default_model").unwrap(), None);

        write_settings(&s, &model_patch("openai/gpt-5")).unwrap();
        assert_eq!(read_settings(&s).unwrap().default_model, "openai/gpt-5");

        write_settings(&s, &model_patch("")).unwrap();
        assert_eq!(s.setting_get("default_model").unwrap(), None);
        assert_eq!(read_settings(&s).unwrap().default_model, DEFAULT_MODEL);
    }

    #[test]
    fn apply_registers_and_drops_bedrock() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let s = Store::open_in_memory().unwrap();
        s.setting_set(&setting_key("bedrock"), "aws-bedrock-token")
            .unwrap();
        s.setting_set(BEDROCK_REGION_KEY, "us-east-1").unwrap();

        let active = apply(&s).unwrap();
        assert!(active.contains(&"bedrock"), "{active:?}");
        {
            let reg = ailoy::lang_model::get_lm_providers();
            let def = reg.get("default").unwrap();
            assert!(def.get("bedrock/anthropic.claude-opus-5").is_some());
        }

        s.setting_delete(&setting_key("bedrock")).unwrap();
        let active = apply(&s).unwrap();
        assert!(!active.contains(&"bedrock"), "{active:?}");
        let reg = ailoy::lang_model::get_lm_providers();
        let def = reg.get("default").unwrap();
        assert!(def.get("bedrock/anthropic.claude-opus-5").is_none());
    }

    /// The stored region is an input to Bedrock and to nothing else, so an unusable one is
    /// only an error once Bedrock is configured — and when it is, it is an error that lands
    /// before the registry is touched at all.
    #[test]
    fn a_bad_region_only_fails_apply_once_bedrock_has_a_key() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let s = Store::open_in_memory().unwrap();
        s.setting_set(BEDROCK_REGION_KEY, "nowhere-1").unwrap();
        s.setting_set(&setting_key("openai"), "sk-test-region")
            .unwrap();

        // No bedrock key: the unusable row is never read, so an unrelated provider applies.
        let active = apply(&s).unwrap();
        assert_eq!(active, vec!["openai"]);
        assert!(
            ailoy::lang_model::get_lm_providers()
                .get("default")
                .unwrap()
                .get("openai/gpt-5")
                .is_some()
        );

        // Now it matters. The OpenAI key is withdrawn at the same time, so the registry can
        // say whether `apply` got as far as the loop: the removal it would have done first
        // is the evidence.
        s.setting_set(&setting_key("bedrock"), "aws-bedrock-token")
            .unwrap();
        s.setting_delete(&setting_key("openai")).unwrap();
        let err = apply(&s).unwrap_err();
        assert!(matches!(err, EngineError::Invalid(_)), "{err:?}");
        let reg = ailoy::lang_model::get_lm_providers();
        let def = reg.get("default").unwrap();
        assert!(
            def.get("openai/gpt-5").is_some(),
            "apply reached the registry before failing on the region"
        );
        assert!(def.get("bedrock/anthropic.claude-opus-5").is_none());
    }

    #[test]
    fn apply_registers_only_keyed_providers() {
        let _g = REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let s = Store::open_in_memory().unwrap();
        s.setting_set(&setting_key("openai"), "sk-test").unwrap();
        let active = apply(&s).unwrap();
        assert_eq!(active, vec!["openai"]);
        let reg = ailoy::lang_model::get_lm_providers();
        let def = reg.get("default").unwrap();
        assert!(def.get("openai/gpt-5").is_some());
        assert!(
            def.get("anthropic/claude-opus-5").is_none()
                || std::env::var("ANTHROPIC_API_KEY").is_ok()
        );
    }
}
