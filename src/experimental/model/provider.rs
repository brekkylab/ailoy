use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// What a model needs from its provider to reach a vendor: a secret, and for Bedrock where
/// to send it. How it goes on the wire (`x-api-key`, `Authorization: Bearer`, a beta
/// header, …) is up to each model implementation.
#[derive(Clone)]
pub enum ProviderEntry {
    /// A long-lived API key issued by the vendor's console.
    ApiKey(String),
    /// An OAuth access token, e.g. from `ant auth print-credentials --access-token` after
    /// `ant auth login` for Anthropic.
    OAuthToken(String),
    /// A Bedrock API key and the AWS region whose runtime endpoint
    /// (`bedrock-runtime.<region>.amazonaws.com`) it calls, e.g. `us-east-1`.
    Bedrock { region: String, api_key: String },
}

impl std::fmt::Debug for ProviderEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("ApiKey(..)"),
            Self::OAuthToken(_) => f.write_str("OAuthToken(..)"),
            Self::Bedrock { region, .. } => f
                .debug_struct("Bedrock")
                .field("region", region)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelProvider {
    inner: HashMap<String, ProviderEntry>,
}

impl ModelProvider {
    /// A provider with no vendor configured.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Sets `vendor`'s entry, replacing any existing one.
    pub fn insert(&mut self, vendor: impl Into<String>, entry: ProviderEntry) {
        self.inner.insert(vendor.into(), entry);
    }

    pub fn with(mut self, vendor: impl Into<String>, entry: ProviderEntry) -> Self {
        self.insert(vendor, entry);
        self
    }

    pub fn remove(&mut self, vendor: &str) -> Option<ProviderEntry> {
        self.inner.remove(vendor)
    }

    pub fn get(&self, vendor: &str) -> Option<&ProviderEntry> {
        self.inner.get(vendor)
    }

    /// [`get`](Self::get), failing when `vendor` is not configured.
    pub fn require(&self, vendor: &str) -> anyhow::Result<&ProviderEntry> {
        self.get(vendor)
            .ok_or_else(|| anyhow::anyhow!("no {vendor} entry in the model provider"))
    }

    /// Reads each known vendor's entry from the environment, in the order the vendor's
    /// official SDKs check them. A vendor with none of its variables set is left out.
    ///
    /// - `anthropic`: `ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN` as an OAuth token.
    ///   Both bill the Console organization they belong to; a claude.ai subscription
    ///   (Pro/Max) login is not one of them, as Anthropic does not allow third-party products
    ///   to offer it without approval.
    /// - `openai`: `OPENAI_API_KEY`.
    /// - `gemini`: `GOOGLE_API_KEY`, then `GEMINI_API_KEY`.
    /// - `openrouter`: `OPENROUTER_API_KEY`.
    /// - `bedrock`: `AWS_BEARER_TOKEN_BEDROCK`, a Bedrock API key, in the region
    ///   `AWS_REGION`, then `AWS_DEFAULT_REGION`, then `us-east-1`, as the AWS SDKs pick it.
    /// - `deepseek`: `DEEPSEEK_API_KEY`.
    /// - `moonshot` (Kimi): `MOONSHOT_API_KEY`, then `KIMI_API_KEY`.
    /// - `zai` (GLM): `ZAI_API_KEY`.
    pub fn from_env() -> Self {
        let mut p = Self::new();
        let anthropic = env_var("ANTHROPIC_API_KEY")
            .map(ProviderEntry::ApiKey)
            .or_else(|| env_var("ANTHROPIC_AUTH_TOKEN").map(ProviderEntry::OAuthToken));
        let api_keys = [
            ("openai", env_var("OPENAI_API_KEY")),
            (
                "gemini",
                env_var("GOOGLE_API_KEY").or_else(|| env_var("GEMINI_API_KEY")),
            ),
            ("openrouter", env_var("OPENROUTER_API_KEY")),
            ("deepseek", env_var("DEEPSEEK_API_KEY")),
            (
                "moonshot",
                env_var("MOONSHOT_API_KEY").or_else(|| env_var("KIMI_API_KEY")),
            ),
            ("zai", env_var("ZAI_API_KEY")),
        ];
        if let Some(c) = anthropic {
            p.insert("anthropic", c);
        }
        if let Some(api_key) = env_var("AWS_BEARER_TOKEN_BEDROCK") {
            let region = env_var("AWS_REGION")
                .or_else(|| env_var("AWS_DEFAULT_REGION"))
                .unwrap_or_else(|| "us-east-1".to_owned());
            p.insert("bedrock", ProviderEntry::Bedrock { region, api_key });
        }
        for (vendor, key) in api_keys {
            if let Some(key) = key {
                p.insert(vendor, ProviderEntry::ApiKey(key));
            }
        }
        p
    }
}

/// Reads an env var, treating blank as unset: a `.env` copied from `.env.example` has empty
/// keys, which would register a vendor that resolves but fails later with a 401.
fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// The `vendor` entry of the provider registered as `provider`, for a request about
/// to be sent; read on every request, so a registry update reaches models already built.
pub(crate) fn find_entry(provider: &str, vendor: &str) -> anyhow::Result<ProviderEntry> {
    let providers = get_model_providers();
    let provider = providers
        .get(provider)
        .ok_or_else(|| anyhow::anyhow!("model provider {provider:?} is not registered"))?;
    Ok(provider.require(vendor)?.clone())
}

/// Process-wide registry of [`ModelProvider`]s, seeded at first access with a `"default"`
/// entry from [`ModelProvider::from_env`].
static MODEL_PROVIDERS: LazyLock<RwLock<HashMap<String, ModelProvider>>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    map.insert("default".to_string(), ModelProvider::from_env());
    RwLock::new(map)
});

/// Borrow the process-wide [`ModelProvider`] registry for reading.
pub fn get_model_providers() -> RwLockReadGuard<'static, HashMap<String, ModelProvider>> {
    MODEL_PROVIDERS
        .read()
        .expect("model_providers lock poisoned")
}

/// Borrow the process-wide [`ModelProvider`] registry for writing.
pub fn get_model_providers_mut() -> RwLockWriteGuard<'static, HashMap<String, ModelProvider>> {
    MODEL_PROVIDERS
        .write()
        .expect("model_providers lock poisoned")
}
