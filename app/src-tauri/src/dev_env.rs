//! Dev-only API keys from a `.env`, so a `tauri dev` session starts with its providers set.
//!
//! Debug builds only: a release build never reads a `.env` and the command answers empty.
//! The variable names are the ones ailoy's own `LangModelProvider::default()` reads, so one
//! `.env` serves both.

use std::collections::BTreeMap;

use serde::Serialize;

/// Provider key (as the frontend names it) → the environment variable holding its key.
const KEYS: &[(&str, &str)] = &[
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("google", "GEMINI_API_KEY"),
    ("bedrock", "AWS_BEARER_TOKEN_BEDROCK"),
];

#[derive(Serialize, Default)]
pub struct DevEnvKeys {
    keys: BTreeMap<String, String>,
    bedrock_region: Option<String>,
}

/// Loads the nearest `.env` above this crate — `app/.env` first, then the repo root's.
/// Variables already set in the shell win over the file.
pub fn load() {
    if !cfg!(debug_assertions) {
        return;
    }
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    if let Some(path) = here
        .ancestors()
        .map(|d| d.join(".env"))
        .find(|p| p.is_file())
    {
        if let Err(err) = dotenvy::from_path(&path) {
            eprintln!("could not read {}: {err}", path.display());
        }
    }
}

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

#[tauri::command]
pub fn dev_env_keys() -> DevEnvKeys {
    if !cfg!(debug_assertions) {
        return DevEnvKeys::default();
    }
    DevEnvKeys {
        keys: KEYS
            .iter()
            .filter_map(|(provider, name)| var(name).map(|v| (provider.to_string(), v)))
            .collect(),
        // Same precedence as ailoy and the AWS SDKs.
        bedrock_region: var("AWS_REGION").or_else(|| var("AWS_DEFAULT_REGION")),
    }
}
