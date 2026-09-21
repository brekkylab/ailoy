//! The values that cross the IPC boundary. Every type here is what a Tauri command
//! hands the webview or takes from it, so all of them are `Clone + Debug + Serde`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Which backend stands behind a mount, without the credentials that reach it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MountKind {
    Local,
    Notion,
    S3,
}

/// A mount's backend and everything needed to open it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum MountConfig {
    Root,
    Local {
        host_root: PathBuf,
    },
    Notion {
        api_key: String,
    },
    S3 {
        bucket: String,
        region: String,
        access_key_id: String,
        secret_access_key: String,
        endpoint: Option<String>,
        key_prefix: Option<String>,
    },
}

/// Whether a mount is answering, and why not when it is not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum MountStatus {
    Ok,
    Error { message: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MountInfo {
    pub id: String,
    pub path: String,
    pub kind: MountKind,
    pub label: String,
    pub detail: String,
    pub writable: bool,
    pub status: MountStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MountRequest {
    pub path: String,
    pub label: Option<String>,
    pub config: MountConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub model: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub running: bool,
}

/// One message as it sits in the store: the agent's own `Message`, plus where in the
/// run it came from.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredMessage {
    pub seq: i64,
    pub depth: u8,
    pub source_agent: Option<String>,
    pub message: ailoy::message::Message,
    pub usage: Option<ailoy::message::TokenUsage>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderSetting {
    pub key: String,
    pub label: String,
    pub has_key: bool,
    pub key_hint: String,
    pub region: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub providers: Vec<ProviderSetting>,
    pub default_model: String,
    pub max_tokens: u64,
    pub max_turns: u32,
    pub catalog_refresh: bool,
}

/// A partial update to [`Settings`]. Every field is optional so the webview may send
/// only what changed.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SettingsPatch {
    pub provider_keys: BTreeMap<String, Option<String>>,
    pub bedrock_region: Option<String>,
    pub default_model: Option<String>,
    pub max_tokens: Option<u64>,
    pub max_turns: Option<u32>,
    pub catalog_refresh: Option<bool>,
}

/// USD per 1M tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelCost {
    pub input: Option<f64>,
    pub output: Option<f64>,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub provider: String,
    pub name: String,
    pub context: Option<u64>,
    pub output: Option<u64>,
    pub cost: Option<ModelCost>,
    pub reasoning: bool,
    pub tool_call: bool,
    pub available: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    pub estimated_cost_usd: Option<f64>,
    pub context_used: Option<u64>,
    pub context_limit: Option<u64>,
}

/// Whether the workspace is where a kernel answers, and why not when it is not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum WorkspaceStatus {
    Mounted,
    Degraded { reason: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub mountpoint: PathBuf,
    pub files_root: PathBuf,
    pub status: WorkspaceStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub name: String,
    pub path: String,
    pub kind: String,
    pub size: Option<u64>,
    pub mtime_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileContent {
    pub path: String,
    pub text: Option<String>,
    pub size: u64,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportReport {
    pub files: usize,
    pub bytes: u64,
    pub skipped: Vec<String>,
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or_default()
}

impl MountConfig {
    /// A copy safe to show: secrets replaced by their last four characters.
    pub fn masked(&self) -> MountConfig {
        fn hint(s: &str) -> String {
            let tail: String = s
                .chars()
                .rev()
                .take(4)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            format!("…{tail}")
        }
        match self {
            MountConfig::Notion { api_key } => MountConfig::Notion {
                api_key: hint(api_key),
            },
            MountConfig::S3 {
                bucket,
                region,
                access_key_id,
                secret_access_key,
                endpoint,
                key_prefix,
            } => MountConfig::S3 {
                bucket: bucket.clone(),
                region: region.clone(),
                access_key_id: hint(access_key_id),
                secret_access_key: hint(secret_access_key),
                endpoint: endpoint.clone(),
                key_prefix: key_prefix.clone(),
            },
            other => other.clone(),
        }
    }
}
