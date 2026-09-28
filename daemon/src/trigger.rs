//! A trigger registration: which event types wake it, which of them it produces
//! itself, and which automation it runs.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The body of `PUT /triggers/{name}`. The automation directory's `trigger.json` is a
/// draft of it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TriggerConfig {
    /// Absolute path of the automation directory (console.json, workflow.json, tasks/).
    pub automation: PathBuf,

    /// Events this trigger makes for itself, by type: a clock, a watched directory.
    #[serde(default)]
    pub sources: BTreeMap<String, SourceConfig>,

    /// Event types made elsewhere that also wake this trigger: types posted to
    /// `POST /events/{type}`, or `runs:<trigger>` for another trigger's run results.
    #[serde(default)]
    pub watch: Vec<String>,

    /// Bound on one call of the automation's `trigger.py`, in seconds. The script
    /// runs on the shared console, so a call of one trigger delays the others.
    #[serde(default = "default_script_timeout_secs")]
    pub script_timeout_secs: u64,
}

/// The script deciding whether the next run is due and with what. Absent from the
/// automation directory: the oldest event without a run becomes one, with payload
/// `{ "type", "at", "payload" }`.
pub const TRIGGER_SCRIPT: &str = "trigger.py";

fn default_script_timeout_secs() -> u64 {
    30
}

/// An event source the daemon runs on the trigger's behalf.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceConfig {
    /// Publishes `{ "at": <unix> }` on a schedule. 5 fields, seconds optional.
    Cron { schedule: String },

    /// Publishes `{ "path", "event" }` for changes under a host directory.
    Fs {
        path: PathBuf,
        /// `create`, `modify`, `remove`. Empty means all three.
        #[serde(default)]
        events: Vec<String>,
    },
}

impl TriggerConfig {
    pub fn schema() -> serde_json::Value {
        serde_json::to_value(schemars::schema_for!(TriggerConfig)).expect("schema serializes")
    }

    /// Hash of the canonical JSON, so a draft file and a registration can be compared.
    pub fn hash(&self) -> String {
        let canon = serde_json::to_vec(self).expect("config serializes");
        hex::encode(Sha256::digest(canon))
    }

    /// Every event type that wakes this trigger and that its script may query: its
    /// own sources and what it watches.
    pub fn types(&self) -> Vec<String> {
        let mut types: Vec<String> = self.sources.keys().cloned().collect();
        for w in &self.watch {
            if !types.contains(w) {
                types.push(w.clone());
            }
        }
        types
    }

    /// The automation's `trigger.py`, when it has one. None means identity logic.
    pub fn script_path(&self) -> Option<PathBuf> {
        let path = self.automation.join(TRIGGER_SCRIPT);
        path.is_file().then_some(path)
    }

    /// Everything that makes a registration unusable, all at once.
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if !self.automation.is_absolute() {
            problems.push(format!(
                "automation: {} is not an absolute path",
                self.automation.display()
            ));
        } else if let Err(e) = ailoy::automation::AutomationDef::load(&self.automation) {
            problems.push(format!("automation: {e}"));
        }
        if self.sources.is_empty() && self.watch.is_empty() {
            problems.push("sources or watch: at least one event type is required".into());
        }
        for (name, source) in &self.sources {
            if name.is_empty() || name.contains(':') {
                problems.push(format!(
                    "sources: `{name}` must be a plain name without `:`"
                ));
            }
            match source {
                SourceConfig::Cron { schedule } => {
                    if let Err(e) = crate::sources::parse_cron(schedule) {
                        problems.push(format!("sources.{name}.schedule: {e}"));
                    }
                }
                SourceConfig::Fs { path, events } => {
                    if !path.is_dir() {
                        problems.push(format!(
                            "sources.{name}.path: {} is not a directory",
                            path.display()
                        ));
                    }
                    for e in events {
                        if !matches!(e.as_str(), "create" | "modify" | "remove") {
                            problems.push(format!(
                                "sources.{name}.events: `{e}` is not create, modify or remove"
                            ));
                        }
                    }
                }
            }
        }
        problems
    }
}

/// The `trigger.json` draft in an automation directory, with `automation` filled in
/// from the directory itself when the file leaves it out or names another path.
pub fn read_draft(dir: &Path, file: Option<&Path>) -> anyhow::Result<TriggerConfig> {
    let dir = std::path::absolute(dir)?;
    let path = file
        .map(Path::to_path_buf)
        .unwrap_or_else(|| dir.join("trigger.json"));
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;
    let mut value: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "automation".into(),
            serde_json::Value::String(dir.to_string_lossy().into_owned()),
        );
    }
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}
