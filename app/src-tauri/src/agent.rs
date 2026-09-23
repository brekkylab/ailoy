//! An agent: one JSON document at `agents/{id}/agent.json`.
//!
//! The collection is what this module has opinions about — the id names the directory,
//! `createdAt` orders the list, and at most one agent is the default. Everything else the
//! editor writes — the model, the prompt, the tools, the sandbox — rides through `rest`
//! untouched: it is the app's shape and not ours, so a field added there needs no change
//! here and cannot be dropped on a round trip by a struct that has not heard of it yet.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::State;

use crate::cache::{Cache, plain_id};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Agent {
    /// Also the name of its directory.
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// The agent a chat starts with when none is named. The collection keeps it to one.
    #[serde(default, rename = "default")]
    pub is_default: bool,
    /// Milliseconds since the epoch. Set once, and kept across every later write; 0 on
    /// the way in means the sender does not know it.
    #[serde(default)]
    pub created_at: u64,
    #[serde(default)]
    pub updated_at: u64,
    /// The editor's own fields, carried through as they arrived.
    #[serde(flatten)]
    pub rest: Map<String, Value>,
}

/// What the collection starts with, so a chat has something to answer with before anyone
/// has opened the Agent tab. It reads the default context (`context::DEFAULT_ID`).
const SEED: &str = r#"{
  "id": "default",
  "name": "Default",
  "default": true,
  "model": "anthropic/claude-sonnet-5",
  "description": "General-purpose agent over your contexts.",
  "context": "default",
  "instruction": "Answer concisely, in the language the question was asked in.\n\nBefore you touch the files in a context, read `README.md` at its root if there is one: it says how the context is laid out and how its files are to be read."
}"#;

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

fn file(dir: &Path, id: &str) -> PathBuf {
    dir.join(id).join("agent.json")
}

fn load(path: &Path) -> io::Result<Option<Agent>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(err),
    }
}

/// Beside the target and renamed over it, so a reader never sees half a file.
fn write(dir: &Path, agent: &Agent) -> io::Result<()> {
    let path = file(dir, &agent.id);
    fs::create_dir_all(path.parent().expect("a file under a directory"))?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(agent)?)?;
    fs::rename(&tmp, &path)
}

/// Every agent under `dir`, oldest first. One that cannot be read is left out, not fatal.
pub fn list(dir: &Path) -> Vec<Agent> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) => {
            eprintln!("could not list agents: {err}");
            return Vec::new();
        }
    };
    let mut agents: Vec<Agent> = entries
        .filter_map(|entry| {
            let path = entry.ok()?.path().join("agent.json");
            load(&path)
                .inspect_err(|err| eprintln!("skipping {}: {err}", path.display()))
                .ok()
                .flatten()
        })
        .collect();
    agents.sort_by(|a, b| {
        a.created_at
            .cmp(&b.created_at)
            .then_with(|| a.id.cmp(&b.id))
    });
    agents
}

/// Writes `agent` under its id, creating it if there is none yet. Made the default, it
/// takes the flag off whoever had it rather than being refused: that is the only thing
/// sending it could mean.
pub fn save(dir: &Path, mut agent: Agent) -> io::Result<Agent> {
    if !plain_id(&agent.id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not an agent id: {:?}", agent.id),
        ));
    }
    let existing = load(&file(dir, &agent.id))?;
    let now = now();
    agent.created_at = existing
        .map(|a| a.created_at)
        .filter(|&t| t > 0)
        .or(Some(agent.created_at).filter(|&t| t > 0))
        .unwrap_or(now);
    agent.updated_at = now;
    write(dir, &agent)?;
    if agent.is_default {
        for mut other in list(dir) {
            if other.id != agent.id && other.is_default {
                other.is_default = false;
                other.updated_at = now;
                write(dir, &other)?;
            }
        }
    }
    Ok(agent)
}

/// The agent `id` and its directory. One that is not there is already what was asked.
pub fn remove(dir: &Path, id: &str) -> io::Result<()> {
    if !plain_id(id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("not an agent id: {id:?}"),
        ));
    }
    match fs::remove_dir_all(dir.join(id)) {
        Err(err) if err.kind() != io::ErrorKind::NotFound => Err(err),
        _ => Ok(()),
    }
}

/// Writes the seed agent when the collection is empty — only then, so a start never
/// overwrites what was edited.
pub fn seed(dir: &Path) -> io::Result<()> {
    if !list(dir).is_empty() {
        return Ok(());
    }
    let agent: Agent = serde_json::from_str(SEED)?;
    save(dir, agent).map(|_| ())
}

#[tauri::command]
pub fn list_agents(cache: State<'_, Cache>) -> Vec<Agent> {
    list(&cache.agents())
}

#[tauri::command]
pub fn save_agent(cache: State<'_, Cache>, agent: Agent) -> Result<Agent, String> {
    save(&cache.agents(), agent).map_err(|err| err.to_string())
}

#[tauri::command]
pub fn remove_agent(cache: State<'_, Cache>, id: String) -> Result<(), String> {
    remove(&cache.agents(), &id).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ailoy-agent-{name}-{}", now()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn agent(id: &str, is_default: bool) -> Agent {
        Agent {
            id: id.into(),
            name: id.into(),
            is_default,
            created_at: 0,
            updated_at: 0,
            rest: Map::from_iter([("model".into(), Value::from("openai/gpt-5"))]),
        }
    }

    #[test]
    fn seeds_once_and_keeps_edits() {
        let dir = scratch("seed");
        seed(&dir).unwrap();
        let mut seeded = list(&dir).remove(0);
        assert!(seeded.is_default);
        seeded.name = "Edited".into();
        save(&dir, seeded).unwrap();
        seed(&dir).unwrap();
        assert_eq!(list(&dir).len(), 1);
        assert_eq!(list(&dir)[0].name, "Edited");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn one_default_and_rest_round_trips() {
        let dir = scratch("default");
        let a = save(&dir, agent("a", true)).unwrap();
        save(&dir, agent("b", true)).unwrap();
        let all = list(&dir);
        assert_eq!(all.iter().filter(|x| x.is_default).count(), 1);
        assert!(all.iter().find(|x| x.id == "b").unwrap().is_default);
        let again = all.iter().find(|x| x.id == "a").unwrap();
        assert_eq!(again.created_at, a.created_at);
        assert_eq!(again.rest["model"], "openai/gpt-5");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_ids_that_step_outside() {
        let dir = scratch("ids");
        assert!(save(&dir, agent("../x", false)).is_err());
        assert!(remove(&dir, "..").is_err());
        remove(&dir, "missing").unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
}
