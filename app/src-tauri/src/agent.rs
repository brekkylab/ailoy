//! An agent: one JSON document at `agents/{id}/agent.json`.
//!
//! The collection is what this module has opinions about — the id names the directory,
//! `createdAt` orders the list, and at most one agent is the default. Everything else the
//! editor writes — the model, the prompt, the tools, the sandbox — rides through `rest`
//! untouched: it is the app's shape and not ours, so a field added there needs no change
//! here and cannot be dropped on a round trip by a struct that has not heard of it yet.
//!
//! A few agents live beside the collection rather than in it: the [`HELPERS`], one behind
//! the helper pane of each tab that has one — `context` changes a context's files,
//! `agentmaker` builds agents. They are the same kind of document in the same directory,
//! but [`list`] leaves them out and [`save`] and [`remove`] refuse them, so the Agent tab
//! never shows them and nothing there can pick one as a sub-agent or a default. Their
//! settings are edited in their files.

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

/// The helpers, by id — also the name of each one's directory — with what each starts with.
/// None names a `context`: each works on whatever its tab has open.
pub const HELPERS: &[(&str, &str)] = &[
    (
        "context",
        r#"{
  "id": "context",
  "name": "Context agent",
  "model": "anthropic/claude-sonnet-5",
  "description": "Organizes the files of the context open in the Context tab.",
  "instruction": "You work on one context: a directory of files, mounted at your working directory for you to change. Do what is asked of its files — sort, rename, convert, summarize, write new ones — and nothing beyond it.\n\nBefore you change anything, read `README.md` at the root if there is one: it says how the context is laid out. Keep it true to what you leave behind.\n\nNever delete or overwrite a file unless you were asked to. When you are done, list what you changed, one line per file.\n\nAnswer concisely, in the language the request was made in."
}"#,
    ),
    (
        "agentmaker",
        r#"{
  "id": "agentmaker",
  "name": "agentmaker",
  "model": "anthropic/claude-sonnet-5",
  "description": "Builds and edits the agents in the Agent tab.",
  "instruction": "You help build agents for Ailoy. Each agent is one JSON document, `{id}/agent.json`, in the directory mounted at your working directory; the one open in the Agent tab is named in the request.\n\nA document has `name`, `description`, `model` (`provider/model`, e.g. `anthropic/claude-sonnet-5`), `instruction` (the system prompt), `engines` (web search engines; empty means all), `mcp` (servers: `name`, `transport` `http` or `stdio`, `target`), `subagents` (ids of other agents), `context` (the id of a context mounted read-only, or null), `options` (`temperature`, `topP`, `topK`, `maxTokens`, as strings) and `sandbox` (`base` image and `steps`). Leave `id`, `default`, `createdAt` and `updatedAt` as they are.\n\nChange only what was asked. Write instructions that are short, concrete and in the second person. When you are done, say what you changed, field by field.\n\nAnswer concisely, in the language the request was made in."
}"#,
    ),
];

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

fn is_helper(id: &str) -> bool {
    HELPERS.iter().any(|&(helper, _)| helper == id)
}

fn refuse_helper(id: &str) -> io::Result<()> {
    if is_helper(id) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{id} is a helper, kept in its own file and not in the collection"),
        ));
    }
    Ok(())
}

/// Every agent under `dir` but the helpers, oldest first. One that cannot be read is
/// left out, not fatal.
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
            let entry = entry.ok()?;
            if entry.file_name().to_str().is_some_and(is_helper) {
                return None;
            }
            let path = entry.path().join("agent.json");
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
    refuse_helper(&agent.id)?;
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
    refuse_helper(id)?;
    match fs::remove_dir_all(dir.join(id)) {
        Err(err) if err.kind() != io::ErrorKind::NotFound => Err(err),
        _ => Ok(()),
    }
}

/// Writes the seed agent when the collection is empty, and each helper whose file is
/// missing — only then, so a start never overwrites what was edited.
pub fn seed(dir: &Path) -> io::Result<()> {
    for &(id, _) in HELPERS {
        helper(dir, id)?;
    }
    if !list(dir).is_empty() {
        return Ok(());
    }
    let agent: Agent = serde_json::from_str(SEED)?;
    save(dir, agent).map(|_| ())
}

/// The helper `id`, written from its seed first if its file has gone missing.
pub fn helper(dir: &Path, id: &str) -> io::Result<Agent> {
    let Some(&(_, seed)) = HELPERS.iter().find(|&&(helper, _)| helper == id) else {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("no helper {id:?}"),
        ));
    };
    if let Some(agent) = load(&file(dir, id))? {
        return Ok(agent);
    }
    let mut agent: Agent = serde_json::from_str(seed)?;
    agent.created_at = now();
    agent.updated_at = agent.created_at;
    write(dir, &agent)?;
    Ok(agent)
}

#[tauri::command]
pub fn list_agents(cache: State<'_, Cache>) -> Vec<Agent> {
    list(&cache.agents())
}

#[tauri::command]
pub fn get_helper_agent(cache: State<'_, Cache>, id: String) -> Result<Agent, String> {
    helper(&cache.agents(), &id).map_err(|err| err.to_string())
}

#[tauri::command]
pub fn save_agent(cache: State<'_, Cache>, agent: Agent) -> Result<Agent, String> {
    save(&cache.agents(), agent).map_err(|err| err.to_string())
}

#[tauri::command]
pub fn remove_agent(cache: State<'_, Cache>, id: String) -> Result<(), String> {
    remove(&cache.agents(), &id).map_err(|err| err.to_string())
}

/// `agent` — the document as the editor has it, which may be ahead of the file — written
/// to `to`, outside the collection.
#[tauri::command]
pub fn export_agent(agent: Value, to: PathBuf) -> Result<(), String> {
    serde_json::to_vec_pretty(&agent)
        .map_err(io::Error::from)
        .and_then(|bytes| fs::write(&to, bytes))
        .map_err(|err| err.to_string())
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
    fn helpers_are_kept_apart() {
        let dir = scratch("helpers");
        seed(&dir).unwrap();
        // Beside the collection, never in it.
        assert!(list(&dir).iter().all(|a| !is_helper(&a.id)));
        for &(id, _) in HELPERS {
            let agent = helper(&dir, id).unwrap();
            assert_eq!(agent.id, id);
            assert!(!agent.is_default);
            assert!(save(&dir, self::agent(id, false)).is_err());
            assert!(remove(&dir, id).is_err());
        }
        assert!(helper(&dir, "default").is_err());

        // Edited by hand, it stays edited; gone, it comes back from the seed.
        let mut edited = helper(&dir, "context").unwrap();
        edited.name = "Edited".into();
        write(&dir, &edited).unwrap();
        seed(&dir).unwrap();
        assert_eq!(helper(&dir, "context").unwrap().name, "Edited");
        fs::remove_dir_all(dir.join("context")).unwrap();
        assert_eq!(helper(&dir, "context").unwrap().name, "Context agent");
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
