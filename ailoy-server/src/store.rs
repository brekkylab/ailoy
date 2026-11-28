//! [`FileStore`]: the agents and their sessions as JSON files under one directory.
//!
//! ```text
//! <root>/agents/<name>/agent.json
//! <root>/agents/<name>/sessions/<id>.json
//! ```
//!
//! A name or an id is a path segment as it is, but for each byte other than an ASCII letter, a
//! digit, `-` or `_`, written `%XX`: none is `.` or `..`, nor holds a separator. A file is
//! written whole, to a file beside it that is then renamed over it.

use std::{
    io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use ailoy_core::agent::{AgentPutArgs, AgentStore, SessionRecord};
use anyhow::Context as _;
use futures::future::{BoxFuture, FutureExt as _};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::sync::Mutex;

/// The agents and their sessions, kept as files under a directory, one writer at a time.
pub struct FileStore {
    root: PathBuf,
    /// Held by each change, so that none sees another's half done, as a session kept for an
    /// agent being removed.
    writing: Mutex<()>,
}

/// An agent as its file holds it.
#[derive(Serialize, Deserialize)]
struct AgentFile {
    name: String,
    /// When it was first put, in nanoseconds since the Unix epoch: the agents are listed so.
    created: u128,
    args: AgentPutArgs,
}

/// A session as its file holds it.
#[derive(Serialize, Deserialize)]
struct SessionFile {
    /// When it was started, as [`AgentFile::created`]: an agent's sessions are listed so.
    created: u128,
    session: SessionRecord,
}

impl FileStore {
    /// The store under `root`, made as it is first written.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        FileStore {
            root: root.into(),
            writing: Mutex::new(()),
        }
    }

    /// `f` of the root, on a thread that may block, holding the writer's lock if `writes`.
    fn run<'a, T: Send + 'static>(
        &'a self,
        writes: bool,
        f: impl FnOnce(&Path) -> anyhow::Result<T> + Send + 'static,
    ) -> BoxFuture<'a, anyhow::Result<T>> {
        let root = self.root.clone();
        async move {
            let _writing = match writes {
                true => Some(self.writing.lock().await),
                false => None,
            };
            tokio::task::spawn_blocking(move || f(&root)).await?
        }
        .boxed()
    }
}

/// `segment` as a path segment, as the module's docs say.
fn encode(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' => encoded.push(byte as char),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn agent_dir(root: &Path, name: &str) -> PathBuf {
    root.join("agents").join(encode(name))
}

fn agent_path(root: &Path, name: &str) -> PathBuf {
    agent_dir(root, name).join("agent.json")
}

fn sessions_dir(root: &Path, agent: &str) -> PathBuf {
    agent_dir(root, agent).join("sessions")
}

fn session_path(root: &Path, agent: &str, id: &str) -> PathBuf {
    sessions_dir(root, agent).join(format!("{}.json", encode(id)))
}

fn now() -> u128 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |since| since.as_nanos())
}

/// What the file at `path` holds; `None` where there is none.
fn read<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Option<T>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let value = serde_json::from_slice(&bytes);
            Ok(Some(
                value.with_context(|| format!("reading {}", path.display()))?,
            ))
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

/// Write `value` to the file at `path` whole, its directory made if it is not there.
fn write(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let written = || -> io::Result<()> {
        let dir = path.parent().expect("a file is in a directory");
        std::fs::create_dir_all(dir)?;
        let name = path
            .file_name()
            .expect("a file has a name")
            .to_string_lossy();
        let beside = dir.join(format!(".{name}.tmp"));
        std::fs::write(&beside, serde_json::to_vec_pretty(value)?)?;
        std::fs::rename(&beside, path)
    };
    written().with_context(|| format!("writing {}", path.display()))
}

/// What each `.json` file in `dir` holds; none, where there is no `dir`.
fn read_all<T: DeserializeOwned>(dir: &Path) -> anyhow::Result<Vec<T>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let mut all = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
        if name.is_some_and(|name| name.ends_with(".json") && !name.starts_with('.'))
            && let Some(value) = read(&path)?
        {
            all.push(value);
        }
    }
    Ok(all)
}

fn sessions(root: &Path, agent: &str) -> anyhow::Result<Vec<SessionRecord>> {
    let mut files: Vec<SessionFile> = read_all(&sessions_dir(root, agent))?;
    files.sort_by_key(|file| file.created);
    Ok(files.into_iter().map(|file| file.session).collect())
}

fn agents(root: &Path) -> anyhow::Result<Vec<AgentFile>> {
    let dir = root.join("agents");
    let names = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let mut agents = Vec::new();
    for entry in names {
        if let Some(agent) = read::<AgentFile>(&entry?.path().join("agent.json"))? {
            agents.push(agent);
        }
    }
    agents.sort_by_key(|agent| agent.created);
    Ok(agents)
}

impl AgentStore for FileStore {
    fn put_agent<'a>(
        &'a self,
        name: &'a str,
        args: &'a AgentPutArgs,
    ) -> BoxFuture<'a, anyhow::Result<()>> {
        let (name, args) = (name.to_owned(), args.clone());
        self.run(true, move |root| {
            let path = agent_path(root, &name);
            let created = read::<AgentFile>(&path)?.map_or_else(now, |kept| kept.created);
            write(
                &path,
                &AgentFile {
                    name,
                    created,
                    args,
                },
            )
        })
    }

    fn get_agent<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Option<AgentPutArgs>>> {
        let name = name.to_owned();
        self.run(false, move |root| {
            let kept = read::<AgentFile>(&agent_path(root, &name))?;
            Ok(kept.map(|kept| kept.args))
        })
    }

    fn list_agents(&self) -> BoxFuture<'_, anyhow::Result<Vec<(String, AgentPutArgs)>>> {
        self.run(false, |root| {
            let agents = agents(root)?.into_iter();
            Ok(agents.map(|agent| (agent.name, agent.args)).collect())
        })
    }

    fn remove_agent<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Vec<SessionRecord>>> {
        let name = name.to_owned();
        self.run(true, move |root| {
            let removed = sessions(root, &name)?;
            let dir = agent_dir(root, &name);
            match std::fs::remove_dir_all(&dir) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => {
                    Err(e).with_context(|| format!("removing {}", dir.display()))?
                }
                _ => Ok(removed),
            }
        })
    }

    fn create_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, anyhow::Result<bool>> {
        let (agent, session) = (agent.to_owned(), session.clone());
        self.run(true, move |root| {
            if read::<AgentFile>(&agent_path(root, &agent))?.is_none() {
                return Ok(false);
            }
            let path = session_path(root, &agent, &session.state.id);
            let created = now();
            write(&path, &SessionFile { created, session })?;
            Ok(true)
        })
    }

    fn update_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, anyhow::Result<bool>> {
        let (agent, session) = (agent.to_owned(), session.clone());
        self.run(true, move |root| {
            let path = session_path(root, &agent, &session.state.id);
            let Some(kept) = read::<SessionFile>(&path)? else {
                return Ok(false);
            };
            let created = kept.created;
            write(&path, &SessionFile { created, session })?;
            Ok(true)
        })
    }

    fn get_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Option<SessionRecord>>> {
        let path = session_path(&self.root, agent, id);
        self.run(false, move |_| {
            Ok(read::<SessionFile>(&path)?.map(|kept| kept.session))
        })
    }

    fn list_sessions<'a>(
        &'a self,
        agent: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Option<Vec<SessionRecord>>>> {
        let agent = agent.to_owned();
        self.run(false, move |root| {
            if read::<AgentFile>(&agent_path(root, &agent))?.is_none() {
                return Ok(None);
            }
            Ok(Some(sessions(root, &agent)?))
        })
    }

    fn remove_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Option<SessionRecord>>> {
        let path = session_path(&self.root, agent, id);
        self.run(true, move |_| {
            let Some(kept) = read::<SessionFile>(&path)? else {
                return Ok(None);
            };
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
            Ok(Some(kept.session))
        })
    }

    fn find_grant<'a>(
        &'a self,
        grant_hash: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<Option<(String, String)>>> {
        let grant_hash = grant_hash.to_owned();
        self.run(false, move |root| {
            for agent in agents(root)? {
                for session in sessions(root, &agent.name)? {
                    if session.grant_hash.as_deref() == Some(&grant_hash) {
                        return Ok(Some((agent.name, session.state.id)));
                    }
                }
            }
            Ok(None)
        })
    }
}

#[cfg(test)]
mod tests {
    use ailoy_core::agent::{AgentSpec, AgentState};

    use super::*;

    fn store() -> (tempfile::TempDir, FileStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(dir.path());
        (dir, store)
    }

    #[tokio::test]
    async fn test_agents_are_kept_in_the_order_first_put_and_go_with_their_sessions() {
        let (_dir, store) = store();
        for (name, model) in [("b", "openai"), ("a/../x", "anthropic"), ("b", "google")] {
            let args = AgentPutArgs::new(AgentSpec::new(model));
            store.put_agent(name, &args).await.unwrap();
        }
        let agents = store.list_agents().await.unwrap();
        let listed: Vec<_> = (agents.iter())
            .map(|(name, args)| (name.as_str(), args.spec.model.as_str()))
            .collect();
        assert_eq!(listed, [("b", "google"), ("a/../x", "anthropic")]);

        let session = SessionRecord::new(AgentState::new());
        assert!(store.create_session("a/../x", &session).await.unwrap());
        let removed = store.remove_agent("a/../x").await.unwrap();
        assert_eq!(removed[0].state.id, session.state.id);
        assert!(store.get_agent("a/../x").await.unwrap().is_none());
        assert!(store.list_sessions("a/../x").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_a_session_is_kept_only_where_its_agent_and_it_are() {
        let (_dir, store) = store();
        let mut session = SessionRecord::new(AgentState::new());
        assert!(!store.create_session("a", &session).await.unwrap());
        assert!(!store.update_session("a", &session).await.unwrap());

        let args = AgentPutArgs::new(AgentSpec::new("openai"));
        store.put_agent("a", &args).await.unwrap();
        assert!(store.create_session("a", &session).await.unwrap());
        let grant = session.grant();
        assert!(store.update_session("a", &session).await.unwrap());

        let hash = SessionRecord::hash_grant(&grant);
        let found = store.find_grant(&hash).await.unwrap();
        assert_eq!(found, Some(("a".to_owned(), session.state.id.clone())));
        let id = session.state.id.clone();
        assert!(store.remove_session("a", &id).await.unwrap().is_some());
        assert!(!store.update_session("a", &session).await.unwrap());
        assert!(store.find_grant(&hash).await.unwrap().is_none());
    }
}
