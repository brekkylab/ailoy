use std::sync::Mutex;

use futures::future::{BoxFuture, FutureExt as _};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};

use crate::agent::{AgentPutArgs, AgentState};

/// A session as it is kept: its state, and what reaches it and its sandbox.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRecord {
    pub state: AgentState,

    /// The grant an app reaches the session with, as its [`hash`](Self::hash_grant): the grant
    /// itself is kept nowhere, so what is kept reaches no session.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grant_hash: Option<String>,

    /// The token that reaches the session's sandbox on virtx, for one made with an app's own
    /// credentials: its sandbox is reached again with it, where the server's own client
    /// reaches it not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sandbox_grant: Option<String>,
}

impl SessionRecord {
    pub fn new(state: AgentState) -> Self {
        SessionRecord {
            state,
            grant_hash: None,
            sandbox_grant: None,
        }
    }

    /// Give out a new grant that reaches this session, in place of one given before.
    pub fn grant(&mut self) -> String {
        let grant = uuid::Uuid::new_v4().simple().to_string();
        self.grant_hash = Some(Self::hash_grant(&grant));
        grant
    }

    /// What `grant` is kept and found as: its SHA-256, in hex.
    pub fn hash_grant(grant: &str) -> String {
        let digest = Sha256::digest(grant.as_bytes());
        digest.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// Where an [`AgentProvider`](crate::agent::AgentProvider) keeps the agents put to it, by name,
/// and each one's sessions, by id: in memory ([`MemoryStore`]), in files, or in a database, as
/// the server chooses. An `Err` is the store's own failure, not a refusal: what is not there is
/// `None`, or `false`.
pub trait AgentStore: Send + Sync {
    /// Keep the agent `args` describes under `name`, in place of one already there, whose
    /// sessions stay.
    fn put_agent<'a>(&'a self, name: &'a str, args: &'a AgentPutArgs) -> BoxFuture<'a, Result<()>>;

    /// The agent kept under `name`.
    fn get_agent<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Option<AgentPutArgs>>>;

    /// Every agent kept, in the order each name was first put.
    fn list_agents(&self) -> BoxFuture<'_, Result<Vec<(String, AgentPutArgs)>>>;

    /// Keep the agent `name` no longer, nor its sessions, which are answered, for their
    /// sandboxes to be removed.
    fn remove_agent<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Vec<SessionRecord>>>;

    /// Keep `session` as a new session of the agent `agent`: `false`, keeping nothing, where
    /// there is no such agent.
    fn create_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, Result<bool>>;

    /// Keep `session` in place of the session of its id of the agent `agent`: `false`,
    /// keeping nothing, where there is no such session, as one removed while its turn ran.
    fn update_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, Result<bool>>;

    /// The session `id` of the agent `agent`.
    fn get_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, Result<Option<SessionRecord>>>;

    /// Every session of the agent `agent`, in the order each was started: `None` where there
    /// is no such agent.
    fn list_sessions<'a>(
        &'a self,
        agent: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<SessionRecord>>>>;

    /// Keep the session `id` of the agent `agent` no longer, answering it, for its sandbox to
    /// be removed.
    fn remove_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, Result<Option<SessionRecord>>>;

    /// The session whose grant has the hash `grant_hash`, as its agent and id.
    fn find_grant<'a>(
        &'a self,
        grant_hash: &'a str,
    ) -> BoxFuture<'a, Result<Option<(String, String)>>>;
}

type Result<T> = anyhow::Result<T>;

/// Agents and sessions kept in memory, gone with the process.
#[derive(Default)]
pub struct MemoryStore {
    agents: Mutex<IndexMap<String, Kept>>,
}

/// An agent kept in memory, and its sessions by id.
struct Kept {
    args: AgentPutArgs,
    sessions: IndexMap<String, SessionRecord>,
}

impl MemoryStore {
    /// Answer `f` of the agents, as a future that is ready.
    fn with<'a, T: Send + 'a>(
        &'a self,
        f: impl FnOnce(&mut IndexMap<String, Kept>) -> T,
    ) -> BoxFuture<'a, Result<T>> {
        let answer = f(&mut self.agents.lock().unwrap());
        futures::future::ready(Ok(answer)).boxed()
    }
}

impl AgentStore for MemoryStore {
    fn put_agent<'a>(&'a self, name: &'a str, args: &'a AgentPutArgs) -> BoxFuture<'a, Result<()>> {
        self.with(|agents| match agents.get_mut(name) {
            Some(kept) => kept.args = args.clone(),
            None => {
                let sessions = IndexMap::new();
                agents.insert(
                    name.to_owned(),
                    Kept {
                        args: args.clone(),
                        sessions,
                    },
                );
            }
        })
    }

    fn get_agent<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Option<AgentPutArgs>>> {
        self.with(|agents| agents.get(name).map(|kept| kept.args.clone()))
    }

    fn list_agents(&self) -> BoxFuture<'_, Result<Vec<(String, AgentPutArgs)>>> {
        self.with(|agents| {
            (agents.iter())
                .map(|(name, kept)| (name.clone(), kept.args.clone()))
                .collect()
        })
    }

    fn remove_agent<'a>(&'a self, name: &'a str) -> BoxFuture<'a, Result<Vec<SessionRecord>>> {
        self.with(|agents| match agents.shift_remove(name) {
            Some(kept) => kept.sessions.into_values().collect(),
            None => Vec::new(),
        })
    }

    fn create_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, Result<bool>> {
        self.with(|agents| match agents.get_mut(agent) {
            Some(kept) => {
                kept.sessions
                    .insert(session.state.id.clone(), session.clone());
                true
            }
            None => false,
        })
    }

    fn update_session<'a>(
        &'a self,
        agent: &'a str,
        session: &'a SessionRecord,
    ) -> BoxFuture<'a, Result<bool>> {
        self.with(|agents| {
            let kept =
                (agents.get_mut(agent)).and_then(|kept| kept.sessions.get_mut(&session.state.id));
            match kept {
                Some(kept) => {
                    *kept = session.clone();
                    true
                }
                None => false,
            }
        })
    }

    fn get_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, Result<Option<SessionRecord>>> {
        self.with(|agents| agents.get(agent)?.sessions.get(id).cloned())
    }

    fn list_sessions<'a>(
        &'a self,
        agent: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<SessionRecord>>>> {
        self.with(|agents| Some(agents.get(agent)?.sessions.values().cloned().collect()))
    }

    fn remove_session<'a>(
        &'a self,
        agent: &'a str,
        id: &'a str,
    ) -> BoxFuture<'a, Result<Option<SessionRecord>>> {
        self.with(|agents| agents.get_mut(agent)?.sessions.shift_remove(id))
    }

    fn find_grant<'a>(
        &'a self,
        grant_hash: &'a str,
    ) -> BoxFuture<'a, Result<Option<(String, String)>>> {
        self.with(|agents| {
            agents.iter().find_map(|(name, kept)| {
                let (id, _) = (kept.sessions.iter())
                    .find(|(_, session)| session.grant_hash.as_deref() == Some(grant_hash))?;
                Some((name.clone(), id.clone()))
            })
        })
    }
}
