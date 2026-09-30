//! The app's persistent directory: `AILOY_CACHE` if set, else `~/.cache/ailoy`.
//!
//! ```text
//! contexts/{id}/             mounted as it stands, so nothing of ours goes inside
//! contexts/{id}.json         its name, beside it
//! agents/{id}/agent.json     see `agent.rs`
//!   … context/, agentmaker/  the helpers, kept out of the collection
//! messages/{id}/chat.json
//! ```

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::context::Context;

pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Resolves the cache directory, creating it and its collections if missing.
    /// Canonicalized so every path handed out is absolute, as cortex requires of mount points.
    pub fn open() -> io::Result<Self> {
        let root = match std::env::var_os("AILOY_CACHE").filter(|v| !v.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => std::env::home_dir()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no home directory"))?
                .join(".cache")
                .join("ailoy"),
        };
        for dir in ["contexts", "agents", "messages"] {
            fs::create_dir_all(root.join(dir))?;
        }
        Ok(Self {
            root: root.canonicalize()?,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Each context's tree, with its `.json` beside it.
    pub fn contexts_dir(&self) -> PathBuf {
        self.root.join("contexts")
    }

    /// Every context: the default first, then by name. Unreadable ones are skipped.
    pub fn contexts(&self) -> Vec<Context> {
        let entries = match fs::read_dir(self.contexts_dir()) {
            Ok(entries) => entries,
            Err(err) => {
                eprintln!("could not list contexts: {err}");
                return Vec::new();
            }
        };
        let mut contexts: Vec<Context> = entries
            .filter_map(|entry| {
                let dir = entry.ok()?.path();
                Context::load(&dir)
                    .inspect_err(|err| eprintln!("skipping {}: {err}", dir.display()))
                    .ok()
                    .flatten()
            })
            .collect();
        contexts.sort_by(|a, b| {
            (!a.is_default())
                .cmp(&!b.is_default())
                .then_with(|| a.name.cmp(&b.name))
                .then_with(|| a.id.cmp(&b.id))
        });
        contexts
    }

    /// The context `id`, if any. An id that is not [`plain_id`] yields `None`, since it
    /// comes from the frontend.
    pub fn context(&self, id: &str) -> Option<Context> {
        if !plain_id(id) {
            return None;
        }
        Context::load(&self.contexts_dir().join(id))
            .inspect_err(|err| eprintln!("could not read context {id}: {err}"))
            .ok()
            .flatten()
    }

    pub fn create_context(&self, name: &str) -> io::Result<Context> {
        Context::create(&self.contexts_dir(), name)
    }

    pub fn agents(&self) -> PathBuf {
        self.root.join("agents")
    }

    pub fn messages(&self) -> PathBuf {
        self.root.join("messages")
    }
}

/// Whether a frontend-supplied `id` is safe as a directory name: only ASCII letters,
/// digits, `-` and `_`, so it cannot escape its collection.
pub fn plain_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}
