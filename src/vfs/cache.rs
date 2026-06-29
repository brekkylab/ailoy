//! In-memory metadata index. A `readdir` populates it with each child's
//! type + size; a `stat` reads from it (fast-path hit + negative caching) so
//! the per-entry `getattr` storm the kernel issues after a listing (e.g.
//! `ls -la`) costs no provider round trips. [`CachedResource`] wraps a provider
//! [`Resource`] so both FUSE frontends benefit transparently.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::vfs::{
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

/// Default listing TTL (600s).
const DEFAULT_TTL: Duration = Duration::from_secs(600);

#[derive(Clone, Copy)]
struct Entry {
    is_dir: bool,
    size: u64,
}

#[derive(Default)]
struct Inner {
    /// path -> metadata.
    entries: HashMap<String, Entry>,
    /// directory path -> its child full-paths.
    children: HashMap<String, Vec<String>>,
    /// directory path -> listing expiry.
    expiry: HashMap<String, Instant>,
}

/// Per-mount metadata index. Only directory listings expire (TTL); individual
/// entries live until their listing is overwritten or invalidated.
struct IndexCache {
    inner: Mutex<Inner>,
    ttl: Duration,
}

impl IndexCache {
    fn new(ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            ttl,
        }
    }

    /// Entry metadata for a path, if known.
    fn get(&self, path: &str) -> Option<Entry> {
        self.inner.lock().unwrap().entries.get(path).copied()
    }

    /// Whether `path`'s listing is present and unexpired (basis for negative
    /// caching: a fresh listing means non-members provably don't exist).
    fn is_listed(&self, path: &str) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(inner.expiry.get(path), Some(exp) if *exp > Instant::now())
    }

    /// Reconstruct a directory's entries from the cache if its listing is
    /// fresh; `None` means not listed or expired (caller must hit the network).
    /// Done under one lock so the listing and its entries stay consistent.
    fn list_dir_entries(&self, path: &str) -> Option<Vec<DirEntry>> {
        let inner = self.inner.lock().unwrap();
        match inner.expiry.get(path) {
            Some(exp) if *exp > Instant::now() => {}
            _ => return None,
        }
        let children = inner.children.get(path)?;
        let out = children
            .iter()
            .filter_map(|full| {
                inner.entries.get(full).map(|e| DirEntry {
                    name: basename(full).to_string(),
                    kind: if e.is_dir {
                        FileKind::Dir
                    } else {
                        FileKind::File
                    },
                    size: e.size,
                })
            })
            .collect();
        Some(out)
    }

    /// Record a directory listing: store each child's metadata + the child set
    /// with a fresh TTL.
    fn set_dir(&self, path: &str, entries: &[DirEntry]) {
        let prefix = if path == "/" {
            "/".to_string()
        } else {
            format!("{path}/")
        };
        let mut inner = self.inner.lock().unwrap();
        let mut child_keys = Vec::with_capacity(entries.len());
        for e in entries {
            let full = format!("{prefix}{}", e.name);
            inner.entries.insert(
                full.clone(),
                Entry {
                    is_dir: matches!(e.kind, FileKind::Dir),
                    size: e.size,
                },
            );
            child_keys.push(full);
        }
        inner.children.insert(path.to_string(), child_keys);
        inner.expiry.insert(path.to_string(), Instant::now() + self.ttl);
    }

    fn invalidate_dir(&self, path: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(children) = inner.children.remove(path) {
            for c in children {
                inner.entries.remove(&c);
            }
        }
        inner.expiry.remove(path);
    }

    /// Invalidate the listing of `path`'s parent directory (so a created /
    /// removed / resized child is re-listed).
    fn invalidate_parent(&self, path: &str) {
        self.invalidate_dir(parent_of(path));
    }

    fn clear(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.clear();
        inner.children.clear();
        inner.expiry.clear();
    }
}

/// Wraps a provider [`Resource`] with an [`IndexCache`]: `readdir` fills the
/// cache, `stat` serves from it (and negative-caches misses), mutations
/// invalidate. Read/write payloads are delegated unchanged.
pub struct CachedResource {
    inner: Arc<dyn Resource>,
    cache: IndexCache,
}

impl CachedResource {
    pub fn new(inner: Arc<dyn Resource>) -> Self {
        Self {
            inner,
            cache: IndexCache::new(DEFAULT_TTL),
        }
    }
}

#[async_trait]
impl Resource for CachedResource {
    async fn read_bytes(&self, path: &VPath, range: Option<Range<u64>>) -> anyhow::Result<Vec<u8>> {
        self.inner.read_bytes(path, range).await
    }

    async fn write_bytes(&self, path: &VPath, data: Vec<u8>) -> anyhow::Result<()> {
        let r = self.inner.write_bytes(path, data).await;
        if r.is_ok() {
            self.cache.invalidate_parent(path.as_str());
        }
        r
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        if let Some(entries) = self.cache.list_dir_entries(path.as_str()) {
            return Ok(entries);
        }
        let entries = self.inner.readdir(path).await?;
        self.cache.set_dir(path.as_str(), &entries);
        Ok(entries)
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        let key = path.as_str();
        match self.cache.get(key) {
            Some(e) if e.is_dir => {
                return Ok(FileStat {
                    kind: FileKind::Dir,
                    ..Default::default()
                });
            }
            // A file with a known (>0) size: serve it.
            Some(e) if e.size > 0 => {
                return Ok(FileStat {
                    kind: FileKind::File,
                    size: e.size,
                    ..Default::default()
                });
            }
            // A file with size 0 is ambiguous: providers report 0 for sizes
            // they don't cheaply know (rendered Notion page.json, exported
            // Google docs). Don't trust it — compute via the provider so reads
            // aren't clamped to nothing. (A genuinely empty file just re-stats.)
            Some(_) => return self.inner.stat(path).await,
            None => {
                // Negative cache: a fresh parent listing that lacks this path
                // proves it does not exist — skip the network probe.
                if !path.is_root() && self.cache.is_listed(parent_of(key)) {
                    anyhow::bail!("not found: {key}");
                }
            }
        }
        self.inner.stat(path).await
    }

    async fn unlink(&self, path: &VPath) -> anyhow::Result<()> {
        let r = self.inner.unlink(path).await;
        if r.is_ok() {
            // The path itself may have been a directory; drop its listing too.
            self.cache.invalidate_dir(path.as_str());
            self.cache.invalidate_parent(path.as_str());
        }
        r
    }

    async fn mkdir(&self, path: &VPath) -> anyhow::Result<()> {
        let r = self.inner.mkdir(path).await;
        if r.is_ok() {
            self.cache.invalidate_parent(path.as_str());
        }
        r
    }

    async fn rmdir(&self, path: &VPath) -> anyhow::Result<()> {
        let r = self.inner.rmdir(path).await;
        if r.is_ok() {
            self.cache.invalidate_dir(path.as_str());
            self.cache.invalidate_parent(path.as_str());
        }
        r
    }

    async fn rename(&self, from: &VPath, to: &VPath) -> anyhow::Result<()> {
        let r = self.inner.rename(from, to).await;
        if r.is_ok() {
            self.cache.invalidate_dir(from.as_str());
            self.cache.invalidate_parent(from.as_str());
            self.cache.invalidate_parent(to.as_str());
        }
        r
    }

    async fn command(&self, name: &str, body: &[u8]) -> anyhow::Result<Vec<u8>> {
        let r = self.inner.command(name, body).await;
        // A domain write (e.g. Notion page-create) may change listings anywhere
        // in the mount; conservatively drop the whole index.
        if r.is_ok() {
            self.cache.clear();
        }
        r
    }

    fn prompt(&self) -> &str {
        self.inner.prompt()
    }
}

/// Parent directory of an absolute mount-relative path: `/a/b` -> `/a`,
/// `/a` -> `/`, `/` -> `/`.
fn parent_of(path: &str) -> &str {
    match path.rsplit_once('/') {
        Some(("", _)) | None => "/",
        Some((parent, _)) => parent,
    }
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn de(name: &str, kind: FileKind, size: u64) -> DirEntry {
        DirEntry {
            name: name.to_string(),
            kind,
            size,
        }
    }

    #[test]
    fn set_dir_then_get_and_list() {
        let c = IndexCache::new(Duration::from_secs(600));
        c.set_dir(
            "/",
            &[
                de("a.txt", FileKind::File, 10),
                de("sub", FileKind::Dir, 0),
            ],
        );
        // stat fast-path
        let a = c.get("/a.txt").unwrap();
        assert!(!a.is_dir && a.size == 10);
        assert!(c.get("/sub").unwrap().is_dir);
        // listing reconstruction
        let listed = c.list_dir_entries("/").unwrap();
        assert_eq!(listed.len(), 2);
        // negative cache basis
        assert!(c.is_listed("/"));
        assert!(c.get("/missing.txt").is_none());
    }

    #[test]
    fn nested_paths_and_parent() {
        assert_eq!(parent_of("/a/b"), "/a");
        assert_eq!(parent_of("/a"), "/");
        assert_eq!(parent_of("/"), "/");
        let c = IndexCache::new(Duration::from_secs(600));
        c.set_dir("/sub", &[de("c.txt", FileKind::File, 5)]);
        assert_eq!(c.get("/sub/c.txt").unwrap().size, 5);
        assert_eq!(basename("/sub/c.txt"), "c.txt");
    }

    #[test]
    fn invalidation() {
        let c = IndexCache::new(Duration::from_secs(600));
        c.set_dir("/", &[de("a.txt", FileKind::File, 1)]);
        c.invalidate_parent("/a.txt"); // parent of /a.txt is /
        assert!(!c.is_listed("/"));
        assert!(c.get("/a.txt").is_none());
    }

    #[test]
    fn expiry() {
        let c = IndexCache::new(Duration::from_millis(0));
        c.set_dir("/", &[de("a.txt", FileKind::File, 1)]);
        // TTL 0 -> already expired
        assert!(!c.is_listed("/"));
        assert!(c.list_dir_entries("/").is_none());
    }
}
