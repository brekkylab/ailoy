use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::cache::{CacheEntry, filesystem};

/// Represents a lazy source of bytes that can be read on demand.
///
/// This enum allows for memory-efficient caching by keeping large files as references
/// to their on-disk locations rather than loading them entirely into memory.
#[derive(Debug, Clone)]
pub enum ByteSource {
    /// Bytes already loaded in memory (for small files like JSON)
    Eager(Vec<u8>),
    /// Path to file on disk (native) or OPFS (wasm) - loaded on demand
    Lazy(PathBuf),
}

#[allow(dead_code)]
impl ByteSource {
    /// Read all bytes from this source.
    ///
    /// For `Eager`, returns a clone of the in-memory bytes.
    /// For `Lazy`, reads the file from disk asynchronously.
    pub async fn read_all(&self) -> anyhow::Result<Vec<u8>> {
        match self {
            ByteSource::Eager(bytes) => Ok(bytes.clone()),
            ByteSource::Lazy(path) => filesystem::read(path).await,
        }
    }

    /// Get bytes if already in memory, otherwise None.
    pub fn as_eager(&self) -> Option<&[u8]> {
        match self {
            ByteSource::Eager(bytes) => Some(bytes),
            ByteSource::Lazy(_) => None,
        }
    }

    /// Get path if lazy, otherwise None.
    pub fn as_path(&self) -> Option<&Path> {
        match self {
            ByteSource::Eager(_) => None,
            ByteSource::Lazy(path) => Some(path),
        }
    }

    /// Returns true if this is an eager (in-memory) source.
    pub fn is_eager(&self) -> bool {
        matches!(self, ByteSource::Eager(_))
    }

    /// Returns true if this is a lazy (on-disk) source.
    pub fn is_lazy(&self) -> bool {
        matches!(self, ByteSource::Lazy(_))
    }
}

/// A lightweight container of files fetched for a cache-backed build step.
///
/// `CacheContents` holds byte sources keyed by [`CacheEntry`] (a `(dirname, filename)`
/// pair). It is typically produced by a loader/downloader and then consumed by
/// constructors that assemble higher-level types. The optional `root` remembers the
/// local cache directory used during fetching.
///
/// Files can be stored either eagerly (in memory) or lazily (as file paths), allowing
/// memory-efficient handling of large model files while keeping small configuration
/// files readily accessible.
///
/// This type makes minimal assumptions about how it's filled or consumed. It merely
/// stores entries and offers basic operations to inspect or remove them. The exact
/// construction pipeline and access patterns are up to the caller.
#[allow(dead_code)]
#[derive(Debug)]
pub struct CacheContents {
    pub root: PathBuf,
    pub entries: BTreeMap<CacheEntry, ByteSource>,
}

#[allow(dead_code)]
impl CacheContents {
    /// Drain all entries, converting ByteSources to Vec<u8>.
    ///
    /// This is an async operation because lazy sources need to be read from disk.
    /// Note: This loads all lazy sources into memory, which may cause OOM for large models.
    /// Consider using individual `remove` operations instead for memory efficiency.
    pub async fn drain_eager(&mut self) -> anyhow::Result<Vec<(CacheEntry, Vec<u8>)>> {
        let keys = self
            .entries
            .keys()
            .into_iter()
            .map(|v| v.clone())
            .collect::<Vec<_>>();
        let mut rv = Vec::new();
        for k in keys {
            let source = self.entries.remove(&k).unwrap();
            let bytes = source.read_all().await?;
            rv.push((k, bytes));
        }
        Ok(rv)
    }

    /// Drain all entries as ByteSources without loading lazy sources.
    pub fn drain(&mut self) -> impl IntoIterator<Item = (CacheEntry, ByteSource)> {
        let keys = self
            .entries
            .keys()
            .into_iter()
            .map(|v| v.clone())
            .collect::<Vec<_>>();
        let mut rv = Vec::new();
        for k in keys {
            let v = self.entries.remove(&k).unwrap();
            rv.push((k, v));
        }
        rv.into_iter()
    }

    /// Remove and return the ByteSource associated with the exact key.
    pub fn remove(&mut self, entry: &CacheEntry) -> Option<ByteSource> {
        self.entries.remove(entry)
    }

    /// Remove and return one entry whose `filename` matches the given value.
    ///
    /// If multiple entries share the same `filename` across different directories,
    /// which one is returned is unspecified.
    pub fn remove_with_filename(
        &mut self,
        filename: impl AsRef<str>,
    ) -> Option<(CacheEntry, ByteSource)> {
        let entry = {
            let Some((entry, _)) = self
                .entries
                .iter()
                .find(|(k, _)| k.filename() == filename.as_ref())
            else {
                return None;
            };
            entry.clone()
        };
        self.entries.remove_entry(&entry)
    }

    /// Read all bytes from a specific entry (convenience method).
    ///
    /// Returns an error if the entry doesn't exist or if reading fails.
    pub async fn read_entry(&self, entry: &CacheEntry) -> anyhow::Result<Vec<u8>> {
        self.entries
            .get(entry)
            .ok_or_else(|| anyhow::anyhow!("Entry not found: {}", entry.filename()))?
            .read_all()
            .await
    }
}
