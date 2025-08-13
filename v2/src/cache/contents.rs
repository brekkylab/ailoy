use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::cache::CacheEntry;

/// A lightweight container of files fetched for a cache-backed build step.
///
/// `CacheContents` holds raw bytes keyed by [`CacheEntry`] (a `(dirname, filename)`
/// pair). It is typically produced by a loader/downloader and then consumed by
/// constructors that assemble higher-level types. The optional `root` remembers the
/// local cache directory used during fetching.
///
/// This type makes minimal assumptions about how it’s filled or consumed. It merely
/// stores entries and offers basic operations to inspect or remove them. The exact
/// construction pipeline and access patterns are up to the caller.
#[derive(Clone, Debug)]
pub struct CacheContents {
    root: PathBuf,
    inner: BTreeMap<CacheEntry, Vec<u8>>,
}

impl CacheContents {
    /// Create an empty collection with no associated root path.
    pub fn new() -> Self {
        CacheContents {
            root: PathBuf::new(),
            inner: BTreeMap::new(),
        }
    }

    /// Return a new collection that remembers the given local cache root.
    ///
    /// This does not touch the filesystem; it only records the path for later use.
    pub fn with_root(self, root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            inner: self.inner,
        }
    }

    /// The recorded local cache root, if any.
    pub fn get_root(&self) -> &Path {
        &self.root
    }

    /// Remove and return the bytes associated with the exact key.
    pub fn remove(&mut self, entry: &CacheEntry) -> Option<Vec<u8>> {
        self.inner.remove(entry)
    }

    /// Remove and return one entry whose `filename` matches the given value.
    ///
    /// If multiple entries share the same `filename` across different directories,
    /// which one is returned is unspecified.
    pub fn remove_with_filename(
        &mut self,
        filename: impl AsRef<str>,
    ) -> Option<(CacheEntry, Vec<u8>)> {
        let entry = {
            let Some((entry, _)) = self
                .inner
                .iter()
                .find(|(k, _)| k.filename() == filename.as_ref())
            else {
                return None;
            };
            entry.clone()
        };
        self.inner.remove_entry(&entry)
    }
}

impl FromIterator<(CacheEntry, Vec<u8>)> for CacheContents {
    /// Build a collection from `(key, bytes)` pairs.
    ///
    /// Later duplicates overwrite earlier ones. The resulting `root` is empty; set
    /// it explicitly with [`with_root`] if you need it.
    fn from_iter<T: IntoIterator<Item = (CacheEntry, Vec<u8>)>>(iter: T) -> Self {
        CacheContents {
            root: PathBuf::new(),
            inner: iter.into_iter().collect(),
        }
    }
}
