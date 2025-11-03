use std::{collections::BTreeMap, path::PathBuf};

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
#[derive(Debug)]
pub struct CacheContents {
    pub root: PathBuf,
    pub entries: BTreeMap<CacheEntry, Vec<u8>>,
}

impl CacheContents {
    pub fn drain(&mut self) -> impl IntoIterator<Item = (CacheEntry, Vec<u8>)> {
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

    /// Remove and return the bytes associated with the exact key.
    pub fn remove(&mut self, entry: &CacheEntry) -> Option<Vec<u8>> {
        self.entries.remove(entry)
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
}
