use crate::cache::CacheEntry;

/// A declaration of which files are needed to build an object from the cache.
///
/// `CacheClaim` is returned by [`TryFromCache::claim_files`] (or [`FromCache::claim_files`]).
/// It tells the cache system **which files to fetch** (`entries`)
#[derive(Clone)]
pub struct CacheClaim {
    /// A list of [`CacheEntry`] values describing the logical files (by `dirname` / `filename`) required for this object.
    pub entries: Vec<CacheEntry>,
}

impl CacheClaim {
    /// Create a new claim
    pub fn new(entries: impl IntoIterator<Item = impl Into<CacheEntry>>) -> Self {
        Self {
            entries: entries.into_iter().map(|v| v.into()).collect(),
        }
    }
}
