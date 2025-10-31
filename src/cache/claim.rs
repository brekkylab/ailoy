use crate::cache::CacheEntry;

/// A declaration of which files are needed to build an object from the cache.
///
/// `CacheClaim` is returned by [`TryFromCache::claim_files`] (or [`FromCache::claim_files`]).
/// It tells the cache system:
/// - **which files to fetch** (`entries`)
/// - and optionally, **extra context** (`ctx`) that helps during construction.
///
/// # Notes
/// - The `ctx` value is type-erased (`Box<dyn Any>`), so both sides must agree
///   on what type of data is stored there.
/// - `entries` should only describe *logical filenames* that exist in the
///   remote/local manifest.
pub struct CacheClaim {
    /// A list of [`CacheEntry`] values describing the logical files (by `dirname` / `filename`) required for this object.
    pub entries: Vec<CacheEntry>,
}

impl CacheClaim {
    /// Create a new claim
    ///
    /// ```rust,ignore
    /// let claim = CacheClaim::new([
    ///     CacheEntry::new("my_model", "tokenizer.json"),
    ///     CacheEntry::new("my_model", "weights.safetensors"),
    /// ]);
    /// ```
    pub fn new(entries: impl IntoIterator<Item = impl Into<CacheEntry>>) -> Self {
        Self {
            entries: entries.into_iter().map(|v| v.into()).collect(),
        }
    }
}
