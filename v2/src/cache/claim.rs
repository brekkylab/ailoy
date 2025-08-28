use std::any::Any;

use crate::{cache::CacheEntry, dyn_maybe_send};

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

    /// An optional context object. This is a boxed [`Any`] value that
    /// implementations can use to carry extra information from `claim_files` to
    /// `try_from_contents`. Most types will not need it and can leave it as `None`.
    pub ctx: Option<Box<dyn_maybe_send!(Any)>>,
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
            ctx: None,
        }
    }

    /// Create a new claim with a context
    ///
    /// Use this if you also need to pass additional metadata
    /// or parameters along to the build step:
    ///
    /// ```rust,ignore
    /// let claim = CacheClaim::with_ctx(
    ///     [CacheEntry::new("dataset", "index.json")],
    ///     Box::new(("shard_count", 4)), // custom info in ctx
    /// );
    /// ```
    pub fn with_ctx(
        entries: impl IntoIterator<Item = impl Into<CacheEntry>>,
        ctx: Box<dyn_maybe_send!(Any)>,
    ) -> Self {
        Self {
            entries: entries.into_iter().map(|v| v.into()).collect(),
            ctx: Some(ctx),
        }
    }
}
