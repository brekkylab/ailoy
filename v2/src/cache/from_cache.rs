use std::pin::Pin;

use crate::cache::{Cache, CacheContents, CacheEntry};

pub trait TryFromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheEntry>, String>>>>;

    /// Create from cache
    fn try_from_contents(contents: &mut CacheContents) -> Result<Self, String>
    where
        Self: Sized;
}

pub trait FromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Vec<CacheEntry>>>>;

    /// Create from cache
    fn try_from_contents(contents: &mut CacheContents) -> Self
    where
        Self: Sized;
}
