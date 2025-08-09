use std::pin::Pin;

use crate::cache::{Cache, CacheElement};

pub trait TryFromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CacheElement>, String>>>>;

    /// Create from cache
    fn try_from_files(cache: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Result<Self, String>
    where
        Self: Sized;
}

pub trait FromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Vec<CacheElement>>>>;

    /// Create from cache
    fn from_files(cache: &Cache, files: Vec<(CacheElement, Vec<u8>)>) -> Self
    where
        Self: Sized;
}
