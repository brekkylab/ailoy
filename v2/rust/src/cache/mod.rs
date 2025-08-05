mod cache;
mod manifest;

use std::{path::PathBuf, pin::Pin};

pub use cache::*;
pub use manifest::*;

pub trait TryFromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<PathBuf>, String>>>>;

    /// Create from cache
    fn try_from_files(files: Vec<(PathBuf, Vec<u8>)>) -> Result<Self, String>
    where
        Self: Sized;
}

pub trait FromCache {
    /// List of files to be downloaded
    fn claim_files(
        cache: Cache,
        key: impl AsRef<str>,
    ) -> Pin<Box<dyn Future<Output = Vec<PathBuf>>>>;

    /// Create from cache
    fn from_files(files: Vec<(PathBuf, Vec<u8>)>) -> Self
    where
        Self: Sized;
}
