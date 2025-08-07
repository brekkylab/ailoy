mod cache;
mod manifest;

use std::{path::PathBuf, pin::Pin, str::FromStr};

pub use cache::*;
pub use manifest::*;

#[derive(Debug, Clone)]
pub struct CacheElement {
    dirname: String,
    filename: String,
}

impl CacheElement {
    pub fn new(dirname: impl AsRef<str>, filename: impl AsRef<str>) -> Self {
        Self {
            dirname: dirname.as_ref().to_owned(),
            filename: filename.as_ref().to_owned(),
        }
    }

    pub fn dirname(&self) -> &str {
        &self.dirname
    }

    pub fn filename(&self) -> &str {
        &self.filename
    }

    pub fn path(&self) -> String {
        PathBuf::from_str(&self.dirname)
            .unwrap()
            .join(&self.filename)
            .to_string_lossy()
            .to_string()
    }
}

impl AsRef<CacheElement> for CacheElement {
    fn as_ref(&self) -> &CacheElement {
        self
    }
}

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
