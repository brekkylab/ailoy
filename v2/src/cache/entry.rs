use std::{path::PathBuf, str::FromStr};

#[derive(Debug, Clone)]
pub struct CacheEntry {
    dirname: String,
    filename: String,
}

impl CacheEntry {
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

impl AsRef<CacheEntry> for CacheEntry {
    fn as_ref(&self) -> &CacheEntry {
        self
    }
}
