use std::{path::PathBuf, str::FromStr};

#[derive(Debug, Clone)]
pub struct CacheElement {
    pub(crate) dirname: String,
    pub(crate) filename: String,
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
