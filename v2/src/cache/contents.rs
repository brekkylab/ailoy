use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use crate::cache::CacheEntry;

#[derive(Clone, Debug)]
pub struct CacheContents {
    root: PathBuf,
    inner: BTreeMap<CacheEntry, Vec<u8>>,
}

impl CacheContents {
    pub fn new() -> Self {
        CacheContents {
            root: PathBuf::new(),
            inner: BTreeMap::new(),
        }
    }

    pub fn with_root(self, root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            inner: self.inner,
        }
    }

    pub fn get_root(&self) -> &Path {
        &self.root
    }

    pub fn remove(&mut self, entry: &CacheEntry) -> Option<Vec<u8>> {
        self.inner.remove(entry)
    }

    pub fn remove_with_filename(
        &mut self,
        filename: impl AsRef<str>,
    ) -> Option<(CacheEntry, Vec<u8>)> {
        let entry = {
            let Some((entry, _)) = self
                .inner
                .iter()
                .find(|(k, _)| k.filename() == filename.as_ref())
            else {
                return None;
            };
            entry.clone()
        };
        self.inner.remove_entry(&entry)
    }
}

impl FromIterator<(CacheEntry, Vec<u8>)> for CacheContents {
    fn from_iter<T: IntoIterator<Item = (CacheEntry, Vec<u8>)>>(iter: T) -> Self {
        CacheContents {
            root: PathBuf::new(),
            inner: iter.into_iter().collect(),
        }
    }
}
