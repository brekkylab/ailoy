use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{datatype::Bytes, runenv::Console};

/// A file to be pre-filled into a [`Machine`](crate::runenv::Machine)'s
/// console, or read back from one.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct FileEntry {
    pub path: PathBuf,
    pub content: Bytes,
}

impl FileEntry {
    pub fn new(path: impl Into<PathBuf>, content: impl Into<Bytes>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }

    /// Write this entry to `runenv` at its declared path.
    pub async fn write_to(&self, console: &dyn Console) -> anyhow::Result<()> {
        console.write(&self.path, self.content.as_ref()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let f = FileEntry::new("/workspace/foo.txt", b"hi".to_vec());
        assert_eq!(f.path, PathBuf::from("/workspace/foo.txt"));
        assert_eq!(f.content.as_ref(), b"hi");
    }

    #[test]
    fn test_json_roundtrip() {
        let f = FileEntry::new("/p", b"q".to_vec());
        let json = serde_json::to_string(&f).unwrap();
        let back: FileEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back.path, f.path);
        assert_eq!(back.content.as_ref(), f.content.as_ref());
        // No permission field in the wire form.
        assert!(
            !json.contains("permission"),
            "json should not carry permission: {json}"
        );
    }
}
