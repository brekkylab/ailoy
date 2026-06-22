use std::ops::Range;

use async_trait::async_trait;

use crate::vfs::path::VPath;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileKind {
    File,
    Dir,
}

#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub kind: FileKind,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct FileStat {
    pub kind: FileKind,
    pub size: u64,
}

/// A mounted provider. One instance owns one set of credentials, so the same
/// provider type mounted with different credentials is distinct instances.
///
/// Mirrors mirage's `BaseResource`; the FUSE frontends translate filesystem
/// callbacks into these operations. Errors propagate as `anyhow`; callers map
/// them to errno (see mirage `fuse/fs.py`).
#[async_trait]
pub trait Resource: Send + Sync {
    async fn read_bytes(&self, path: &VPath, range: Option<Range<u64>>) -> anyhow::Result<Vec<u8>>;

    async fn write_bytes(&self, path: &VPath, data: Vec<u8>) -> anyhow::Result<()>;

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>>;

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat>;

    async fn unlink(&self, path: &VPath) -> anyhow::Result<()> {
        let _ = path;
        anyhow::bail!("unlink not supported by this resource")
    }

    /// Domain operation routed from a `/<mount>/.cmd/<name>` write
    /// (e.g. Notion `page-create`, GDocs `docs-append`).
    async fn command(&self, name: &str, body: &[u8]) -> anyhow::Result<Vec<u8>> {
        let _ = (name, body);
        anyhow::bail!("command not supported by this resource")
    }

    /// System-prompt section describing this mount's layout and commands.
    fn prompt(&self) -> &str {
        ""
    }
}
