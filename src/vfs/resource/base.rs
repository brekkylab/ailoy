use std::ops::Range;

use async_trait::async_trait;

use crate::vfs::path::VPath;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FileKind {
    #[default]
    File,
    Dir,
}

#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub kind: FileKind,
    pub size: u64,
    /// Last-modified time, if the backend reports one per entry (S3
    /// `LastModified`). Carried from `readdir` so the cache can serve it on the
    /// stat fast-path (`ls -l`) instead of falling back to the UNIX epoch (R2).
    pub mtime: Option<std::time::SystemTime>,
}

#[derive(Clone, Debug, Default)]
pub struct FileStat {
    pub kind: FileKind,
    pub size: u64,
    /// Last-modified time, if the backend reports one (S3 `LastModified`).
    pub mtime: Option<std::time::SystemTime>,
    /// Entity tag / content fingerprint, if available (S3 `ETag`).
    pub etag: Option<String>,
    /// Version id, if the backend is versioned (S3 `VersionId`).
    pub version: Option<String>,
}

/// A mounted provider. One instance owns one set of credentials, so the same
/// provider type mounted with different credentials is distinct instances.
///
/// The FUSE frontends translate filesystem callbacks into these operations.
/// Errors propagate as `anyhow`; callers map them to errno.
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

    /// Create a directory at `path`. Default: unsupported.
    async fn mkdir(&self, path: &VPath) -> anyhow::Result<()> {
        let _ = path;
        anyhow::bail!("mkdir not supported by this resource")
    }

    /// Remove the directory (and its contents) at `path`. Default: unsupported.
    async fn rmdir(&self, path: &VPath) -> anyhow::Result<()> {
        let _ = path;
        anyhow::bail!("rmdir not supported by this resource")
    }

    /// Rename/move `from` to `to`. Default: unsupported.
    async fn rename(&self, from: &VPath, to: &VPath) -> anyhow::Result<()> {
        let _ = (from, to);
        anyhow::bail!("rename not supported by this resource")
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
