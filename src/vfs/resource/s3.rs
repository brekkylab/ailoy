use std::ops::Range;

use async_trait::async_trait;
use object_store::path::Path as OsPath;
use object_store::{Error as OsError, GetOptions, GetRange, ObjectStore, PutPayload};

use crate::vfs::accessor::{S3Accessor, S3Config};
use crate::vfs::path::VPath;
use crate::vfs::resource::{DirEntry, FileKind, FileStat, Resource};

const S3_PROMPT: &str = "\
Amazon S3 (read/write). Object keys map to paths; directories are key prefixes.
Standard shell tools work: ls, cat, head, grep, find, tee, rm, cp/mv.
Remote mount — prefer head/grep over cat on large objects.";

pub struct S3Resource {
    accessor: S3Accessor,
}

impl S3Resource {
    pub fn new(config: &S3Config) -> anyhow::Result<Self> {
        Ok(Self {
            accessor: S3Accessor::new(config)?,
        })
    }

    fn os_path(&self, path: &VPath) -> OsPath {
        OsPath::from(self.accessor.key(path))
    }

    fn list_prefix(&self, path: &VPath) -> Option<OsPath> {
        let key = self.accessor.key(path);
        if key.is_empty() {
            None
        } else {
            Some(OsPath::from(key))
        }
    }
}

#[async_trait]
impl Resource for S3Resource {
    async fn read_bytes(&self, path: &VPath, range: Option<Range<u64>>) -> anyhow::Result<Vec<u8>> {
        let opts = GetOptions {
            range: range.map(GetRange::Bounded),
            ..Default::default()
        };
        let res = self.accessor.store.get_opts(&self.os_path(path), opts).await?;
        Ok(res.bytes().await?.to_vec())
    }

    async fn write_bytes(&self, path: &VPath, data: Vec<u8>) -> anyhow::Result<()> {
        self.accessor
            .store
            .put(&self.os_path(path), PutPayload::from(data))
            .await?;
        Ok(())
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        let res = self
            .accessor
            .store
            .list_with_delimiter(self.list_prefix(path).as_ref())
            .await?;
        let mut out = Vec::new();
        for cp in res.common_prefixes {
            if let Some(name) = cp.filename() {
                out.push(DirEntry {
                    name: name.to_string(),
                    kind: FileKind::Dir,
                    size: 0,
                });
            }
        }
        for obj in res.objects {
            if let Some(name) = obj.location.filename() {
                out.push(DirEntry {
                    name: name.to_string(),
                    kind: FileKind::File,
                    size: obj.size,
                });
            }
        }
        Ok(out)
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        if path.is_root() {
            return Ok(FileStat {
                kind: FileKind::Dir,
                size: 0,
            });
        }
        match self.accessor.store.head(&self.os_path(path)).await {
            Ok(meta) => Ok(FileStat {
                kind: FileKind::File,
                size: meta.size,
            }),
            Err(OsError::NotFound { .. }) => {
                let res = self
                    .accessor
                    .store
                    .list_with_delimiter(self.list_prefix(path).as_ref())
                    .await?;
                if res.common_prefixes.is_empty() && res.objects.is_empty() {
                    anyhow::bail!("not found: {}", path.as_str());
                }
                Ok(FileStat {
                    kind: FileKind::Dir,
                    size: 0,
                })
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn unlink(&self, path: &VPath) -> anyhow::Result<()> {
        self.accessor.store.delete(&self.os_path(path)).await?;
        Ok(())
    }

    fn prompt(&self) -> &str {
        S3_PROMPT
    }
}
