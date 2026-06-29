use std::ops::Range;

use async_trait::async_trait;
use futures::StreamExt;
use object_store::{
    Error as OsError, GetOptions, GetRange, ObjectStore, PutPayload, path::Path as OsPath,
};

use crate::vfs::{
    accessor::{S3Accessor, S3Config},
    path::VPath,
    resource::{DirEntry, FileKind, FileStat, Resource},
};

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
            range: range.clone().map(GetRange::Bounded),
            ..Default::default()
        };
        match self
            .accessor
            .store
            .get_opts(&self.os_path(path), opts)
            .await
        {
            Ok(res) => Ok(res.bytes().await?.to_vec()),
            Err(e) => {
                // S3-2: a bounded range starting at/after EOF returns 416. Treat
                // it as a clean EOF (empty) rather than EIO, so cat/wc/dd over a
                // direct_io mount (size unknown up front) stop cleanly. Only the
                // error path pays the extra head.
                if let Some(r) = &range
                    && let Ok(meta) = self.accessor.store.head(&self.os_path(path)).await
                    && r.start >= meta.size
                {
                    return Ok(Vec::new());
                }
                Err(e.into())
            }
        }
    }

    async fn write_bytes(&self, path: &VPath, data: Vec<u8>) -> anyhow::Result<()> {
        self.accessor
            .store
            .put(&self.os_path(path), PutPayload::from(data))
            .await?;
        Ok(())
    }

    async fn readdir(&self, path: &VPath) -> anyhow::Result<Vec<DirEntry>> {
        let listing = self.list_prefix(path);
        let res = self
            .accessor
            .store
            .list_with_delimiter(listing.as_ref())
            .await?;
        // The key we listed under; an object whose key equals it is the
        // zero-byte "directory marker" for this prefix and must be skipped.
        let marker = listing.as_ref().map(|p| p.as_ref()).unwrap_or("");
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
            if obj.location.as_ref() == marker {
                continue;
            }
            if let Some(name) = obj.location.filename() {
                out.push(DirEntry {
                    name: name.to_string(),
                    kind: FileKind::File,
                    size: obj.size,
                });
            }
        }
        // Return one merged, name-sorted listing; coreutils `ls` re-sorts
        // anyway, but this keeps the raw readdir order stable.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    async fn stat(&self, path: &VPath) -> anyhow::Result<FileStat> {
        if path.is_root() {
            return Ok(FileStat {
                kind: FileKind::Dir,
                ..Default::default()
            });
        }
        match self.accessor.store.head(&self.os_path(path)).await {
            Ok(meta) => Ok(FileStat {
                kind: FileKind::File,
                size: meta.size,
                mtime: Some(meta.last_modified.into()),
                etag: meta.e_tag.clone(),
                version: meta.version.clone(),
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
                    ..Default::default()
                })
            }
            Err(e) => Err(e.into()),
        }
    }

    async fn unlink(&self, path: &VPath) -> anyhow::Result<()> {
        self.accessor.store.delete(&self.os_path(path)).await?;
        Ok(())
    }

    async fn mkdir(&self, _path: &VPath) -> anyhow::Result<()> {
        // Object stores have no real directories: a prefix exists implicitly once
        // a key is written under it, and `object_store::Path` can't represent a
        // trailing-slash marker. So mkdir is a no-op success (the dir appears as
        // soon as something is written into it).
        Ok(())
    }

    async fn rmdir(&self, path: &VPath) -> anyhow::Result<()> {
        // Recursively delete everything under the prefix (mirrors mirage's
        // prefix batch delete).
        let prefix = self.list_prefix(path);
        let mut stream = self.accessor.store.list(prefix.as_ref());
        while let Some(item) = stream.next().await {
            let meta = item?;
            self.accessor.store.delete(&meta.location).await?;
        }
        Ok(())
    }

    async fn rename(&self, from: &VPath, to: &VPath) -> anyhow::Result<()> {
        // S3 has no native rename: copy then delete the source (mirrors mirage).
        let (from, to) = (self.os_path(from), self.os_path(to));
        self.accessor.store.copy(&from, &to).await?;
        self.accessor.store.delete(&from).await?;
        Ok(())
    }

    fn prompt(&self) -> &str {
        S3_PROMPT
    }
}
