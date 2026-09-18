//! The file-tree half of a session: listing, reading, editing, and getting host files in.
//!
//! Every function here is a thin translation of one [`FileSystem`] call, with three things added
//! that a tree drawn in a webview needs and the trait deliberately does not provide.
//!
//! * **Timestamps as numbers.** `SystemTime` has no JSON spelling; milliseconds since the epoch
//!   is the one JavaScript reads without a library.
//! * **A size that may be absent.** [`Dirent::stat`] is `Some` only when the listing produced it
//!   for free, and this passes that through rather than filling it in. A directory of a hundred
//!   Notion pages would otherwise cost a hundred round trips to draw a column most readers do
//!   not look at — the trait's docs make that an N+1 on purpose, and so does this.
//! * **A ceiling on a read.** The editor holds the whole file as a string, so a read is capped
//!   and says when it hit the cap. Without that, opening one wrong object out of a bucket is a
//!   session that stops answering.
//!
//! [`Dirent::stat`]: cortex::fs::Dirent::stat

use std::path::{Component, Path, PathBuf};

use cortex::fs::{DirentKind, FileSystem};

use crate::{
    error::{EngineError, Result},
    types::{Entry, FileContent, ImportReport},
};

/// How much of a file the editor will hold. Past this the read stops and says so.
const READ_CAP: u64 = 1 << 20;

/// The largest host file an import will copy in.
///
/// The root store is memory, so an import is charged to RAM at full size; a dropped disk image
/// would take the engine down with the allocator rather than with an error anybody could read.
const IMPORT_CAP: u64 = 64 << 20;

/// A `SystemTime` as milliseconds since the epoch, or `None` for one that predates it.
fn epoch_ms(t: std::time::SystemTime) -> Option<u64> {
    t.duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// The children of `path`.
pub async fn list(fs: &dyn FileSystem, path: &str) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    for dirent in fs.list(Path::new(path)).await? {
        let (size, mtime_ms) = match dirent.stat() {
            Some(stat) => (Some(stat.size), stat.mtime.and_then(epoch_ms)),
            None => (None, None),
        };
        entries.push(Entry {
            path: join(path, &dirent.name),
            name: dirent.name,
            kind: kind_str(dirent.kind).to_string(),
            size,
            mtime_ms,
        });
    }
    // Directories first, then by name: the order a store answers in is its own business —
    // `ContextFs` leads with mount points, an object store with whatever its listing returned —
    // and a tree that reorders itself as stores are added is harder to read than a sorted one.
    entries.sort_by_key(|entry| (entry.kind != "dir", entry.name.to_lowercase()));
    Ok(entries)
}

/// The contents of `path`, up to [`READ_CAP`].
pub async fn read(fs: &dyn FileSystem, path: &str) -> Result<FileContent> {
    let target = Path::new(path);
    let stat = fs.stat(target).await?;
    if stat.kind == DirentKind::Dir {
        return Err(EngineError::Invalid(
            "디렉터리는 편집기에서 열 수 없습니다".into(),
        ));
    }

    let buf = read_all(fs, target, READ_CAP).await?;
    let truncated = stat.size > READ_CAP;
    Ok(FileContent {
        path: path.to_string(),
        text: as_text(buf, truncated),
        size: stat.size,
        truncated,
    })
}

/// Replace the contents of `path`, creating the file if it is not there.
pub async fn write(fs: &dyn FileSystem, path: &str, text: &str) -> Result<()> {
    write_file(fs, Path::new(path), text.as_bytes()).await
}

/// An empty file at `path`. Fails if the name is taken.
pub async fn touch(fs: &dyn FileSystem, path: &str) -> Result<()> {
    fs.create(Path::new(path)).await?;
    Ok(())
}

/// A directory at `path`, and every parent it needs.
pub async fn mkdir(fs: &dyn FileSystem, path: &str) -> Result<()> {
    mkdir_p(fs, Path::new(path)).await
}

/// Remove `path`. A directory has to be empty, exactly as `rmdir(2)` requires — nothing here
/// deletes a subtree, because a click that quietly removed one is not a click anybody can undo.
pub async fn delete(fs: &dyn FileSystem, path: &str) -> Result<()> {
    let target = Path::new(path);
    match fs.stat(target).await?.kind {
        DirentKind::Dir => fs.rmdir(target).await?,
        DirentKind::File => fs.unlink(target).await?,
    }
    Ok(())
}

/// Move `from` onto `to`, within one store.
pub async fn rename(fs: &dyn FileSystem, from: &str, to: &str) -> Result<()> {
    fs.rename(Path::new(from), Path::new(to)).await?;
    Ok(())
}

/// Copy host paths into `dest`, which must be a directory in the workspace.
///
/// This is what a drop on the window and the file chooser both end in. Directories are copied
/// whole, with their names preserved, so dragging a folder in gives the folder rather than its
/// loose contents.
pub async fn import(
    fs: &dyn FileSystem,
    dest: &str,
    sources: Vec<PathBuf>,
) -> Result<ImportReport> {
    if fs.stat(Path::new(dest)).await?.kind != DirentKind::Dir {
        return Err(EngineError::Invalid(
            "파일은 디렉터리 위에만 놓을 수 있습니다".into(),
        ));
    }

    let mut report = ImportReport {
        files: 0,
        bytes: 0,
        skipped: Vec::new(),
    };
    // An explicit stack rather than recursion: an `async fn` that calls itself needs boxing at
    // every level, and the tree being walked is the host's, whose depth nobody here chose.
    let mut pending: Vec<(PathBuf, String)> = Vec::new();
    for source in sources {
        let name = match source.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => {
                report
                    .skipped
                    .push(format!("{}: 이름을 읽을 수 없음", source.display()));
                continue;
            }
        };
        pending.push((source, join(dest, &name)));
    }

    while let Some((source, target)) = pending.pop() {
        let meta = match tokio::fs::metadata(&source).await {
            Ok(meta) => meta,
            Err(err) => {
                report.skipped.push(format!("{}: {err}", source.display()));
                continue;
            }
        };

        if meta.is_dir() {
            if let Err(err) = mkdir_p(fs, Path::new(&target)).await {
                report.skipped.push(format!("{}: {err}", source.display()));
                continue;
            }
            match tokio::fs::read_dir(&source).await {
                Ok(mut dir) => {
                    while let Ok(Some(child)) = dir.next_entry().await {
                        let name = child.file_name().to_string_lossy().into_owned();
                        pending.push((child.path(), join(&target, &name)));
                    }
                }
                Err(err) => report.skipped.push(format!("{}: {err}", source.display())),
            }
            continue;
        }

        if meta.len() > IMPORT_CAP {
            report.skipped.push(format!(
                "{}: {} MiB — 한 파일당 {} MiB까지만 가져옵니다",
                source.display(),
                meta.len() >> 20,
                IMPORT_CAP >> 20
            ));
            continue;
        }

        let bytes = match tokio::fs::read(&source).await {
            Ok(bytes) => bytes,
            Err(err) => {
                report.skipped.push(format!("{}: {err}", source.display()));
                continue;
            }
        };
        match write_file(fs, Path::new(&target), &bytes).await {
            Ok(()) => {
                report.files += 1;
                report.bytes += bytes.len() as u64;
            }
            Err(err) => report.skipped.push(format!("{}: {err}", source.display())),
        }
    }
    Ok(report)
}

/// Write `bytes` as the whole of `path`, creating it if needed.
///
/// Truncate first, then write: without it a shorter edit of a longer file leaves the old tail
/// behind, which is the classic way an editor corrupts what it saved.
pub(crate) async fn write_file(fs: &dyn FileSystem, path: &Path, bytes: &[u8]) -> Result<()> {
    match fs.stat(path).await {
        Ok(stat) if stat.kind == DirentKind::Dir => {
            return Err(EngineError::Invalid("디렉터리에는 쓸 수 없습니다".into()));
        }
        Ok(_) => fs.truncate(path, 0).await?,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            fs.create(path).await?;
        }
        Err(err) => return Err(err.into()),
    }

    let mut written = 0usize;
    while written < bytes.len() {
        // A short write is legal and draining the buffer is the caller's job — the trait puts
        // the loop here rather than defaulting it, so that a store cannot disagree with it.
        let n = fs.write_at(path, &bytes[written..], written as u64).await?;
        if n == 0 {
            return Err(EngineError::Invalid(
                "저장소가 쓰기를 더 받지 않습니다".into(),
            ));
        }
        written += n;
    }
    fs.flush(path).await?;
    Ok(())
}

/// Read up to `cap` bytes of `path`, in whatever number of calls the store answers in.
pub(crate) async fn read_all(fs: &dyn FileSystem, path: &Path, cap: u64) -> Result<Vec<u8>> {
    let size = fs.stat(path).await?.size.min(cap);
    let mut buf = vec![0u8; size as usize];
    let mut filled = 0usize;
    while filled < buf.len() {
        // A short read means EOF and nothing else — the trait says so — which is what makes
        // this loop terminate on a store whose size was stale.
        let n = fs.read_at(path, &mut buf[filled..], filled as u64).await?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(buf)
}

/// `mkdir -p`: each level in turn, and a level that is already a directory is not a failure.
pub(crate) async fn mkdir_p(fs: &dyn FileSystem, path: &Path) -> Result<()> {
    let mut here = PathBuf::from("/");
    // `Normal` components only. A leading `RootDir` would make the first level the root itself,
    // and a store asked to create its own root has no name to create — `InMemFs` answers that
    // with `InvalidFilename`, which is not one of the "already there" cases below.
    for component in path.components() {
        let Component::Normal(name) = component else {
            continue;
        };
        here.push(name);
        match fs.mkdir(&here).await {
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                // Already there, but a file with the same name is not a directory to descend
                // into, and continuing would report success for a path that cannot hold one.
                if fs.stat(&here).await?.kind != DirentKind::Dir {
                    return Err(EngineError::Invalid(format!(
                        "{} 는 파일입니다",
                        here.display()
                    )));
                }
            }
            // A mount point, or a directory the mount table synthesized: it exists and is a
            // directory, and `ContextFs` answers a create aimed at one with `EROFS`.
            Err(err) if err.kind() == std::io::ErrorKind::ReadOnlyFilesystem => {
                if fs.stat(&here).await?.kind != DirentKind::Dir {
                    return Err(err.into());
                }
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

/// `buf` as a string, or `None` if it is not text.
///
/// A truncated read is allowed to end mid-character — the cap is a byte count and knows nothing
/// about UTF-8 — so an invalid tail there is cut off rather than treated as evidence of binary.
pub(crate) fn as_text(buf: Vec<u8>, truncated: bool) -> Option<String> {
    match String::from_utf8(buf) {
        Ok(text) => Some(text),
        Err(err) if truncated => {
            let valid = err.utf8_error().valid_up_to();
            let mut bytes = err.into_bytes();
            bytes.truncate(valid);
            String::from_utf8(bytes).ok()
        }
        Err(_) => None,
    }
}

pub(crate) fn kind_str(kind: DirentKind) -> &'static str {
    match kind {
        DirentKind::Dir => "dir",
        DirentKind::File => "file",
    }
}

/// `dir` and `name` as one `/`-rooted path.
pub fn join(dir: &str, name: &str) -> String {
    let trimmed = dir.trim_end_matches('/');
    format!("{trimmed}/{name}")
}

#[cfg(test)]
mod tests {
    use cortex::fs::{ContextFs, InMemFs};

    use super::*;

    fn ws() -> ContextFs {
        ContextFs::new().try_with_mount("", InMemFs::new()).unwrap()
    }

    #[tokio::test]
    async fn write_read_list_delete() {
        let fs = ws();
        mkdir(&fs, "/docs/notes").await.unwrap();
        write(&fs, "/docs/notes/a.md", "hello").await.unwrap();
        write(&fs, "/docs/notes/a.md", "hi").await.unwrap(); // shorter rewrite truncates
        let got = read(&fs, "/docs/notes/a.md").await.unwrap();
        assert_eq!(got.text.as_deref(), Some("hi"));
        assert!(!got.truncated);
        let entries = list(&fs, "/docs").await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, "dir");
        assert_eq!(entries[0].path, "/docs/notes");
        rename(&fs, "/docs/notes/a.md", "/docs/notes/b.md")
            .await
            .unwrap();
        delete(&fs, "/docs/notes/b.md").await.unwrap();
        assert!(list(&fs, "/docs/notes").await.unwrap().is_empty());
        assert!(matches!(
            read(&fs, "/docs").await,
            Err(EngineError::Invalid(_))
        ));
    }

    #[tokio::test]
    async fn import_copies_files_and_reports_skips() {
        let fs = ws();
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("x.txt"), b"xx").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/y.txt"), b"y").unwrap();
        let report = import(
            &fs,
            "/",
            vec![
                dir.path().join("x.txt"),
                dir.path().join("sub"),
                dir.path().join("missing"),
            ],
        )
        .await
        .unwrap();
        assert_eq!(report.files, 2);
        assert_eq!(report.bytes, 3);
        assert_eq!(report.skipped.len(), 1);
        assert_eq!(
            read(&fs, "/sub/y.txt").await.unwrap().text.as_deref(),
            Some("y")
        );
    }
}
