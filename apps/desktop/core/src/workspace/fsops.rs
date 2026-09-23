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
//!
//! # Timing
//!
//! Every store call this makes is timed onto the `fs_timing` target, one line each, because
//! a remote source charges per call and a cache is only worth what the calls cost. The split
//! matters as much as the total: a read is a `stat` and then a body, and which of the two
//! dominates decides whether a cache should hold metadata, bytes, or both.
//!
//! These are the *window's* numbers. The agent reads through the FUSE mount, which serves
//! the same `ContextFs` without passing through here, so nothing below sees it.
//!
//! `RUST_LOG=fs_timing=off` turns the lines off without a rebuild; `scripts/fs-timings.py`
//! summarises a log that has them.

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

/// Time one store call, and say what it cost.
///
/// `size` describes the answer in whatever unit the operation deals in — entries for a
/// listing, bytes for a read — so one line says both how long a call took and how much it
/// was for. A failure is logged too, at the same level: an error that took four seconds is
/// a latency measurement like any other, and a source that is failing slowly is exactly what
/// this is here to catch.
async fn timed<T, E: std::fmt::Display>(
    op: &'static str,
    path: &str,
    call: impl Future<Output = std::result::Result<T, E>>,
    size: impl FnOnce(&T) -> u64,
) -> std::result::Result<T, E> {
    let started = std::time::Instant::now();
    let out = call.await;
    let ms = started.elapsed().as_millis();
    match &out {
        Ok(v) => tracing::info!(target: "fs_timing", op, path, ms, n = size(v)),
        Err(e) => tracing::info!(target: "fs_timing", op, path, ms, err = %e),
    }
    out
}

/// A `SystemTime` as milliseconds since the epoch, or `None` for one that predates it.
fn epoch_ms(t: std::time::SystemTime) -> Option<u64> {
    t.duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// The children of `path`.
pub async fn list(fs: &dyn FileSystem, path: &str) -> Result<Vec<Entry>> {
    let mut entries = Vec::new();
    let listed = timed("list", path, fs.list(Path::new(path)), |v| v.len() as u64).await?;
    for dirent in listed {
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
    let stat = timed("stat", path, fs.stat(target), |s| s.size).await?;
    if stat.kind == DirentKind::Dir {
        return Err(EngineError::Invalid(
            "A directory cannot be opened in the editor".into(),
        ));
    }

    let buf = timed(
        "read",
        path,
        read_all(fs, target, stat.size.min(READ_CAP)),
        |b| b.len() as u64,
    )
    .await?;
    let truncated = stat.size > READ_CAP;
    let (text, encoding) = match as_text(buf, truncated) {
        Some((text, encoding)) => (Some(text), Some(encoding.to_string())),
        None => (None, None),
    };
    Ok(FileContent {
        path: path.to_string(),
        text,
        encoding,
        size: stat.size,
        truncated,
    })
}

/// How much of a file may be handed over as bytes.
///
/// Far above [`READ_CAP`], because these are the formats whose whole point is that they are
/// not text: a scanned PDF or a deck of photographs is tens of megabytes and is still one
/// document. The cap is here so that a mistake — a disk image, a video — is refused rather
/// than read into memory twice on its way to the window.
const BYTES_CAP: u64 = 64 << 20;

/// The contents of `path` as bytes, whatever they are.
///
/// The sibling of [`read`], for the viewers that open a format rather than read characters.
/// Where `read` decodes and gives up on anything that is not text, this gives up on nothing
/// and decides nothing: what the bytes mean is the caller's to work out from the name.
pub async fn read_bytes(fs: &dyn FileSystem, path: &str) -> Result<Vec<u8>> {
    let target = Path::new(path);
    let stat = timed("stat", path, fs.stat(target), |s| s.size).await?;
    if stat.kind == DirentKind::Dir {
        return Err(EngineError::Invalid(
            "A directory cannot be opened in the editor".into(),
        ));
    }
    if stat.size > BYTES_CAP {
        return Err(EngineError::Invalid(format!(
            "{}: {} MiB — files above {} MiB are not opened here",
            path,
            stat.size >> 20,
            BYTES_CAP >> 20
        )));
    }
    timed("read", path, read_all(fs, target, stat.size), |b| {
        b.len() as u64
    })
    .await
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
            "Files can only be dropped onto a directory".into(),
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
                    .push(format!("{}: could not read the name", source.display()));
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
                "{}: {} MiB — imports stop at {} MiB per file",
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
            return Err(EngineError::Invalid(
                "A directory cannot be written to".into(),
            ));
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
                "The store is not accepting further writes".into(),
            ));
        }
        written += n;
    }
    fs.flush(path).await?;
    Ok(())
}

/// Read `size` bytes of `path`, in whatever number of calls the store answers in.
///
/// The size is the caller's, not one this asks for. Every caller has just `stat`ed the file —
/// to refuse a directory, to cap the read, to report a size — and on a remote store that stat
/// is a request: a `head` against S3, a page render against Notion. Asking again here made
/// every read cost two of them, which the `fs_timing` lines showed as exactly two `stat`s per
/// `read` on both sources.
async fn read_all(fs: &dyn FileSystem, path: &Path, size: u64) -> Result<Vec<u8>> {
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
                        "{} is a file",
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
/// `buf` as characters, with the encoding worked out from the bytes.
///
/// UTF-8 is not the only thing on a disk. A Korean spreadsheet exported from anything
/// Windows-hosted is CP949, and read as UTF-8 it is not an error anywhere — it is a file
/// that decodes to nothing and is shown as binary, or to mojibake nobody can read.
///
/// The order is what keeps the inference honest in both directions. A byte order mark
/// settles it outright. Failing that UTF-8 is tried *strictly*, so a plain ASCII file —
/// identical bytes in every encoding here — is never "detected" as Korean; and Korean text
/// in CP949 is not valid UTF-8, so it falls through on the first Hangul syllable rather
/// than on a heuristic. Only a file that fails UTF-8 is tried as CP949, and only a file
/// that fails both is binary.
///
/// A truncated read is the one place errors are tolerated: the cap lands mid-character, so
/// the tail is dropped for UTF-8 and replaced for CP949 rather than losing the whole file
/// over its last few bytes.
pub fn as_text(buf: Vec<u8>, truncated: bool) -> Option<(String, &'static str)> {
    // The marks Windows editors write, which no UTF-8 decoder recovers from.
    if let Some(rest) = buf.strip_prefix(&[0xFF, 0xFE]) {
        let (text, _, _) = encoding_rs::UTF_16LE.decode(rest);
        return Some((text.into_owned(), "UTF-16"));
    }
    if let Some(rest) = buf.strip_prefix(&[0xFE, 0xFF]) {
        let (text, _, _) = encoding_rs::UTF_16BE.decode(rest);
        return Some((text.into_owned(), "UTF-16"));
    }

    let buf = match String::from_utf8(buf) {
        Ok(text) => return Some((text, "UTF-8")),
        Err(err) => {
            let valid = err.utf8_error().valid_up_to();
            let bytes = err.into_bytes();
            // A UTF-8 file that the cap cut mid-character fails within a character's
            // length of the end, and nowhere else. Anything failing earlier is not UTF-8
            // that got cut — it is another encoding, and taking the prefix would hand back
            // whatever happened to precede the first Hangul syllable, which for a Korean
            // file is the empty string.
            if truncated && valid + MAX_UTF8_CHAR >= bytes.len() {
                let mut prefix = bytes;
                prefix.truncate(valid);
                return String::from_utf8(prefix).ok().map(|t| (t, "UTF-8"));
            }
            bytes
        }
    };

    // `EUC_KR` is the label; the index behind it is Windows-949, which is what these files
    // actually use — EUC-KR plus the syllables it left out.
    let (text, _, had_errors) = encoding_rs::EUC_KR.decode(&buf);
    // CP949 accepts almost any byte, so "it decoded" is not evidence of anything the way
    // valid UTF-8 is. A PNG header decodes to characters too. What separates a document
    // from a binary is that a document has no control bytes in it.
    if had_errors && !truncated {
        return None;
    }
    // A cut file ends mid-character, so the decoder's replacement there is expected and is
    // not evidence of anything. Everywhere else it still is.
    let body = if truncated {
        text.trim_end_matches('\u{FFFD}')
    } else {
        &text
    };
    if !looks_like_text(body) {
        return None;
    }
    Some((body.to_string(), "CP949"))
}

/// The longest a single UTF-8 character can be, which is how far from the end a cut file
/// is allowed to stop being valid.
const MAX_UTF8_CHAR: usize = 4;

/// Whether decoded characters read as a document rather than as bytes that happened to map.
///
/// Control characters are the tell: tab, newline and carriage return belong in text and
/// nothing else in that range does. The replacement character counts against it too — it is
/// what a decoder emits for a byte it could not place.
fn looks_like_text(text: &str) -> bool {
    !text
        .chars()
        .any(|c| (c.is_control() && c != '\t' && c != '\n' && c != '\r') || c == '\u{FFFD}')
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
    use super::as_text;

    /// A `fmt` layer writing into a buffer this test can read back.
    #[derive(Clone, Default)]
    struct Captured(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Captured {
        type Writer = Captured;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Every store call says what it cost, in the shape the summary script reads.
    ///
    /// `scripts/fs-timings.py` parses `op=`, `path=`, `ms=` and `n=` out of these lines, and
    /// nothing else in the codebase would notice if a tracing upgrade — or a careless edit —
    /// spelled them differently. A run that measures nothing looks exactly like a run that
    /// was fast, which is the failure this is here to make loud.
    #[test]
    fn every_store_call_says_what_it_cost() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"hello").unwrap();
        let fs = cortex::fs::PassthroughFs::new(dir.path());

        let captured = Captured::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(captured.clone())
            .with_ansi(false)
            .without_time()
            .finish();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        tracing::subscriber::with_default(subscriber, || {
            rt.block_on(async {
                super::list(&fs, "/").await.unwrap();
                super::read(&fs, "/a.txt").await.unwrap();
            });
        });

        let log = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
        let line = |op: &str| {
            log.lines()
                .find(|l| l.contains("fs_timing") && l.contains(&format!("op=\"{op}\"")))
                .unwrap_or_else(|| panic!("no {op} line in:\n{log}"))
                .to_string()
        };

        let listing = line("list");
        assert!(listing.contains("path=\"/\""), "{listing}");
        assert!(
            listing.contains("n=1"),
            "one entry in the directory: {listing}"
        );
        assert!(listing.contains("ms="), "{listing}");

        // A read is a `stat` and then a body, and the split is the point: which of the two
        // costs is what a cache would have to hold.
        assert!(line("stat").contains("path=\"/a.txt\""));
        // One stat, not two. `read_all` used to ask for the size its caller had just paid a
        // `head` (or a page render) for, which doubled the cost of every read on a remote
        // store — the kind of waste that only shows up once something counts.
        assert_eq!(
            log.lines().filter(|l| l.contains("op=\"stat\"")).count(),
            1,
            "{log}"
        );
        assert!(
            line("read").contains("n=5"),
            "the body's bytes: {}",
            line("read")
        );
    }

    /// The bytes a Windows-hosted Korean export actually contains.
    fn cp949(text: &str) -> Vec<u8> {
        let (bytes, _, had_errors) = encoding_rs::EUC_KR.encode(text);
        assert!(!had_errors, "the fixture has to be encodable");
        bytes.into_owned()
    }

    #[test]
    fn text_is_decoded_as_what_it_was_written_in() {
        // The ordinary case, and the one every other case must not be mistaken for.
        assert_eq!(
            as_text("hello".as_bytes().to_vec(), false),
            Some(("hello".into(), "UTF-8"))
        );
        assert_eq!(
            as_text("안녕".as_bytes().to_vec(), false),
            Some(("안녕".into(), "UTF-8"))
        );

        // The file this was written for: Korean, CP949, not valid UTF-8 anywhere in it.
        let korean = "구분,고급휘발유\n에쓰오일,1798";
        let bytes = cp949(korean);
        assert!(
            String::from_utf8(bytes.clone()).is_err(),
            "or it proves nothing"
        );
        assert_eq!(as_text(bytes, false), Some((korean.into(), "CP949")));

        // ASCII is the same bytes in both, so plain English is never "detected" as Korean.
        assert_eq!(
            as_text(b"a,b,c\n1,2,3".to_vec(), false),
            Some(("a,b,c\n1,2,3".into(), "UTF-8"))
        );
    }

    #[test]
    fn a_byte_order_mark_settles_it() {
        let mut le = vec![0xFF, 0xFE];
        for u in "안녕".encode_utf16() {
            le.extend_from_slice(&u.to_le_bytes());
        }
        assert_eq!(as_text(le, false), Some(("안녕".into(), "UTF-16")));

        let mut be = vec![0xFE, 0xFF];
        for u in "안녕".encode_utf16() {
            be.extend_from_slice(&u.to_be_bytes());
        }
        assert_eq!(as_text(be, false), Some(("안녕".into(), "UTF-16")));
    }

    #[test]
    fn bytes_that_are_no_encoding_stay_binary() {
        // A PNG header: not UTF-8, and not text in CP949 either.
        let png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00];
        assert_eq!(as_text(png, false), None);
    }

    #[test]
    fn a_cut_file_keeps_what_it_has() {
        // The cap lands mid-character. UTF-8 drops the partial tail...
        let mut cut = "안녕하세요".as_bytes().to_vec();
        cut.truncate(cut.len() - 1);
        let (text, encoding) = as_text(cut, true).expect("a clean prefix is still text");
        assert_eq!(encoding, "UTF-8");
        assert!(text.starts_with("안녕하세"), "{text}");

        // ...and CP949 is allowed its replacement rather than losing the file over a byte.
        let mut cut = cp949("가나다");
        cut.truncate(cut.len() - 1);
        let (_, encoding) = as_text(cut, true).expect("still text");
        assert_eq!(encoding, "CP949");
    }
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
