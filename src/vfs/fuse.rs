use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use fuser::{
    BsdFileFlags, Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags,
    Generation, INodeNo, LockOwner, MountOption, OpenFlags, RenameFlags, ReplyAttr, ReplyCreate,
    ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
    WriteFlags,
};
use tokio::runtime::Handle;

use crate::vfs::{
    Vfs,
    resource::{FileKind, FileStat},
};

const TTL: Duration = Duration::from_secs(1);
const ROOT_INO: u64 = 1;

/// A live host FUSE mount backed by a [`Vfs`]. Unmounts on drop.
pub struct VfsMount {
    _session: fuser::BackgroundSession,
    mountpoint: PathBuf,
}

impl VfsMount {
    /// Mount `vfs` at `mountpoint` (must exist). `rt` is the tokio runtime
    /// handle used to drive async [`Resource`](crate::vfs::Resource) calls from
    /// the synchronous FUSE callbacks. Returns a guard that unmounts on drop.
    pub fn spawn(vfs: Arc<Vfs>, mountpoint: impl AsRef<Path>, rt: Handle) -> anyhow::Result<Self> {
        let mountpoint = mountpoint.as_ref().to_path_buf();
        let fs = VfsFs::new(vfs, rt);
        // `Config` is #[non_exhaustive]; build via Default + set the public field.
        let mut config = Config::default();
        config.mount_options = vec![
            MountOption::FSName("ailoy-vfs".to_string()),
            MountOption::DefaultPermissions,
        ];
        let session = fuser::spawn_mount2(fs, &mountpoint, &config)?;
        Ok(Self {
            _session: session,
            mountpoint,
        })
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }
}

/// A pending write buffer for one open file. `data` holds the full file content
/// being assembled (preloaded with the existing content so partial writes,
/// appends, and truncates don't clobber the rest); `dirty` tracks whether
/// anything was actually written/truncated, so an open-read-close cycle that
/// never modifies the file is NOT flushed back (which would corrupt rendered /
/// read-only files like a Notion `page.json`).
///
/// `preload_failed` records that the existing-content preload (for a partial
/// write / append / non-zero truncate of an *existing* file) failed transiently
/// (timeout / 5xx / network) — distinct from "file doesn't exist" (a legitimate
/// empty base for a new file). When set, `release` aborts the flush with EIO
/// instead of overwriting the original with only the partial new bytes (R1).
struct WriteBuf {
    data: Vec<u8>,
    dirty: bool,
    preload_failed: bool,
}

/// Mutable bookkeeping. fuser 0.17 `Filesystem` methods take `&self`, so the
/// inode<->path maps and pending write buffers live behind a `Mutex` (the host
/// mount is single-threaded, so contention is nil).
struct FsState {
    ino_path: HashMap<u64, String>,
    path_ino: HashMap<String, u64>,
    next_ino: u64,
    /// Pending write buffers keyed by inode; flushed to the resource on release.
    wbuf: HashMap<u64, WriteBuf>,
    /// Last JSON result of a `/<mount>/.cmd/<op>` write, keyed by that control
    /// path, so the agent can read it back (C4: e.g. the new page id from
    /// `page-create` before `block-append`).
    cmd_results: HashMap<String, Vec<u8>>,
}

impl FsState {
    fn intern(&mut self, path: &str) -> u64 {
        if let Some(&ino) = self.path_ino.get(path) {
            return ino;
        }
        self.next_ino += 1;
        let ino = self.next_ino;
        self.path_ino.insert(path.to_string(), ino);
        self.ino_path.insert(ino, path.to_string());
        ino
    }

    fn path_of(&self, ino: u64) -> Option<String> {
        self.ino_path.get(&ino).cloned()
    }
}

struct VfsFs {
    vfs: Arc<Vfs>,
    rt: Handle,
    state: Mutex<FsState>,
}

impl VfsFs {
    fn new(vfs: Arc<Vfs>, rt: Handle) -> Self {
        let mut ino_path = HashMap::new();
        let mut path_ino = HashMap::new();
        ino_path.insert(ROOT_INO, "/".to_string());
        path_ino.insert("/".to_string(), ROOT_INO);
        Self {
            vfs,
            rt,
            state: Mutex::new(FsState {
                ino_path,
                path_ino,
                next_ino: ROOT_INO,
                wbuf: HashMap::new(),
                cmd_results: HashMap::new(),
            }),
        }
    }

    /// Ensure a [`WriteBuf`] exists for `ino`. `truncate` (e.g. `O_TRUNC`, a `>`
    /// redirect) resets it to empty + dirty without reading. Otherwise the
    /// existing file content is preloaded once (dirty=false) so a later partial
    /// write / append / truncate splices onto it instead of clobbering it.
    /// Must not be called while holding the state lock (it `block_on`s a read).
    fn ensure_wbuf(&self, ino: u64, truncate: bool) {
        if truncate {
            self.state.lock().unwrap().wbuf.insert(
                ino,
                WriteBuf {
                    data: Vec::new(),
                    dirty: true,
                    preload_failed: false,
                },
            );
            return;
        }
        let path = {
            let st = self.state.lock().unwrap();
            if st.wbuf.contains_key(&ino) {
                return;
            }
            st.path_of(ino)
        };
        // Distinguish a transient read failure on an *existing* file from a
        // genuinely-new file. The former must NOT become an empty base (R1):
        // a later partial write + flush would overwrite the original with only
        // the new bytes. Mark it `preload_failed` so `release` aborts the flush.
        let (data, preload_failed) = match path {
            Some(p) => match self.rt.block_on(read_full(&self.vfs, &p)) {
                Ok(d) => (d, false),
                Err(_) => {
                    let exists = self.rt.block_on(stat_path(&self.vfs, &p)).is_ok();
                    (Vec::new(), exists)
                }
            },
            None => (Vec::new(), false),
        };
        self.state
            .lock()
            .unwrap()
            .wbuf
            .entry(ino)
            .or_insert(WriteBuf {
                data,
                dirty: false,
                preload_failed,
            });
    }
}

fn join(parent: &str, name: &str) -> String {
    if parent == "/" {
        format!("/{name}")
    } else {
        format!("{parent}/{name}")
    }
}

fn make_attr(ino: u64, kind: FileKind, size: u64) -> FileAttr {
    let (ftype, perm) = match kind {
        FileKind::Dir => (FileType::Directory, 0o755),
        FileKind::File => (FileType::RegularFile, 0o644),
    };
    let uid = unsafe { libc::getuid() };
    let gid = unsafe { libc::getgid() };
    FileAttr {
        ino: INodeNo(ino),
        size,
        blocks: size.div_ceil(512),
        atime: UNIX_EPOCH,
        mtime: UNIX_EPOCH,
        ctime: UNIX_EPOCH,
        crtime: UNIX_EPOCH,
        kind: ftype,
        perm,
        nlink: if matches!(kind, FileKind::Dir) { 2 } else { 1 },
        uid,
        gid,
        rdev: 0,
        flags: 0,
        blksize: 512,
    }
}

async fn stat_path(vfs: &Vfs, path: &str) -> anyhow::Result<FileStat> {
    if path == "/" {
        return Ok(FileStat {
            kind: FileKind::Dir,
            ..Default::default()
        });
    }
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.stat(&vp).await
}

/// Build a [`FileAttr`], using the backend's mtime when it reports one (S3-1) so
/// `ls -l` / make / rsync see real timestamps instead of the epoch.
fn attr_from_stat(ino: u64, s: &FileStat) -> FileAttr {
    let mut attr = make_attr(ino, s.kind, s.size);
    if let Some(t) = s.mtime {
        attr.mtime = t;
        attr.ctime = t;
    }
    attr
}

async fn list_path(vfs: &Vfs, path: &str) -> anyhow::Result<Vec<(String, FileKind, u64)>> {
    if path == "/" {
        return Ok(vfs
            .mount_names()
            .into_iter()
            .map(|n| (n, FileKind::Dir, 0))
            .collect());
    }
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    let entries = res.readdir(&vp).await?;
    Ok(entries
        .into_iter()
        .map(|e| (e.name, e.kind, e.size))
        .collect())
}

async fn read_path(vfs: &Vfs, path: &str, offset: u64, size: u64) -> anyhow::Result<Vec<u8>> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.read_bytes(&vp, Some(offset..offset + size)).await
}

/// Read the whole current content of `path` (for read-modify-write preload).
async fn read_full(vfs: &Vfs, path: &str) -> anyhow::Result<Vec<u8>> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.read_bytes(&vp, None).await
}

/// Flush a write. A `/<mount>/.cmd/<op>` path is a control write: it runs the
/// domain command and returns its JSON result (for C4 read-back); a normal path
/// writes bytes and returns `None`.
async fn put_path(vfs: &Vfs, path: &str, data: Vec<u8>) -> anyhow::Result<Option<Vec<u8>>> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    if let Some(op) = vp.as_str().strip_prefix("/.cmd/") {
        Ok(Some(res.command(op, &data).await?))
    } else {
        res.write_bytes(&vp, data).await?;
        Ok(None)
    }
}

async fn unlink_path(vfs: &Vfs, path: &str) -> anyhow::Result<()> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.unlink(&vp).await
}

async fn mkdir_path(vfs: &Vfs, path: &str) -> anyhow::Result<()> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.mkdir(&vp).await
}

async fn rmdir_path(vfs: &Vfs, path: &str) -> anyhow::Result<()> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    res.rmdir(&vp).await
}

async fn rename_path(vfs: &Vfs, from: &str, to: &str) -> anyhow::Result<()> {
    let (res, from_vp) = vfs
        .route(from)
        .ok_or_else(|| anyhow::anyhow!("no mount for {from}"))?;
    let (_, to_vp) = vfs
        .route(to)
        .ok_or_else(|| anyhow::anyhow!("no mount for {to}"))?;
    res.rename(&from_vp, &to_vp).await
}

impl Filesystem for VfsFs {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let Some(parent_path) = self.state.lock().unwrap().path_of(parent.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(Errno::ENOENT);
            return;
        };
        let path = join(&parent_path, name);
        // C4: a `.cmd/<op>` path with a stashed result reads back as a file.
        if let Some(len) = self
            .state
            .lock()
            .unwrap()
            .cmd_results
            .get(&path)
            .map(|b| b.len() as u64)
        {
            let ino = self.state.lock().unwrap().intern(&path);
            reply.entry(&TTL, &make_attr(ino, FileKind::File, len), Generation(0));
            return;
        }
        let vfs = self.vfs.clone();
        match self.rt.block_on(stat_path(&vfs, &path)) {
            Ok(s) => {
                let ino = self.state.lock().unwrap().intern(&path);
                reply.entry(&TTL, &attr_from_stat(ino, &s), Generation(0));
            }
            Err(_) => reply.error(Errno::ENOENT),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        let ino = ino.0;
        // Decide under the lock, act (reply / block_on) after releasing it.
        enum Next {
            Buffered(u64),
            Stat(String),
            NotFound,
        }
        let next = {
            let st = self.state.lock().unwrap();
            if let Some(buf) = st.wbuf.get(&ino) {
                Next::Buffered(buf.data.len() as u64)
            } else if let Some(path) = st.path_of(ino) {
                if let Some(b) = st.cmd_results.get(&path) {
                    Next::Buffered(b.len() as u64)
                } else {
                    Next::Stat(path)
                }
            } else {
                Next::NotFound
            }
        };
        match next {
            Next::Buffered(size) => reply.attr(&TTL, &make_attr(ino, FileKind::File, size)),
            Next::NotFound => reply.error(Errno::ENOENT),
            Next::Stat(path) => {
                let vfs = self.vfs.clone();
                match self.rt.block_on(stat_path(&vfs, &path)) {
                    Ok(s) => reply.attr(&TTL, &attr_from_stat(ino, &s)),
                    Err(_) => reply.error(Errno::ENOENT),
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn setattr(
        &self,
        _req: &Request,
        ino: INodeNo,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>,
        _ctime: Option<SystemTime>,
        _fh: Option<FileHandle>,
        _crtime: Option<SystemTime>,
        _chgtime: Option<SystemTime>,
        _bkuptime: Option<SystemTime>,
        _flags: Option<BsdFileFlags>,
        reply: ReplyAttr,
    ) {
        let ino = ino.0;
        // Truncation (e.g. `>` redirect, or an explicit truncate(2)). Preserve
        // the existing content up to the new length instead of zeroing it.
        if let Some(new_size) = size {
            if new_size == 0 {
                self.state.lock().unwrap().wbuf.insert(
                    ino,
                    WriteBuf {
                        data: Vec::new(),
                        dirty: true,
                        preload_failed: false,
                    },
                );
            } else {
                self.ensure_wbuf(ino, false);
                let mut st = self.state.lock().unwrap();
                let wb = st.wbuf.entry(ino).or_insert_with(|| WriteBuf {
                    data: Vec::new(),
                    dirty: false,
                    preload_failed: false,
                });
                wb.data.resize(new_size as usize, 0);
                wb.dirty = true;
            }
            reply.attr(&TTL, &make_attr(ino, FileKind::File, new_size));
            return;
        }
        enum Next {
            Buffered(u64),
            Stat(String),
            NotFound,
        }
        let next = {
            let st = self.state.lock().unwrap();
            if let Some(buf) = st.wbuf.get(&ino) {
                Next::Buffered(buf.data.len() as u64)
            } else if let Some(path) = st.path_of(ino) {
                Next::Stat(path)
            } else {
                Next::NotFound
            }
        };
        match next {
            Next::Buffered(size) => reply.attr(&TTL, &make_attr(ino, FileKind::File, size)),
            Next::NotFound => reply.error(Errno::ENOENT),
            Next::Stat(path) => {
                let vfs = self.vfs.clone();
                match self.rt.block_on(stat_path(&vfs, &path)) {
                    Ok(s) => reply.attr(&TTL, &attr_from_stat(ino, &s)),
                    Err(_) => reply.error(Errno::ENOENT),
                }
            }
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, flags: OpenFlags, reply: ReplyOpen) {
        let write = flags.0 & (libc::O_WRONLY | libc::O_RDWR) != 0;
        if write {
            // Preload existing content (unless O_TRUNC) so partial writes/appends
            // splice onto it; an unwritten close stays non-dirty and isn't flushed.
            let truncate = flags.0 & libc::O_TRUNC != 0;
            self.ensure_wbuf(ino.0, truncate);
        }
        // direct_io: the kernel forwards reads straight through instead of
        // clamping/zero-filling against the stat size. Workspace docs (gdrive)
        // report a generous over-estimate size from stat to avoid a full export
        // per `stat`; without direct_io the kernel would pad reads with zeros up
        // to that size.
        reply.opened(FileHandle(0), FopenFlags::FOPEN_DIRECT_IO);
    }

    fn create(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let Some(parent_path) = self.state.lock().unwrap().path_of(parent.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(Errno::EINVAL);
            return;
        };
        let path = join(&parent_path, name);
        let ino = {
            let mut st = self.state.lock().unwrap();
            let ino = st.intern(&path);
            // New file: empty + dirty so an immediate close still creates it.
            st.wbuf.insert(
                ino,
                WriteBuf {
                    data: Vec::new(),
                    dirty: true,
                    preload_failed: false,
                },
            );
            ino
        };
        reply.created(
            &TTL,
            &make_attr(ino, FileKind::File, 0),
            Generation(0),
            FileHandle(0),
            FopenFlags::FOPEN_DIRECT_IO,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let ino = ino.0;
        // Serve from the write buffer if open for writing; else read the provider.
        enum Next {
            Buffered(Vec<u8>),
            Read(String),
            NotFound,
        }
        let next = {
            let st = self.state.lock().unwrap();
            if let Some(buf) = st.wbuf.get(&ino) {
                let start = (offset as usize).min(buf.data.len());
                let end = (start + size as usize).min(buf.data.len());
                Next::Buffered(buf.data[start..end].to_vec())
            } else if let Some(path) = st.path_of(ino) {
                if let Some(b) = st.cmd_results.get(&path) {
                    let start = (offset as usize).min(b.len());
                    let end = (start + size as usize).min(b.len());
                    Next::Buffered(b[start..end].to_vec())
                } else {
                    Next::Read(path)
                }
            } else {
                Next::NotFound
            }
        };
        match next {
            Next::Buffered(data) => reply.data(&data),
            Next::NotFound => reply.error(Errno::ENOENT),
            Next::Read(path) => {
                let vfs = self.vfs.clone();
                match self
                    .rt
                    .block_on(read_path(&vfs, &path, offset, size as u64))
                {
                    Ok(data) => reply.data(&data),
                    Err(_) => reply.error(Errno::EIO),
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        data: &[u8],
        _write_flags: WriteFlags,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyWrite,
    ) {
        // Preload existing content first (no-op if already buffered) so a write
        // at a non-zero offset / append doesn't NUL-fill the untouched bytes.
        self.ensure_wbuf(ino.0, false);
        let mut st = self.state.lock().unwrap();
        let wb = st.wbuf.entry(ino.0).or_insert_with(|| WriteBuf {
            data: Vec::new(),
            dirty: false,
            preload_failed: false,
        });
        let off = offset as usize;
        if off > wb.data.len() {
            wb.data.resize(off, 0);
        }
        let end = off + data.len();
        if end > wb.data.len() {
            wb.data.resize(end, 0);
        }
        wb.data[off..end].copy_from_slice(data);
        wb.dirty = true;
        reply.written(data.len() as u32);
    }

    fn flush(
        &self,
        _req: &Request,
        _ino: INodeNo,
        _fh: FileHandle,
        _lock_owner: LockOwner,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn release(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        let ino = ino.0;
        let taken = {
            let mut st = self.state.lock().unwrap();
            st.wbuf.remove(&ino).map(|wb| (wb, st.path_of(ino)))
        };
        let Some((wb, path)) = taken else {
            reply.ok();
            return;
        };
        // The existing-content preload failed transiently on a file that does
        // exist — flushing now would overwrite the original with only the
        // partial new bytes. Abort with EIO and preserve the original (R1).
        if wb.preload_failed {
            reply.error(Errno::EIO);
            return;
        }
        // Nothing was written/truncated — don't flush (would clobber a
        // read-only/rendered file just because it was opened for writing).
        if !wb.dirty {
            reply.ok();
            return;
        }
        let Some(path) = path else {
            reply.error(Errno::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        match self.rt.block_on(put_path(&vfs, &path, wb.data)) {
            Ok(result) => {
                if let Some(bytes) = result {
                    // C4: stash the command's JSON result so a read of this
                    // `.cmd/<op>` path returns it.
                    self.state.lock().unwrap().cmd_results.insert(path, bytes);
                }
                reply.ok()
            }
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let ino = ino.0;
        let Some(path) = self.state.lock().unwrap().path_of(ino) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        let entries = match self.rt.block_on(list_path(&vfs, &path)) {
            Ok(e) => e,
            Err(_) => {
                reply.error(Errno::EIO);
                return;
            }
        };
        let mut rows: Vec<(u64, FileType, String)> = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ROOT_INO, FileType::Directory, "..".to_string()),
        ];
        {
            let mut st = self.state.lock().unwrap();
            for (name, kind, _size) in entries {
                let child = join(&path, &name);
                let child_ino = st.intern(&child);
                let ftype = match kind {
                    FileKind::Dir => FileType::Directory,
                    FileKind::File => FileType::RegularFile,
                };
                rows.push((child_ino, ftype, name));
            }
        }
        for (i, (e_ino, ftype, name)) in rows.into_iter().enumerate().skip(offset as usize) {
            if reply.add(INodeNo(e_ino), (i + 1) as u64, ftype, name) {
                break;
            }
        }
        reply.ok();
    }

    fn unlink(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let Some(parent_path) = self.state.lock().unwrap().path_of(parent.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(Errno::EINVAL);
            return;
        };
        let path = join(&parent_path, name);
        let vfs = self.vfs.clone();
        match self.rt.block_on(unlink_path(&vfs, &path)) {
            Ok(()) => reply.ok(),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn mkdir(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let Some(parent_path) = self.state.lock().unwrap().path_of(parent.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(Errno::EINVAL);
            return;
        };
        let path = join(&parent_path, name);
        let vfs = self.vfs.clone();
        match self.rt.block_on(mkdir_path(&vfs, &path)) {
            Ok(()) => {
                let ino = self.state.lock().unwrap().intern(&path);
                reply.entry(&TTL, &make_attr(ino, FileKind::Dir, 0), Generation(0));
            }
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn rmdir(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let Some(parent_path) = self.state.lock().unwrap().path_of(parent.0) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(Errno::EINVAL);
            return;
        };
        let path = join(&parent_path, name);
        let vfs = self.vfs.clone();
        match self.rt.block_on(rmdir_path(&vfs, &path)) {
            Ok(()) => reply.ok(),
            Err(_) => reply.error(Errno::EIO),
        }
    }

    fn rename(
        &self,
        _req: &Request,
        parent: INodeNo,
        name: &OsStr,
        newparent: INodeNo,
        newname: &OsStr,
        _flags: RenameFlags,
        reply: ReplyEmpty,
    ) {
        let (from_parent, to_parent) = {
            let st = self.state.lock().unwrap();
            (st.path_of(parent.0), st.path_of(newparent.0))
        };
        let (Some(from_parent), Some(to_parent)) = (from_parent, to_parent) else {
            reply.error(Errno::ENOENT);
            return;
        };
        let (Some(name), Some(newname)) = (name.to_str(), newname.to_str()) else {
            reply.error(Errno::EINVAL);
            return;
        };
        let from = join(&from_parent, name);
        let to = join(&to_parent, newname);
        let vfs = self.vfs.clone();
        match self.rt.block_on(rename_path(&vfs, &from, &to)) {
            Ok(()) => reply.ok(),
            Err(_) => reply.error(Errno::EIO),
        }
    }
}
