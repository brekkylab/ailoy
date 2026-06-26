use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use fuser::{
    BsdFileFlags, Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags, Generation,
    INodeNo, LockOwner, MountOption, OpenFlags, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow, WriteFlags,
};
use tokio::runtime::Handle;

use crate::vfs::{Vfs, resource::FileKind};

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

/// Mutable bookkeeping. fuser 0.17 `Filesystem` methods take `&self`, so the
/// inode<->path maps and pending write buffers live behind a `Mutex` (the host
/// mount is single-threaded, so contention is nil).
struct FsState {
    ino_path: HashMap<u64, String>,
    path_ino: HashMap<String, u64>,
    next_ino: u64,
    /// Pending write buffers keyed by inode; flushed to the resource on release.
    wbuf: HashMap<u64, Vec<u8>>,
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
            }),
        }
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

async fn stat_path(vfs: &Vfs, path: &str) -> anyhow::Result<(FileKind, u64)> {
    if path == "/" {
        return Ok((FileKind::Dir, 0));
    }
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    let s = res.stat(&vp).await?;
    Ok((s.kind, s.size))
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

async fn put_path(vfs: &Vfs, path: &str, data: Vec<u8>) -> anyhow::Result<()> {
    let (res, vp) = vfs
        .route(path)
        .ok_or_else(|| anyhow::anyhow!("no mount for {path}"))?;
    if let Some(op) = vp.as_str().strip_prefix("/.cmd/") {
        res.command(op, &data).await?;
        Ok(())
    } else {
        res.write_bytes(&vp, data).await
    }
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
        let vfs = self.vfs.clone();
        match self.rt.block_on(stat_path(&vfs, &path)) {
            Ok((kind, size)) => {
                let ino = self.state.lock().unwrap().intern(&path);
                reply.entry(&TTL, &make_attr(ino, kind, size), Generation(0));
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
                Next::Buffered(buf.len() as u64)
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
                    Ok((kind, size)) => reply.attr(&TTL, &make_attr(ino, kind, size)),
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
        // Only truncation matters for our write path (e.g. `>` redirect).
        if let Some(new_size) = size {
            self.state
                .lock()
                .unwrap()
                .wbuf
                .entry(ino)
                .or_default()
                .resize(new_size as usize, 0);
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
                Next::Buffered(buf.len() as u64)
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
                    Ok((kind, sz)) => reply.attr(&TTL, &make_attr(ino, kind, sz)),
                    Err(_) => reply.error(Errno::ENOENT),
                }
            }
        }
    }

    fn open(&self, _req: &Request, ino: INodeNo, flags: OpenFlags, reply: ReplyOpen) {
        let write = flags.0 & (libc::O_WRONLY | libc::O_RDWR) != 0;
        if write {
            self.state.lock().unwrap().wbuf.entry(ino.0).or_default();
        }
        reply.opened(FileHandle(0), FopenFlags::empty());
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
            st.wbuf.insert(ino, Vec::new());
            ino
        };
        reply.created(
            &TTL,
            &make_attr(ino, FileKind::File, 0),
            Generation(0),
            FileHandle(0),
            FopenFlags::empty(),
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
                let start = (offset as usize).min(buf.len());
                let end = (start + size as usize).min(buf.len());
                Next::Buffered(buf[start..end].to_vec())
            } else if let Some(path) = st.path_of(ino) {
                Next::Read(path)
            } else {
                Next::NotFound
            }
        };
        match next {
            Next::Buffered(data) => reply.data(&data),
            Next::NotFound => reply.error(Errno::ENOENT),
            Next::Read(path) => {
                let vfs = self.vfs.clone();
                match self.rt.block_on(read_path(&vfs, &path, offset, size as u64)) {
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
        let mut st = self.state.lock().unwrap();
        let buf = st.wbuf.entry(ino.0).or_default();
        let off = offset as usize;
        if off > buf.len() {
            buf.resize(off, 0);
        }
        let end = off + data.len();
        if end > buf.len() {
            buf.resize(end, 0);
        }
        buf[off..end].copy_from_slice(data);
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
            st.wbuf.remove(&ino).map(|data| (data, st.path_of(ino)))
        };
        let Some((data, path)) = taken else {
            reply.ok();
            return;
        };
        let Some(path) = path else {
            reply.error(Errno::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        match self.rt.block_on(put_path(&vfs, &path, data)) {
            Ok(()) => reply.ok(),
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
}
