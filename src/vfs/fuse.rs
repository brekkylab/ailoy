use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, UNIX_EPOCH},
};

use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory,
    ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
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
        let options = vec![
            MountOption::FSName("ailoy-vfs".to_string()),
            MountOption::DefaultPermissions,
        ];
        let session = fuser::spawn_mount2(fs, &mountpoint, &options)?;
        Ok(Self {
            _session: session,
            mountpoint,
        })
    }

    pub fn mountpoint(&self) -> &Path {
        &self.mountpoint
    }
}

struct VfsFs {
    vfs: Arc<Vfs>,
    rt: Handle,
    ino_path: HashMap<u64, String>,
    path_ino: HashMap<String, u64>,
    next_ino: u64,
    /// Pending write buffers keyed by inode; flushed to the resource on release.
    wbuf: HashMap<u64, Vec<u8>>,
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
            ino_path,
            path_ino,
            next_ino: ROOT_INO,
            wbuf: HashMap::new(),
        }
    }

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
        ino,
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
    fn lookup(&mut self, _req: &Request<'_>, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(parent_path) = self.path_of(parent) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(libc::ENOENT);
            return;
        };
        let path = join(&parent_path, name);
        let vfs = self.vfs.clone();
        match self.rt.block_on(stat_path(&vfs, &path)) {
            Ok((kind, size)) => {
                let ino = self.intern(&path);
                reply.entry(&TTL, &make_attr(ino, kind, size), 0);
            }
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    fn getattr(&mut self, _req: &Request<'_>, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let Some(path) = self.path_of(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        if let Some(buf) = self.wbuf.get(&ino) {
            reply.attr(&TTL, &make_attr(ino, FileKind::File, buf.len() as u64));
            return;
        }
        let vfs = self.vfs.clone();
        match self.rt.block_on(stat_path(&vfs, &path)) {
            Ok((kind, size)) => reply.attr(&TTL, &make_attr(ino, kind, size)),
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    fn setattr(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _mode: Option<u32>,
        _uid: Option<u32>,
        _gid: Option<u32>,
        size: Option<u64>,
        _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>,
        _ctime: Option<std::time::SystemTime>,
        _fh: Option<u64>,
        _crtime: Option<std::time::SystemTime>,
        _chgtime: Option<std::time::SystemTime>,
        _bkuptime: Option<std::time::SystemTime>,
        _flags: Option<u32>,
        reply: ReplyAttr,
    ) {
        // Only truncation matters for our write path (e.g. `>` redirect).
        if let Some(new_size) = size {
            let buf = self.wbuf.entry(ino).or_default();
            buf.resize(new_size as usize, 0);
            reply.attr(&TTL, &make_attr(ino, FileKind::File, new_size));
            return;
        }
        let Some(path) = self.path_of(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        if let Some(buf) = self.wbuf.get(&ino) {
            reply.attr(&TTL, &make_attr(ino, FileKind::File, buf.len() as u64));
            return;
        }
        let vfs = self.vfs.clone();
        match self.rt.block_on(stat_path(&vfs, &path)) {
            Ok((kind, sz)) => reply.attr(&TTL, &make_attr(ino, kind, sz)),
            Err(_) => reply.error(libc::ENOENT),
        }
    }

    fn open(&mut self, _req: &Request<'_>, ino: u64, flags: i32, reply: ReplyOpen) {
        let write = flags & (libc::O_WRONLY | libc::O_RDWR) != 0;
        if write {
            self.wbuf.entry(ino).or_default();
        }
        reply.opened(0, 0);
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        parent: u64,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let Some(parent_path) = self.path_of(parent) else {
            reply.error(libc::ENOENT);
            return;
        };
        let Some(name) = name.to_str() else {
            reply.error(libc::EINVAL);
            return;
        };
        let path = join(&parent_path, name);
        let ino = self.intern(&path);
        self.wbuf.insert(ino, Vec::new());
        reply.created(&TTL, &make_attr(ino, FileKind::File, 0), 0, 0, 0);
    }

    fn read(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        size: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyData,
    ) {
        if let Some(buf) = self.wbuf.get(&ino) {
            let start = (offset as usize).min(buf.len());
            let end = (start + size as usize).min(buf.len());
            reply.data(&buf[start..end]);
            return;
        }
        let Some(path) = self.path_of(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        match self
            .rt
            .block_on(read_path(&vfs, &path, offset as u64, size as u64))
        {
            Ok(data) => reply.data(&data),
            Err(_) => reply.error(libc::EIO),
        }
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        data: &[u8],
        _write_flags: u32,
        _flags: i32,
        _lock_owner: Option<u64>,
        reply: ReplyWrite,
    ) {
        let buf = self.wbuf.entry(ino).or_default();
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
        &mut self,
        _req: &Request<'_>,
        _ino: u64,
        _fh: u64,
        _lock_owner: u64,
        reply: ReplyEmpty,
    ) {
        reply.ok();
    }

    fn release(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        _flags: i32,
        _lock_owner: Option<u64>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        let Some(data) = self.wbuf.remove(&ino) else {
            reply.ok();
            return;
        };
        let Some(path) = self.path_of(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        match self.rt.block_on(put_path(&vfs, &path, data)) {
            Ok(()) => reply.ok(),
            Err(_) => reply.error(libc::EIO),
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        ino: u64,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let Some(path) = self.path_of(ino) else {
            reply.error(libc::ENOENT);
            return;
        };
        let vfs = self.vfs.clone();
        let entries = match self.rt.block_on(list_path(&vfs, &path)) {
            Ok(e) => e,
            Err(_) => {
                reply.error(libc::EIO);
                return;
            }
        };
        let mut rows: Vec<(u64, FileType, String)> = vec![
            (ino, FileType::Directory, ".".to_string()),
            (ROOT_INO, FileType::Directory, "..".to_string()),
        ];
        for (name, kind, _size) in entries {
            let child = join(&path, &name);
            let child_ino = self.intern(&child);
            let ftype = match kind {
                FileKind::Dir => FileType::Directory,
                FileKind::File => FileType::RegularFile,
            };
            rows.push((child_ino, ftype, name));
        }
        for (i, (e_ino, ftype, name)) in rows.into_iter().enumerate().skip(offset as usize) {
            if reply.add(e_ino, (i + 1) as i64, ftype, name) {
                break;
            }
        }
        reply.ok();
    }
}
