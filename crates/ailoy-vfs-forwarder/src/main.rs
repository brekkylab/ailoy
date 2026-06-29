// Static, dependency-free in-guest VFS forwarder. Mounts a FUSE filesystem and
// forwards operations to the host forward server over plain HTTP (no TLS, no
// libfuse, no python). Cross-compiled to <arch>-linux-musl; needs only
// /dev/fuse (built into the guest kernel) and runs as root.
use std::{
    collections::{HashMap, VecDeque},
    ffi::OsStr,
    io::{Read, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Append a timestamped diagnostic line to /tmp/ailoy-vfs-fwd.log when VFS_DIAG
/// is set. /tmp is a persisted volume in the sandbox, so the log survives a VM
/// restart and can be inspected after a hang. No-op unless VFS_DIAG=1.
fn diag(msg: &str) {
    static ON: OnceLock<bool> = OnceLock::new();
    static T0: OnceLock<Instant> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("VFS_DIAG").as_deref() == Ok("1")) {
        return;
    }
    let t0 = T0.get_or_init(Instant::now);
    let ms = t0.elapsed().as_millis();
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/ailoy-vfs-fwd.log")
    {
        let _ = writeln!(f, "[{ms:>8}ms] {msg}");
    }
}

use fuser::{
    BsdFileFlags, Config, Errno, FileAttr, FileHandle, FileType, Filesystem, FopenFlags,
    Generation, INodeNo, LockOwner, MountOption, OpenFlags, RenameFlags, ReplyAttr, ReplyCreate,
    ReplyData, ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
    WriteFlags,
};

const TTL: Duration = Duration::from_secs(1);

fn host_port() -> String {
    // VFS_HOST = http://host.microsandbox.internal:PORT
    std::env::var("VFS_HOST")
        .unwrap_or_default()
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string()
}

fn token() -> String {
    std::env::var("VFS_TOKEN").unwrap_or_default()
}

fn pct(s: &str) -> String {
    let mut o = String::new();
    for b in s.as_bytes() {
        let c = *b;
        if c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'/' | b'~') {
            o.push(c as char);
        } else {
            o.push_str(&format!("%{c:02X}"));
        }
    }
    o
}

/// Resolve `host.microsandbox.internal:<port>` ONCE and cache it.
///
/// `to_socket_addrs` is itself unbounded and the lookup goes guest→host, so
/// resolving per request means a degraded guest→host channel hangs *every* FUSE
/// op forever at the resolve step (the symptom: `ls` on the mount never returns).
/// The gateway address is stable for the VM's lifetime, so resolve it a single
/// time — each attempt bounded by a watchdog thread, since `to_socket_addrs`
/// can't be cancelled — and reuse the cached addr. Thereafter the only network
/// wait is the bounded `connect_timeout` below, so a dead channel fails in ~8s
/// (EIO) instead of hanging indefinitely.
fn host_addr() -> std::io::Result<SocketAddr> {
    static ADDR: OnceLock<SocketAddr> = OnceLock::new();
    if let Some(a) = ADDR.get() {
        return Ok(*a);
    }
    let hp = host_port();
    for attempt in 0..6 {
        let target = hp.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(target.to_socket_addrs().ok().and_then(|mut it| it.next()));
        });
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Some(a)) => return Ok(*ADDR.get_or_init(|| a)),
            _ => diag(&format!("resolve {hp} attempt {attempt} failed/timed out")),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "resolve host timed out",
    ))
}

/// Minimal HTTP/1.1 client over raw TCP. Returns (status, body).
///
/// Bounded at every step so a host server that is gone/unreachable surfaces as a
/// fast error instead of wedging the FUSE op (and any process touching the mount)
/// indefinitely: name resolution (cached + watchdog-bounded), connect, and
/// read/write are all capped.
fn http(
    method: &str,
    route: &str,
    query: &str,
    body: Option<&[u8]>,
) -> std::io::Result<(u16, Vec<u8>)> {
    diag(&format!("http {method} {route} -> resolve"));
    let addr = host_addr()?;
    diag(&format!("http {method} {route} -> connect {addr}"));
    let t = Instant::now();
    let mut s = TcpStream::connect_timeout(&addr, Duration::from_secs(8))?;
    diag(&format!(
        "http {method} {route} -> connected in {}ms",
        t.elapsed().as_millis()
    ));
    // Short per-syscall timeout so each read()/write() returns promptly; the
    // overall request is bounded by an explicit wall-clock deadline below. We do
    // NOT rely on read_to_end + SO_RCVTIMEO alone: on this musl build a recv
    // timeout did not reliably abort a stalled read_to_end (observed a multi-
    // minute hang), which would wedge the FUSE op — and the whole mount — forever.
    s.set_read_timeout(Some(Duration::from_secs(5)))?;
    s.set_write_timeout(Some(Duration::from_secs(5)))?;
    let mut head = format!(
        "{method} {route}?{query} HTTP/1.1\r\nHost: vfs\r\nx-vfs-token: {}\r\nConnection: close\r\n",
        token()
    );
    if let Some(b) = body {
        head.push_str(&format!(
            "Content-Type: application/octet-stream\r\nContent-Length: {}\r\n",
            b.len()
        ));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    if let Some(b) = body {
        s.write_all(b)?;
    }
    s.flush()?;
    diag(&format!("http {method} {route} -> sent, reading"));
    // Read until EOF, but enforce a hard wall-clock deadline: a host that
    // accepted the connection but never answers must not block this FUSE op
    // forever. WouldBlock/TimedOut/Interrupted just means "no data yet" — keep
    // going until either EOF or the deadline.
    let mut resp = Vec::new();
    let rt = Instant::now();
    let deadline = Duration::from_secs(45);
    let mut buf = [0u8; 16384];
    loop {
        if rt.elapsed() > deadline {
            diag(&format!(
                "http {method} {route} -> DEADLINE after {}ms ({} bytes)",
                rt.elapsed().as_millis(),
                resp.len()
            ));
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "read deadline exceeded",
            ));
        }
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => resp.extend_from_slice(&buf[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(e) => {
                diag(&format!(
                    "http {method} {route} -> READ ERR {e} after {}ms",
                    rt.elapsed().as_millis()
                ));
                return Err(e);
            }
        }
    }
    diag(&format!(
        "http {method} {route} -> read {} bytes in {}ms",
        resp.len(),
        rt.elapsed().as_millis()
    ));
    let sep = resp
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .unwrap_or(resp.len());
    let head_s = String::from_utf8_lossy(&resp[..sep]);
    let status = head_s
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .unwrap_or(0);
    let bstart = (sep + 4).min(resp.len());
    Ok((status, resp[bstart..].to_vec()))
}

fn json_str<'a>(j: &'a str, key: &str) -> Option<&'a str> {
    // tiny scan for "key": "value" or "key": value (bool/number) — body is our
    // own forward server's compact JSON, so a minimal parser suffices.
    let pat = format!("\"{key}\":");
    let i = j.find(&pat)? + pat.len();
    let rest = j[i..].trim_start();
    if let Some(r) = rest.strip_prefix('"') {
        let end = r.find('"')?;
        Some(&r[..end])
    } else {
        let end = rest.find([',', '}']).unwrap_or(rest.len());
        Some(rest[..end].trim())
    }
}

struct Stat {
    exists: bool,
    is_dir: bool,
    size: u64,
    mtime: u64,
}
fn stat(path: &str) -> Stat {
    match http("GET", "/stat", &format!("path={}", pct(path)), None) {
        Ok((200, body)) => {
            let j = String::from_utf8_lossy(&body);
            Stat {
                exists: json_str(&j, "exists") == Some("true"),
                is_dir: json_str(&j, "is_dir") == Some("true"),
                size: json_str(&j, "size")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
                mtime: json_str(&j, "mtime")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0),
            }
        }
        _ => Stat {
            exists: false,
            is_dir: false,
            size: 0,
            mtime: 0,
        },
    }
}

fn dir_attr(ino: u64) -> FileAttr {
    mk(ino, FileType::Directory, 0, UNIX_EPOCH)
}
fn file_attr(ino: u64, size: u64) -> FileAttr {
    mk(ino, FileType::RegularFile, size, UNIX_EPOCH)
}
/// File attr with a backend mtime (epoch seconds; 0 = unknown → epoch). S3-1.
fn file_attr_mt(ino: u64, size: u64, mtime_secs: u64) -> FileAttr {
    let t = if mtime_secs > 0 {
        UNIX_EPOCH + Duration::from_secs(mtime_secs)
    } else {
        UNIX_EPOCH
    };
    mk(ino, FileType::RegularFile, size, t)
}
fn mk(ino: u64, kind: FileType, size: u64, mtime: SystemTime) -> FileAttr {
    FileAttr {
        ino: INodeNo(ino),
        size,
        blocks: 1,
        atime: mtime,
        mtime,
        ctime: mtime,
        crtime: UNIX_EPOCH,
        kind,
        perm: if kind == FileType::Directory {
            0o755
        } else {
            0o644
        },
        nlink: if kind == FileType::Directory { 2 } else { 1 },
        uid: 0,
        gid: 0,
        rdev: 0,
        blksize: 65536,
        flags: 0,
    }
}

// Linux open(2) flags (the guest is always Linux/musl; this crate has no libc dep).
const O_ACCMODE: i32 = 0o3;
const O_WRONLY: i32 = 0o1;
const O_RDWR: i32 = 0o2;
const O_TRUNC: i32 = 0o1000;

/// Pending write for one open file, buffered until flush. Mirrors mirage's model:
/// record `(offset, chunk)` writes (in order) and, at flush, read the current
/// content, apply an optional truncate, splice the chunks on top, and write the
/// merged result back. `dirty` gates the write so an open-for-write that never
/// modifies anything is NOT flushed (which would clobber a rendered/read-only
/// file). `truncate == Some(0)` (a `>` redirect / create) means "start empty"
/// (skip reading the existing content).
#[derive(Default)]
struct Pending {
    truncate: Option<u64>,
    chunks: Vec<(u64, Vec<u8>)>,
    dirty: bool,
}

impl Pending {
    /// Best-effort size while still buffered (for getattr during a write).
    fn pending_size(&self) -> u64 {
        let mut sz = self.truncate.unwrap_or(0);
        for (off, c) in &self.chunks {
            sz = sz.max(off + c.len() as u64);
        }
        sz
    }
}

/// Shared forwarder state (inode<->path map + write buffers), behind an `Arc`
/// so worker threads can access it while the FUSE dispatch loop moves on.
struct Inner {
    ino_to_path: Mutex<HashMap<u64, String>>,
    path_to_ino: Mutex<HashMap<String, u64>>,
    next: Mutex<u64>,
    wbuf: Mutex<HashMap<u64, Pending>>,
    /// Last JSON result of a `/<mount>/.cmd/<op>` write, keyed by that control
    /// path, so a subsequent read returns it (C4).
    cmd_results: Mutex<HashMap<String, Vec<u8>>>,
}
impl Inner {
    fn path(&self, ino: u64) -> Option<String> {
        self.ino_to_path.lock().unwrap().get(&ino).cloned()
    }
    fn intern(&self, path: &str) -> u64 {
        let mut p2i = self.path_to_ino.lock().unwrap();
        if let Some(&i) = p2i.get(path) {
            return i;
        }
        let mut n = self.next.lock().unwrap();
        let ino = *n;
        *n += 1;
        p2i.insert(path.to_string(), ino);
        self.ino_to_path
            .lock()
            .unwrap()
            .insert(ino, path.to_string());
        ino
    }
    fn child(&self, parent: &str, name: &str) -> String {
        if parent == "/" {
            format!("/{name}")
        } else {
            format!("{parent}/{name}")
        }
    }
    /// Read-modify-write flush: read current content (unless truncated to empty),
    /// apply truncate, splice the buffered chunks, write the merged result.
    /// Returns `false` when the existing-content preload failed transiently on a
    /// file that does exist — the PUT is skipped to preserve the original, and
    /// the caller maps that to EIO (R1). All legitimate outcomes return `true`.
    fn put(&self, ino: u64) -> bool {
        let Some(p) = self.wbuf.lock().unwrap().remove(&ino) else {
            return true;
        };
        if !p.dirty {
            return true;
        }
        let Some(path) = self.path(ino) else {
            return true;
        };
        let mut merged = match p.truncate {
            Some(0) => Vec::new(),
            other => {
                let mut base = match http("GET", "/read", &format!("path={}", pct(&path)), None) {
                    Ok((200, d)) => d,
                    // Read failed. If the file exists this is a transient failure
                    // (timeout / 5xx) — skip the PUT so a partial write doesn't
                    // overwrite the original. If it doesn't exist, this is a new
                    // file and an empty base is correct.
                    _ => {
                        if stat(&path).exists {
                            return false;
                        }
                        Vec::new()
                    }
                };
                if let Some(n) = other {
                    base.resize(n as usize, 0);
                }
                base
            }
        };
        for (off, chunk) in &p.chunks {
            let end = *off as usize + chunk.len();
            if end > merged.len() {
                merged.resize(end, 0);
            }
            merged[*off as usize..end].copy_from_slice(chunk);
        }
        if let Ok((200, body)) = http(
            "PUT",
            "/write",
            &format!("path={}", pct(&path)),
            Some(&merged),
        ) {
            // C4: a `.cmd/<op>` write returns the command result — cache it so a
            // read of that path returns it (e.g. the new page id).
            if path.contains("/.cmd/") {
                self.cmd_results.lock().unwrap().insert(path, body);
            }
        }
        true
    }
}

/// Minimal fixed-size worker pool. FUSE callbacks offload their blocking HTTP
/// round trip here and return immediately, so the single-threaded FUSE dispatch
/// loop is never blocked by one slow op — a slow read/render no longer freezes
/// the whole mount, and independent ops run concurrently.
type Job = Box<dyn FnOnce() + Send + 'static>;
struct Pool {
    q: Mutex<VecDeque<Job>>,
    cv: Condvar,
}
impl Pool {
    fn new(workers: usize) -> Arc<Self> {
        let pool = Arc::new(Pool {
            q: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        });
        for _ in 0..workers {
            let p = pool.clone();
            std::thread::spawn(move || loop {
                let job = {
                    let mut q = match p.q.lock() {
                        Ok(g) => g,
                        Err(e) => e.into_inner(), // a prior panicking job poisoned the queue; keep serving
                    };
                    loop {
                        if let Some(j) = q.pop_front() {
                            break j;
                        }
                        q = match p.cv.wait(q) {
                            Ok(g) => g,
                            Err(e) => e.into_inner(),
                        };
                    }
                };
                // Isolate a panicking job so it kills only that one FUSE op, never the
                // worker — a dead worker would permanently shrink the pool and could
                // eventually wedge the whole mount.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            });
        }
        pool
    }
    fn submit(&self, job: Job) {
        self.q.lock().unwrap().push_back(job);
        self.cv.notify_one();
    }
}

struct Fs {
    inner: Arc<Inner>,
    pool: Arc<Pool>,
}
impl Fs {
    fn new() -> Self {
        let mut i2p = HashMap::new();
        let mut p2i = HashMap::new();
        i2p.insert(1, "/".to_string());
        p2i.insert("/".to_string(), 1);
        Fs {
            inner: Arc::new(Inner {
                ino_to_path: Mutex::new(i2p),
                path_to_ino: Mutex::new(p2i),
                next: Mutex::new(2),
                wbuf: Mutex::new(HashMap::new()),
                cmd_results: Mutex::new(HashMap::new()),
            }),
            pool: Pool::new(8),
        }
    }
    /// Run `job` on a worker thread (off the FUSE dispatch loop).
    fn spawn(&self, job: impl FnOnce(Arc<Inner>) + Send + 'static) {
        let inner = self.inner.clone();
        self.pool.submit(Box::new(move || job(inner)));
    }
}

impl Filesystem for Fs {
    fn lookup(&self, _r: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let parent = parent.0;
        let nm = match name.to_str() {
            Some(n) => n.to_string(),
            None => return reply.error(Errno::EINVAL),
        };
        self.spawn(move |inner| {
            let Some(pp) = inner.path(parent) else {
                return reply.error(Errno::ENOENT);
            };
            let path = inner.child(&pp, &nm);
            // C4: a `.cmd/<op>` path with a stashed result reads back as a file.
            if let Some(len) = inner
                .cmd_results
                .lock()
                .unwrap()
                .get(&path)
                .map(|b| b.len() as u64)
            {
                let ino = inner.intern(&path);
                return reply.entry(&TTL, &file_attr(ino, len), Generation(0));
            }
            let s = stat(&path);
            if !s.exists {
                return reply.error(Errno::ENOENT);
            }
            let ino = inner.intern(&path);
            reply.entry(
                &TTL,
                &if s.is_dir {
                    dir_attr(ino)
                } else {
                    file_attr_mt(ino, s.size, s.mtime)
                },
                Generation(0),
            );
        });
    }
    fn getattr(&self, _r: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        let ino = ino.0;
        self.spawn(move |inner| {
            let Some(path) = inner.path(ino) else {
                return reply.error(Errno::ENOENT);
            };
            if let Some(p) = inner.wbuf.lock().unwrap().get(&ino) {
                return reply.attr(&TTL, &file_attr(ino, p.pending_size()));
            }
            if let Some(len) = inner
                .cmd_results
                .lock()
                .unwrap()
                .get(&path)
                .map(|b| b.len() as u64)
            {
                return reply.attr(&TTL, &file_attr(ino, len));
            }
            if path == "/" {
                return reply.attr(&TTL, &dir_attr(1));
            }
            let s = stat(&path);
            if !s.exists {
                return reply.error(Errno::ENOENT);
            }
            reply.attr(
                &TTL,
                &if s.is_dir {
                    dir_attr(ino)
                } else {
                    file_attr_mt(ino, s.size, s.mtime)
                },
            );
        });
    }
    fn readdir(
        &self,
        _r: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let ino = ino.0;
        self.spawn(move |inner| {
            let Some(path) = inner.path(ino) else {
                return reply.error(Errno::ENOENT);
            };
            let (status, body) = http("GET", "/readdir", &format!("path={}", pct(&path)), None)
                .unwrap_or((0, vec![]));
            if status != 200 {
                return reply.error(Errno::EIO);
            }
            let j = String::from_utf8_lossy(&body);
            let mut names: Vec<(String, bool)> = Vec::new();
            for chunk in j.split("{\"name\":").skip(1) {
                let frag = format!("{{\"name\":{chunk}");
                if let Some(n) = json_str(&frag, "name") {
                    names.push((n.to_string(), json_str(&frag, "is_dir") == Some("true")));
                }
            }
            let mut entries: Vec<(u64, FileType, String)> = vec![
                (ino, FileType::Directory, ".".into()),
                (1, FileType::Directory, "..".into()),
            ];
            for (n, is_dir) in names {
                let cp = inner.child(&path, &n);
                let cino = inner.intern(&cp);
                entries.push((
                    cino,
                    if is_dir {
                        FileType::Directory
                    } else {
                        FileType::RegularFile
                    },
                    n,
                ));
            }
            for (i, (e_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
                if reply.add(INodeNo(*e_ino), (i + 1) as u64, *kind, name) {
                    break;
                }
            }
            reply.ok();
        });
    }
    #[allow(clippy::too_many_arguments)]
    fn setattr(
        &self,
        _r: &Request,
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
        self.spawn(move |inner| {
            // Honor truncate (e.g. `echo > file` opens O_TRUNC). Recorded as a
            // base-resize applied before the buffered writes at flush time.
            if let Some(sz) = size {
                let mut wb = inner.wbuf.lock().unwrap();
                let p = wb.entry(ino).or_default();
                p.truncate = Some(sz);
                p.dirty = true;
                return reply.attr(&TTL, &file_attr(ino, sz));
            }
            let cur = inner
                .wbuf
                .lock()
                .unwrap()
                .get(&ino)
                .map(|p| p.pending_size());
            match cur {
                Some(n) => reply.attr(&TTL, &file_attr(ino, n)),
                None => match inner.path(ino) {
                    Some(p) if p != "/" => {
                        let s = stat(&p);
                        reply.attr(
                            &TTL,
                            &if s.is_dir {
                                dir_attr(ino)
                            } else {
                                file_attr_mt(ino, s.size, s.mtime)
                            },
                        );
                    }
                    _ => reply.attr(&TTL, &dir_attr(ino)),
                },
            }
        });
    }
    fn open(&self, _r: &Request, ino: INodeNo, flags: OpenFlags, reply: ReplyOpen) {
        // `>` (O_WRONLY|O_TRUNC): start the pending buffer empty so the old
        // content isn't read back and re-merged. Non-truncating writes preload
        // lazily at flush (read-modify-write in `put`).
        let acc = flags.0 & O_ACCMODE;
        let write = acc == O_WRONLY || acc == O_RDWR;
        if write && flags.0 & O_TRUNC != 0 {
            let mut wb = self.inner.wbuf.lock().unwrap();
            let p = wb.entry(ino.0).or_default();
            p.truncate = Some(0);
            p.dirty = true;
        }
        // direct_io: don't clamp reads to stat size (dynamic/rendered files).
        reply.opened(FileHandle(0), FopenFlags::FOPEN_DIRECT_IO);
    }
    #[allow(clippy::too_many_arguments)]
    fn read(
        &self,
        _r: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let ino = ino.0;
        self.spawn(move |inner| {
            let Some(path) = inner.path(ino) else {
                return reply.error(Errno::ENOENT);
            };
            // C4: serve a stashed `.cmd/<op>` result instead of hitting the provider.
            if let Some(buf) = inner.cmd_results.lock().unwrap().get(&path) {
                let start = (offset as usize).min(buf.len());
                let end = (start + size as usize).min(buf.len());
                return reply.data(&buf[start..end]);
            }
            match http(
                "GET",
                "/read",
                &format!("path={}&offset={offset}&size={size}", pct(&path)),
                None,
            ) {
                Ok((200, data)) => reply.data(&data),
                _ => reply.error(Errno::EIO),
            }
        });
    }
    fn create(
        &self,
        _r: &Request,
        parent: INodeNo,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let parent = parent.0;
        let Some(pp) = self.inner.path(parent) else {
            return reply.error(Errno::ENOENT);
        };
        let Some(nm) = name.to_str() else {
            return reply.error(Errno::EINVAL);
        };
        let path = self.inner.child(&pp, nm);
        let ino = self.inner.intern(&path);
        // New file: start empty + dirty so an immediate close still creates it.
        let mut wb = self.inner.wbuf.lock().unwrap();
        let p = wb.entry(ino).or_default();
        p.truncate = Some(0);
        p.dirty = true;
        drop(wb);
        reply.created(
            &TTL,
            &file_attr(ino, 0),
            Generation(0),
            FileHandle(0),
            FopenFlags::FOPEN_DIRECT_IO,
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn write(
        &self,
        _r: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        data: &[u8],
        _wf: WriteFlags,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        reply: ReplyWrite,
    ) {
        // Buffer the chunk in write order; the read-modify-write merge happens in
        // `put` (flush/release), so the existing content is never clobbered.
        let mut wb = self.inner.wbuf.lock().unwrap();
        let p = wb.entry(ino.0).or_default();
        p.chunks.push((offset, data.to_vec()));
        p.dirty = true;
        reply.written(data.len() as u32);
    }
    fn unlink(&self, _r: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let parent = parent.0;
        let Some(nm) = name.to_str().map(str::to_string) else {
            return reply.error(Errno::EINVAL);
        };
        self.spawn(move |inner| {
            let Some(pp) = inner.path(parent) else {
                return reply.error(Errno::ENOENT);
            };
            let path = inner.child(&pp, &nm);
            match http("DELETE", "/unlink", &format!("path={}", pct(&path)), None) {
                Ok((200, _)) => reply.ok(),
                _ => reply.error(Errno::EIO),
            }
        });
    }
    fn mkdir(
        &self,
        _r: &Request,
        parent: INodeNo,
        name: &OsStr,
        _mode: u32,
        _umask: u32,
        reply: ReplyEntry,
    ) {
        let parent = parent.0;
        let Some(nm) = name.to_str().map(str::to_string) else {
            return reply.error(Errno::EINVAL);
        };
        self.spawn(move |inner| {
            let Some(pp) = inner.path(parent) else {
                return reply.error(Errno::ENOENT);
            };
            let path = inner.child(&pp, &nm);
            match http("POST", "/mkdir", &format!("path={}", pct(&path)), None) {
                Ok((200, _)) => {
                    let ino = inner.intern(&path);
                    reply.entry(&TTL, &dir_attr(ino), Generation(0));
                }
                _ => reply.error(Errno::EIO),
            }
        });
    }
    fn rmdir(&self, _r: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEmpty) {
        let parent = parent.0;
        let Some(nm) = name.to_str().map(str::to_string) else {
            return reply.error(Errno::EINVAL);
        };
        self.spawn(move |inner| {
            let Some(pp) = inner.path(parent) else {
                return reply.error(Errno::ENOENT);
            };
            let path = inner.child(&pp, &nm);
            match http("DELETE", "/rmdir", &format!("path={}", pct(&path)), None) {
                Ok((200, _)) => reply.ok(),
                _ => reply.error(Errno::EIO),
            }
        });
    }
    #[allow(clippy::too_many_arguments)]
    fn rename(
        &self,
        _r: &Request,
        parent: INodeNo,
        name: &OsStr,
        newparent: INodeNo,
        newname: &OsStr,
        _flags: RenameFlags,
        reply: ReplyEmpty,
    ) {
        let parent = parent.0;
        let newparent = newparent.0;
        let (Some(nm), Some(nnm)) = (
            name.to_str().map(str::to_string),
            newname.to_str().map(str::to_string),
        ) else {
            return reply.error(Errno::EINVAL);
        };
        self.spawn(move |inner| {
            let (Some(pp), Some(np)) = (inner.path(parent), inner.path(newparent)) else {
                return reply.error(Errno::ENOENT);
            };
            let from = inner.child(&pp, &nm);
            let to = inner.child(&np, &nnm);
            let q = format!("path={}&to={}", pct(&from), pct(&to));
            match http("POST", "/rename", &q, None) {
                Ok((200, _)) => reply.ok(),
                _ => reply.error(Errno::EIO),
            }
        });
    }
    fn flush(
        &self,
        _r: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        _lock: LockOwner,
        reply: ReplyEmpty,
    ) {
        let ino = ino.0;
        self.spawn(move |inner| {
            if inner.put(ino) {
                reply.ok();
            } else {
                reply.error(Errno::EIO);
            }
        });
    }
    fn release(
        &self,
        _r: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        _flags: OpenFlags,
        _lock: Option<LockOwner>,
        _flush: bool,
        reply: ReplyEmpty,
    ) {
        let ino = ino.0;
        self.spawn(move |inner| {
            if inner.put(ino) {
                reply.ok();
            } else {
                reply.error(Errno::EIO);
            }
        });
    }
}

fn main() {
    let mp = std::env::args().nth(1).expect("mountpoint arg");
    // `Config` is #[non_exhaustive]; build via Default + set the public field.
    let mut cfg = Config::default();
    cfg.mount_options = vec![MountOption::FSName("ailoyvfs".into()), MountOption::RW];
    fuser::mount2(Fs::new(), &mp, &cfg).unwrap();
}
