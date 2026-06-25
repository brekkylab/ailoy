// Static, dependency-free in-guest VFS forwarder. Mounts a FUSE filesystem and
// forwards operations to the host forward server over plain HTTP (no TLS, no
// libfuse, no python). Cross-compiled to <arch>-linux-musl; needs only
// /dev/fuse (built into the guest kernel) and runs as root.
use std::collections::HashMap;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::{Duration, UNIX_EPOCH};

use fuser::{
    FileAttr, FileType, Filesystem, MountOption, ReplyAttr, ReplyCreate, ReplyData,
    ReplyDirectory, ReplyEmpty, ReplyEntry, ReplyOpen, ReplyWrite, Request, TimeOrNow,
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

/// Minimal HTTP/1.1 client over raw TCP. Returns (status, body).
fn http(method: &str, route: &str, query: &str, body: Option<&[u8]>) -> std::io::Result<(u16, Vec<u8>)> {
    let mut s = TcpStream::connect(host_port())?;
    s.set_read_timeout(Some(Duration::from_secs(120)))?;
    s.set_write_timeout(Some(Duration::from_secs(120)))?;
    let mut head = format!(
        "{method} {route}?{query} HTTP/1.1\r\nHost: vfs\r\nx-vfs-token: {}\r\nConnection: close\r\n",
        token()
    );
    if let Some(b) = body {
        head.push_str(&format!("Content-Type: application/octet-stream\r\nContent-Length: {}\r\n", b.len()));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    if let Some(b) = body {
        s.write_all(b)?;
    }
    s.flush()?;
    let mut resp = Vec::new();
    s.read_to_end(&mut resp)?;
    let sep = resp.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or(resp.len());
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

struct Stat { exists: bool, is_dir: bool, size: u64 }
fn stat(path: &str) -> Stat {
    match http("GET", "/stat", &format!("path={}", pct(path)), None) {
        Ok((200, body)) => {
            let j = String::from_utf8_lossy(&body);
            Stat {
                exists: json_str(&j, "exists") == Some("true"),
                is_dir: json_str(&j, "is_dir") == Some("true"),
                size: json_str(&j, "size").and_then(|s| s.parse().ok()).unwrap_or(0),
            }
        }
        _ => Stat { exists: false, is_dir: false, size: 0 },
    }
}

fn dir_attr(ino: u64) -> FileAttr { mk(ino, FileType::Directory, 0) }
fn file_attr(ino: u64, size: u64) -> FileAttr { mk(ino, FileType::RegularFile, size) }
fn mk(ino: u64, kind: FileType, size: u64) -> FileAttr {
    FileAttr {
        ino, size, blocks: 1, atime: UNIX_EPOCH, mtime: UNIX_EPOCH, ctime: UNIX_EPOCH,
        crtime: UNIX_EPOCH, kind, perm: if kind == FileType::Directory { 0o755 } else { 0o644 },
        nlink: if kind == FileType::Directory { 2 } else { 1 },
        uid: 0, gid: 0, rdev: 0, blksize: 65536, flags: 0,
    }
}

struct Fs {
    ino_to_path: Mutex<HashMap<u64, String>>,
    path_to_ino: Mutex<HashMap<String, u64>>,
    next: Mutex<u64>,
    wbuf: Mutex<HashMap<u64, Vec<u8>>>,
}
impl Fs {
    fn new() -> Self {
        let mut i2p = HashMap::new();
        let mut p2i = HashMap::new();
        i2p.insert(1, "/".to_string());
        p2i.insert("/".to_string(), 1);
        Fs { ino_to_path: Mutex::new(i2p), path_to_ino: Mutex::new(p2i), next: Mutex::new(2), wbuf: Mutex::new(HashMap::new()) }
    }
    fn path(&self, ino: u64) -> Option<String> { self.ino_to_path.lock().unwrap().get(&ino).cloned() }
    fn intern(&self, path: &str) -> u64 {
        let mut p2i = self.path_to_ino.lock().unwrap();
        if let Some(&i) = p2i.get(path) { return i; }
        let mut n = self.next.lock().unwrap();
        let ino = *n; *n += 1;
        p2i.insert(path.to_string(), ino);
        self.ino_to_path.lock().unwrap().insert(ino, path.to_string());
        ino
    }
    fn child(&self, parent: &str, name: &str) -> String {
        if parent == "/" { format!("/{name}") } else { format!("{parent}/{name}") }
    }
}

impl Filesystem for Fs {
    fn lookup(&mut self, _r: &Request, parent: u64, name: &OsStr, reply: ReplyEntry) {
        let Some(pp) = self.path(parent) else { reply.error(libc::ENOENT); return; };
        let Some(nm) = name.to_str() else { reply.error(libc::EINVAL); return; };
        let path = self.child(&pp, nm);
        let s = stat(&path);
        if !s.exists { reply.error(libc::ENOENT); return; }
        let ino = self.intern(&path);
        let a = if s.is_dir { dir_attr(ino) } else { file_attr(ino, s.size) };
        reply.entry(&TTL, &a, 0);
    }
    fn getattr(&mut self, _r: &Request, ino: u64, _fh: Option<u64>, reply: ReplyAttr) {
        let Some(path) = self.path(ino) else { reply.error(libc::ENOENT); return; };
        if let Some(buf) = self.wbuf.lock().unwrap().get(&ino) {
            reply.attr(&TTL, &file_attr(ino, buf.len() as u64)); return;
        }
        if path == "/" { reply.attr(&TTL, &dir_attr(1)); return; }
        let s = stat(&path);
        if !s.exists { reply.error(libc::ENOENT); return; }
        reply.attr(&TTL, &if s.is_dir { dir_attr(ino) } else { file_attr(ino, s.size) });
    }
    fn readdir(&mut self, _r: &Request, ino: u64, _fh: u64, offset: i64, mut reply: ReplyDirectory) {
        let Some(path) = self.path(ino) else { reply.error(libc::ENOENT); return; };
        let (status, body) = http("GET", "/readdir", &format!("path={}", pct(&path)), None).unwrap_or((0, vec![]));
        if status != 200 { reply.error(libc::EIO); return; }
        let j = String::from_utf8_lossy(&body);
        // entries: [{"name":..,"is_dir":..,"size":..}, ...]
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
            let cp = self.child(&path, &n);
            let cino = self.intern(&cp);
            entries.push((cino, if is_dir { FileType::Directory } else { FileType::RegularFile }, n));
        }
        for (i, (e_ino, kind, name)) in entries.iter().enumerate().skip(offset as usize) {
            if reply.add(*e_ino, (i + 1) as i64, *kind, name) { break; }
        }
        reply.ok();
    }
    #[allow(clippy::too_many_arguments)]
    fn setattr(
        &mut self, _r: &Request, ino: u64, _mode: Option<u32>, _uid: Option<u32>,
        _gid: Option<u32>, size: Option<u64>, _atime: Option<TimeOrNow>,
        _mtime: Option<TimeOrNow>, _ctime: Option<std::time::SystemTime>, _fh: Option<u64>,
        _crtime: Option<std::time::SystemTime>, _chgtime: Option<std::time::SystemTime>,
        _bkuptime: Option<std::time::SystemTime>, _flags: Option<u32>, reply: ReplyAttr,
    ) {
        // Honor truncate (e.g. `echo > file` opens O_TRUNC). Adjust the write
        // buffer so a subsequent write produces the intended content/size.
        if let Some(sz) = size {
            let mut wb = self.wbuf.lock().unwrap();
            let buf = wb.entry(ino).or_default();
            buf.resize(sz as usize, 0);
            reply.attr(&TTL, &file_attr(ino, sz));
            return;
        }
        // No size change requested: report current attrs.
        let cur = self.wbuf.lock().unwrap().get(&ino).map(|b| b.len() as u64);
        match cur {
            Some(n) => reply.attr(&TTL, &file_attr(ino, n)),
            None => match self.path(ino) {
                Some(p) if p != "/" => {
                    let s = stat(&p);
                    reply.attr(&TTL, &if s.is_dir { dir_attr(ino) } else { file_attr(ino, s.size) });
                }
                _ => reply.attr(&TTL, &dir_attr(ino)),
            },
        }
    }
    fn open(&mut self, _r: &Request, _ino: u64, _flags: i32, reply: ReplyOpen) {
        // direct_io: don't clamp reads to stat size (dynamic/rendered files).
        reply.opened(0, fuser::consts::FOPEN_DIRECT_IO);
    }
    fn read(&mut self, _r: &Request, ino: u64, _fh: u64, offset: i64, size: u32, _f: i32, _l: Option<u64>, reply: ReplyData) {
        let Some(path) = self.path(ino) else { reply.error(libc::ENOENT); return; };
        match http("GET", "/read", &format!("path={}&offset={offset}&size={size}", pct(&path)), None) {
            Ok((200, data)) => reply.data(&data),
            _ => reply.error(libc::EIO),
        }
    }
    fn create(&mut self, _r: &Request, parent: u64, name: &OsStr, _mode: u32, _umask: u32, _flags: i32, reply: ReplyCreate) {
        let Some(pp) = self.path(parent) else { reply.error(libc::ENOENT); return; };
        let Some(nm) = name.to_str() else { reply.error(libc::EINVAL); return; };
        let path = self.child(&pp, nm);
        let ino = self.intern(&path);
        self.wbuf.lock().unwrap().insert(ino, Vec::new());
        reply.created(&TTL, &file_attr(ino, 0), 0, 0, fuser::consts::FOPEN_DIRECT_IO);
    }
    fn write(&mut self, _r: &Request, ino: u64, _fh: u64, offset: i64, data: &[u8], _w: u32, _f: i32, _l: Option<u64>, reply: ReplyWrite) {
        let mut wb = self.wbuf.lock().unwrap();
        let buf = wb.entry(ino).or_default();
        let off = offset as usize;
        if off > buf.len() { buf.resize(off, 0); }
        if off + data.len() > buf.len() { buf.resize(off + data.len(), 0); }
        buf[off..off + data.len()].copy_from_slice(data);
        reply.written(data.len() as u32);
    }
    fn unlink(&mut self, _r: &Request, parent: u64, name: &OsStr, reply: ReplyEmpty) {
        let Some(pp) = self.path(parent) else { reply.error(libc::ENOENT); return; };
        let Some(nm) = name.to_str() else { reply.error(libc::EINVAL); return; };
        let path = self.child(&pp, nm);
        match http("DELETE", "/unlink", &format!("path={}", pct(&path)), None) {
            Ok((200, _)) => reply.ok(),
            _ => reply.error(libc::EIO),
        }
    }
    fn flush(&mut self, _r: &Request, ino: u64, _fh: u64, _lock: u64, reply: ReplyEmpty) {
        self.put(ino); reply.ok();
    }
    fn release(&mut self, _r: &Request, ino: u64, _fh: u64, _f: i32, _l: Option<u64>, _fl: bool, reply: ReplyEmpty) {
        self.put(ino); reply.ok();
    }
}
impl Fs {
    fn put(&self, ino: u64) {
        let body = { self.wbuf.lock().unwrap().get(&ino).cloned() };
        let Some(body) = body else { return; };
        let Some(path) = self.path(ino) else { return; };
        let _ = http("PUT", "/write", &format!("path={}", pct(&path)), Some(&body));
        self.wbuf.lock().unwrap().remove(&ino);
    }
}

fn main() {
    let mp = std::env::args().nth(1).expect("mountpoint arg");
    fuser::mount2(Fs::new(), &mp, &[MountOption::FSName("ailoyvfs".into()), MountOption::RW]).unwrap();
}
