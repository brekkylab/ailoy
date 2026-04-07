use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write as IoWrite};
use std::path::PathBuf;

use crate::state::fs::{DirEntry, Directory, File, FileHandle, Node, NodeKind, Stat};

// ---------------------------------------------------------------------------
// File
// ---------------------------------------------------------------------------

/// A [`File`] node backed by a real file on disk.
pub struct FileSystemFile {
    path: PathBuf,
}

impl FileSystemFile {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

/// An open handle to a [`FileSystemFile`].
///
/// All I/O goes directly through the OS file descriptor.  We keep `pos` only
/// to implement [`FileHandle::tell`], which takes `&self` and therefore cannot
/// call `Seek::seek(Current(0))` on the inner file.  `pos` is always kept in
/// sync with the real descriptor position after every read, write, or seek.
///
/// A small `buf` is retained so that [`FileHandle::read`] can return `&[u8]`
/// borrowing from `self`.
pub struct FileSystemFileHandle {
    file: std::fs::File,
    buf: Vec<u8>,
    /// Mirrors the real fd position so that `tell` can be `&self`.
    pos: u64,
    readonly: bool,
}

impl FileSystemFile {
    fn is_readonly(&self) -> bool {
        std::fs::metadata(&self.path)
            .map(|m| m.permissions().readonly())
            .unwrap_or(false)
    }
}

impl File for FileSystemFile {
    type Handle<'a>
        = FileSystemFileHandle
    where
        Self: 'a;

    fn open<'a>(&'a mut self) -> FileSystemFileHandle {
        let readonly = self.is_readonly();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(!readonly)
            .create(!readonly)
            .open(&self.path)
            .unwrap_or_else(|e| panic!("FileSystemFile: cannot open {:?}: {e}", self.path));
        FileSystemFileHandle {
            file,
            buf: Vec::new(),
            pos: 0,
            readonly,
        }
    }

    fn stat(&self) -> Stat {
        let meta = std::fs::metadata(&self.path);
        let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
        let readonly = meta.map(|m| m.permissions().readonly()).unwrap_or(false);
        Stat {
            size,
            kind: NodeKind::File,
            readonly,
        }
    }
}

impl FileHandle for FileSystemFileHandle {
    fn read(&mut self, count: u64) -> &[u8] {
        self.buf.resize(count as usize, 0);
        let n = self.file.read(&mut self.buf).unwrap_or(0);
        self.buf.truncate(n);
        self.pos += n as u64;
        &self.buf
    }

    fn write(&mut self, data: &[u8]) {
        assert!(!self.readonly, "write to read-only file");
        self.file
            .write_all(data)
            .expect("FileSystemFileHandle: write failed");
        self.pos += data.len() as u64;
    }

    fn seek(&mut self, offset: i64) -> u64 {
        // Clamp within [0, file_size] as the trait requires.
        let size = self
            .file
            .seek(SeekFrom::End(0))
            .expect("FileSystemFileHandle: seek(End) failed");
        let new_pos = (self.pos as i64 + offset).clamp(0, size as i64) as u64;
        self.file
            .seek(SeekFrom::Start(new_pos))
            .expect("FileSystemFileHandle: seek(Start) failed");
        self.pos = new_pos;
        new_pos
    }

    fn tell(&self) -> u64 {
        self.pos
    }
}

// ---------------------------------------------------------------------------
// Directory
// ---------------------------------------------------------------------------

/// A [`Directory`] node that transparently mirrors a real directory on disk.
///
/// - `readdir` and `stat` always query the real filesystem — no snapshot is
///   kept, so the listing stays up to date as the directory changes.
/// - `get_child` / `get_child_mut` create child [`Node`]s on demand and store
///   them only to satisfy the borrow-returning signature of the [`Directory`]
///   trait.  The stored nodes are lightweight path wrappers; they do **not**
///   buffer file contents or directory listings.
///
/// # Linking a real path into the virtual tree
///
/// ```rust,ignore
/// use std::path::PathBuf;
/// use ailoy::state::fs::{Node, FileSystemDir};
///
/// root.insert_child(
///     "host_data".to_string(),
///     Node::Directory(Box::new(FileSystemDir::new(PathBuf::from("/tmp/data")))),
/// );
/// ```
pub struct FileSystemDir {
    path: PathBuf,
    /// Nodes created on demand so that `get_child` can return `&Node`.
    /// Only entries that have been looked up are present here; the source of
    /// truth for what actually exists on disk is always the real filesystem.
    nodes: RefCell<HashMap<String, Node>>,
}

impl FileSystemDir {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            nodes: RefCell::new(HashMap::new()),
        }
    }

    fn is_readonly(&self) -> bool {
        std::fs::metadata(&self.path)
            .map(|m| m.permissions().readonly())
            .unwrap_or(false)
    }

    /// Build a [`Node`] for the child named `name` if it exists on disk.
    fn make_node(&self, name: &str) -> Option<Node> {
        let child_path = self.path.join(name);
        if child_path.is_dir() {
            Some(Node::Directory(Box::new(FileSystemDir::new(child_path))))
        } else if child_path.exists() {
            Some(Node::File(Box::new(FileSystemFile::new(child_path))))
        } else {
            None
        }
    }
}

impl Directory for FileSystemDir {
    /// Always reads the real directory; never uses the node map.
    fn readdir(&self) -> Vec<DirEntry> {
        std::fs::read_dir(&self.path)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| {
                let kind = if e.path().is_dir() {
                    NodeKind::Directory
                } else {
                    NodeKind::File
                };
                DirEntry {
                    name: e.file_name().to_string_lossy().into_owned(),
                    kind,
                }
            })
            .collect()
    }

    /// Always queries `std::fs::metadata`; never uses the node map.
    fn stat(&self) -> Stat {
        let child_count = std::fs::read_dir(&self.path)
            .map(|rd| rd.flatten().count())
            .unwrap_or(0);
        let readonly = self.is_readonly();
        Stat {
            size: child_count as u64,
            kind: NodeKind::Directory,
            readonly,
        }
    }

    fn get_child(&self, name: &str) -> Option<&Node> {
        // Insert on first access so we have something to return a reference to.
        {
            let mut nodes = self.nodes.borrow_mut();
            if !nodes.contains_key(name) {
                if let Some(node) = self.make_node(name) {
                    nodes.insert(name.to_string(), node);
                }
            }
        }
        // SAFETY: we hold `&self` for the duration of the caller's borrow, so
        // the HashMap cannot be mutated while the returned reference is live.
        // We use a raw pointer to escape the `Ref` guard's lifetime, which
        // would otherwise be too short (stack-local).
        let nodes = self.nodes.borrow();
        let map: &HashMap<String, Node> =
            unsafe { &*(&*nodes as *const HashMap<String, Node>) };
        map.get(name)
    }

    fn get_child_mut(&mut self, name: &str) -> Option<&mut Node> {
        // Build the node before borrowing `self.nodes` so that `self.path`
        // is still accessible.
        if !self.nodes.get_mut().contains_key(name) {
            let child_path = self.path.join(name);
            let maybe_node = if child_path.is_dir() {
                Some(Node::Directory(Box::new(FileSystemDir::new(child_path))))
            } else if child_path.exists() {
                Some(Node::File(Box::new(FileSystemFile::new(child_path))))
            } else {
                None
            };
            if let Some(node) = maybe_node {
                self.nodes.get_mut().insert(name.to_string(), node);
            }
        }
        self.nodes.get_mut().get_mut(name)
    }

    fn insert_child(&mut self, name: String, node: Node) {
        assert!(!self.is_readonly(), "insert into read-only directory: {name}");
        self.nodes.get_mut().insert(name, node);
    }

    fn remove_child(&mut self, name: &str) {
        assert!(!self.is_readonly(), "remove from read-only directory: {name}");
        self.nodes.get_mut().remove(name);
    }
}
