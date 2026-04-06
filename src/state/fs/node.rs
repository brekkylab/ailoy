/// Distinguishes whether a [`Node`] is a regular file or a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    File,
    Directory,
}

/// Metadata returned by [`File::stat`] and [`Directory::stat`],
/// analogous to POSIX `struct stat` (subset).
#[derive(Debug, Clone)]
pub struct Stat {
    /// For files, the number of bytes of content.
    /// For directories, the number of direct children.
    pub size: u64,

    pub kind: NodeKind,
}

/// POSIX-like interface for a single file.
///
/// # Lifetime and handle borrowing
///
/// `Handle<'a>` is a Generic Associated Type (GAT) whose lifetime `'a` is
/// bound to `&'a mut self`.  This encodes at the type level that an open
/// handle may mutably borrow data from the file it was opened on: the compiler
/// will reject any program where a handle outlives its source file, and
/// prevents opening multiple handles simultaneously.
pub trait File {
    /// The handle type returned by [`open`](File::open).
    ///
    /// The `where Self: 'a` bound is required by Rust's GAT rules: it ensures
    /// that `Handle<'a>` is only constructed while the file itself is live for
    /// at least `'a`.
    type Handle<'a>: FileHandle
    where
        Self: 'a;

    /// Open the file and return a handle positioned at byte 0.
    ///
    /// The returned handle mutably borrows `self` for its lifetime; drop it to
    /// release the borrow and close the file.
    fn open<'a>(&'a mut self) -> Self::Handle<'a>;

    /// Return current metadata without opening a handle.
    fn stat(&self) -> Stat;
}

/// Object-safe bridge from [`File`] to dynamic dispatch.
///
/// [`File`] itself is not object-safe because its GAT `Handle<'a>` prevents
/// it from being used as `dyn File`.  `FileDyn` erases the concrete handle
/// type by boxing it, while still propagating the borrow lifetime through
/// `Box<dyn FileHandle + '_>`.  Any type that implements [`File`] gets a
/// blanket implementation of `FileDyn` for free.
///
/// This trait is what [`Node::File`] stores.
pub trait FileDyn {
    /// Open the file and return a type-erased handle.
    ///
    /// The `'_` lifetime on the return type is tied to `&mut self`, so the
    /// boxed handle cannot outlive the `FileDyn` object it was opened from.
    fn open_dyn(&mut self) -> Box<dyn FileHandle + '_>;

    fn stat_dyn(&self) -> Stat;
}

impl<T: File> FileDyn for T {
    fn open_dyn(&mut self) -> Box<dyn FileHandle + '_> {
        Box::new(self.open())
    }

    fn stat_dyn(&self) -> Stat {
        self.stat()
    }
}

/// Operations on an open file handle, analogous to POSIX file-descriptor calls.
///
/// All cursor movement is relative to the handle's internal position; multiple
/// handles opened from the same file have independent cursors.
pub trait FileHandle {
    /// Read up to `count` bytes from the current cursor position and advance
    /// the cursor by the number of bytes actually read.
    ///
    /// The returned slice borrows from the handle itself (`&self`), so it must
    /// be dropped before the next mutable operation on this handle.
    fn read(&self, count: u64) -> &[u8];

    /// Write `data` at the current cursor position, extending the file if
    /// necessary, then advance the cursor by `data.len()`.
    fn write(&mut self, data: &[u8]);

    /// Move the cursor by `offset` bytes relative to the current position,
    /// clamped to `[0, file_size]`.
    ///
    /// Returns the new absolute cursor position.
    fn seek(&self, offset: i64) -> u64;

    /// Return the current cursor position in bytes from the start of the file.
    fn tell(&self) -> u64;
}

/// A single entry inside a directory listing, returned by [`Directory::readdir`].
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: NodeKind,
}

/// POSIX-like interface for a directory node.
pub trait Directory {
    /// Return metadata for every direct child, analogous to `readdir(3)`.
    fn readdir(&self) -> Vec<DirEntry>;

    fn stat(&self) -> Stat;

    /// Look up a direct child by name.  Returns `None` if no child with that
    /// name exists.
    fn get_child(&self, name: &str) -> Option<&Node>;

    fn get_child_mut(&mut self, name: &str) -> Option<&mut Node>;

    /// Insert `node` as a direct child named `name`, replacing any existing
    /// child with the same name.
    fn insert_child(&mut self, name: String, node: Node);

    /// Remove the direct child named `name`.  No-op if the child does not
    /// exist.
    fn remove_child(&mut self, name: &str);
}

/// A single node in the virtual filesystem tree — either a file or a
/// directory.
///
/// Files are stored as `Box<dyn FileDyn>` rather than `Box<dyn File>` because
/// [`File`] is not object-safe (it has a GAT).  [`FileDyn`] provides the
/// same open/stat interface with the handle type erased to
/// `Box<dyn FileHandle + '_>`.
pub enum Node {
    File(Box<dyn FileDyn>),
    Directory(Box<dyn Directory>),
}

impl Node {
    /// Return the [`NodeKind`] of this node without opening it.
    pub fn kind(&self) -> NodeKind {
        match self {
            Node::File(_) => NodeKind::File,
            Node::Directory(_) => NodeKind::Directory,
        }
    }
}
