//! A namespace that stitches several stores into one filesystem.
//!
//! A [`ContextFs`] is a longest-prefix mount table: each store is registered at a root-relative
//! path and serves every request that falls under it, with the path re-based onto the store's
//! own root. It is itself a [`FileSystem`], so a binding drives it like any single store.
//!
//! Carried here from cortex, where it was public, because virtx's own tree does not do this
//! job: [`virtx::fs::Directory`] grafts host directories only, never at the root and never one
//! inside another, while the workspace is the user's directory at the root with Notion and S3
//! stores, and the artifacts directory, mounted inside it.
//!
//! **The `mount` here is the table's, not the operating system's.**
//! [`mount`](ContextFs::mount) grafts a store onto a path in *this* tree and nothing outside the
//! process learns of it. Putting the result where a kernel can see it is a binding's
//! ([`FuseTMount`](virtx::fs::FuseTMount)), so a `ContextFs` with ten mounts in it may still be
//! mounted nowhere at all.

use std::{
    collections::BTreeMap,
    io,
    ops::Bound::{Excluded, Unbounded},
    path::{Component, Path, PathBuf},
    time::SystemTime,
};

use virtx::{
    BoxFuture,
    fs::{Dirent, DirentKind, FileSystem, Stat},
};

/// Several stores under one root: the workspace a session is given to work from, the
/// user's own files with every connected source grafted in.
pub struct ContextFs {
    /// Mount points keyed by their normalized, root-relative path.
    ///
    /// `PathBuf`'s component-wise `Ord` guarantees that, among all keys that are a prefix of a
    /// request, the longest is also the lexicographically greatest — so a longest-prefix lookup
    /// is a reverse range scan (see [`ContextFs::route`]), and all keys sharing a prefix form one
    /// contiguous run (see [`ContextFs::descendant_mounts`]). A store mounted at the empty path is
    /// the root and serves anything no deeper mount claims.
    mounts: BTreeMap<PathBuf, Box<dyn FileSystem>>,

    /// The timestamp every synthesized directory reports.
    ///
    /// Fixed at construction rather than read per call. A directory that advertises
    /// `SystemTime::now()` on each `stat` looks perpetually modified, and a guest that
    /// negotiated `AUTO_INVAL_DATA` watches exactly that field to decide when to drop cached
    /// pages — so a moving mtime would invalidate forever. Leaving it unset is no better: it
    /// falls back to the UNIX epoch, which `find -newer`, `make` and `rsync` all read.
    born: SystemTime,
}

impl ContextFs {
    /// An empty tree with no mounts.
    ///
    /// The root is still a directory — an empty one, like a freshly mounted tmpfs — so an empty
    /// tree can be mounted and filled in rather than failing at the kernel's first
    /// `getattr`. Every other path is [`NotFound`](io::ErrorKind::NotFound) until something
    /// claims it.
    pub fn new() -> Self {
        ContextFs {
            mounts: BTreeMap::new(),
            born: SystemTime::now(),
        }
    }

    /// Builder-style mount that overwrites any store already at `path`.
    ///
    /// Fails only if `path` escapes the tree's root.
    pub fn try_with_mount<M: FileSystem + 'static>(
        mut self,
        path: impl AsRef<Path>,
        store: M,
    ) -> io::Result<Self> {
        self.mounts
            .insert(mount_key(path.as_ref())?, Box::new(store));
        Ok(self)
    }

    /// Mount `store` at `path` (root-relative). Fails if the path escapes the tree's root or
    /// another store is already mounted there.
    pub fn mount<M: FileSystem + 'static>(
        &mut self,
        path: impl AsRef<Path>,
        store: M,
    ) -> io::Result<()> {
        let key = mount_key(path.as_ref())?;
        if self.mounts.contains_key(&key) {
            return Err(io::ErrorKind::AlreadyExists.into());
        }
        self.mounts.insert(key, Box::new(store));
        Ok(())
    }

    /// Remove the mount registered at `path`, returning the detached store.
    pub fn unmount(&mut self, path: impl AsRef<Path>) -> io::Result<Box<dyn FileSystem>> {
        let key = mount_key(path.as_ref())?;
        self.mounts.remove(&key).ok_or_else(not_found)
    }

    /// The store that owns `key` (longest-prefix match), with `key` re-based onto that store's
    /// mount point. `None` when no mount claims it.
    ///
    /// `key` must already be normalized — the caller does that once and reuses it for the
    /// mount-table queries alongside this one. That is a correctness requirement, not a
    /// convenience: a binding walks the tree as `/`-rooted paths, so a stray
    /// `RootDir` component would miss every mount.
    ///
    /// `Option` rather than `Result` on purpose. Every caller has to tell "no mount claims
    /// this" apart from "a mount claimed it and answered `NotFound`", because only the first is
    /// a candidate for a directory synthesized from the mount table. A `Result` here invites
    /// `?`, which would propagate `NotFound` before that decision is made.
    fn route(&self, key: &Path) -> Option<(&dyn FileSystem, PathBuf)> {
        // Walk every key <= `key` from the greatest downward; the first that is a prefix of
        // `key` is the longest match.
        let (mount, store) = self
            .mounts
            .range(..=key.to_path_buf())
            .rev()
            .find(|(k, _)| key.starts_with(k))?;
        let sub = key
            .strip_prefix(mount)
            .expect("matched mount is a prefix of the request");
        Some((store.as_ref(), sub.to_path_buf()))
    }

    /// [`route`](Self::route) for the data plane, where both refusals are settled the same way
    /// every time: a directory the mount table synthesized is a directory, and a path no mount
    /// claims is missing.
    fn route_file(&self, key: &Path) -> io::Result<(&dyn FileSystem, PathBuf)> {
        self.guard_synthesized(key, io::ErrorKind::IsADirectory)?;
        self.route(key).ok_or_else(not_found)
    }

    /// Every mount *strictly* below `prefix`.
    ///
    /// One contiguous run of the map, not a scan: component-wise `Ord` puts `["a"] <
    /// ["a","b"]` (equal prefix, shorter first) and `["a","z"] < ["b"]` (they diverge at the
    /// first component), so keys sharing a prefix are adjacent and a sibling like `ab` cannot
    /// interleave.
    ///
    /// The bound is `Excluded`, so a mount is not below itself. Both callers share this one
    /// range for that reason — an `Included` bound here would make a store that answers
    /// `NotFound` at its own root look like a synthesized directory, turning a misconfigured
    /// mount into a silently empty one.
    fn descendant_mounts<'a>(
        &'a self,
        prefix: &'a Path,
    ) -> impl Iterator<Item = (&'a PathBuf, &'a Box<dyn FileSystem>)> {
        self.mounts
            .range((Excluded(prefix.to_path_buf()), Unbounded))
            .take_while(move |(k, _)| k.starts_with(prefix))
    }

    /// The child names of `prefix` that come from the mount table, each flagged with whether a
    /// mount sits at *exactly* that path.
    ///
    /// The flag is what decides who wins a name collision with the store: an exact child is a
    /// mount point and shadows whatever the store had there, while a name that merely leads to
    /// a deeper mount keeps the store's entry.
    fn mount_children(&self, prefix: &Path) -> BTreeMap<String, bool> {
        let mut out = BTreeMap::new();
        for (mount, _) in self.descendant_mounts(prefix) {
            let rest = mount
                .strip_prefix(prefix)
                .expect("the range only yields keys prefixed by `prefix`");
            let mut components = rest.components();
            let Some(first) = components.next() else {
                continue;
            };
            let name = first.as_os_str().to_string_lossy().into_owned();
            // One component left over means the mount is at this child itself.
            let exact = components.next().is_none();
            *out.entry(name).or_insert(false) |= exact;
        }
        out
    }

    /// Whether any mount lies strictly below `prefix` — i.e. whether `prefix` is on the way to
    /// one, and so must present as a directory whether or not a store knows about it.
    fn spans_mounts(&self, prefix: &Path) -> bool {
        self.descendant_mounts(prefix).next().is_some()
    }

    /// The mount point that owns `key`, if any.
    ///
    /// Separate from [`route`](Self::route) because a move has to compare *which* mount each of
    /// its two paths belongs to. Comparing the resolved stores instead would work by pointer
    /// identity, which is a far more fragile thing to rest a correctness decision on.
    fn mount_for(&self, key: &Path) -> Option<&Path> {
        self.mounts
            .range(..=key.to_path_buf())
            .rev()
            .find(|(mount, _)| key.starts_with(mount))
            .map(|(mount, _)| mount.as_path())
    }

    /// Whether `key` belongs to the mount table rather than to a store — a mount point itself,
    /// or a directory that exists only because mounts lie below it.
    ///
    /// Neither is the filesystem's to move. A mount point is *not* caught by
    /// [`is_synthesized_dir`](Self::is_synthesized_dir): nothing lies below it, so it is a real
    /// path its own store serves — it just is not one a `rename` may relocate, because doing so
    /// would rewrite the mount table through a file operation.
    fn is_mount_table_owned(&self, key: &Path) -> bool {
        self.mounts.contains_key(key) || self.is_synthesized_dir(key)
    }

    /// Whether `key` is a directory that exists only because of the mount table.
    ///
    /// Either mounts lie below it — so it is on the way to one and must present as a directory
    /// whether or not a store knows about it — or it is the root of a tree with no mounts
    /// at all, which is an *empty* directory rather than a missing one, the way a freshly
    /// mounted tmpfs is.
    ///
    /// Deliberately not "the root, always": a tree whose root store cannot stat its own
    /// root is misconfigured, and there is no mount table to synthesize from, so that error
    /// surfaces instead of being papered over.
    fn is_synthesized_dir(&self, key: &Path) -> bool {
        self.spans_mounts(key) || (key.as_os_str().is_empty() && self.mounts.is_empty())
    }

    /// Refuse a mutation aimed at a directory the mount table synthesized.
    ///
    /// Checked *before* the store is consulted, because whether the path is a directory is
    /// knowable from the mount table alone — and a store asked first often answers `Ok`, which
    /// is how a mount ends up detached: `rmdir` succeeds on a directory the mount table still
    /// populates, `unlink` deletes a mount's parent, `create` makes a file where a mount has to
    /// be.
    fn guard_synthesized(&self, key: &Path, refusal: io::ErrorKind) -> io::Result<()> {
        if self.is_synthesized_dir(key) {
            Err(refusal.into())
        } else {
            Ok(())
        }
    }

    /// The error for a *create* at a path no mount claims.
    ///
    /// `ReadOnlyFilesystem` when the parent is a synthesized directory. This mirrors how a
    /// kernel orders the checks: a create resolves only the *parent*, that parent exists and
    /// was just listed, and the step that fails is the write — so `EROFS`, not the `ENOENT`
    /// that would claim a directory the caller can see is absent. Removals resolve the *target*
    /// instead, so they keep `NotFound`.
    ///
    /// `EROFS` and not `ENOSYS`: a FUSE kernel handed `ENOSYS` reads it as "this filesystem
    /// cannot do that at all" and disables the operation for the whole mount, taking every
    /// writable mount down with it. `EROFS` is per request.
    fn refusal_for_create(&self, key: &Path) -> io::Error {
        match key.parent() {
            Some(parent) if self.is_synthesized_dir(parent) => {
                io::ErrorKind::ReadOnlyFilesystem.into()
            }
            // The parent is not there either, so a component of the path really is missing —
            // the case POSIX answers with `ENOENT`.
            _ => not_found(),
        }
    }

    /// The metadata a directory that exists only in the mount table reports.
    fn synthesized_dir(&self) -> Stat {
        Stat {
            mtime: Some(self.born),
            ctime: Some(self.born),
            created: Some(self.born),
            ..Stat::new(DirentKind::Dir, 0)
        }
    }
}

impl Default for ContextFs {
    fn default() -> Self {
        Self::new()
    }
}

fn not_found() -> io::Error {
    io::ErrorKind::NotFound.into()
}

/// Normalize a path for use as a mount key, additionally refusing components that are not valid
/// UTF-8.
///
/// Only *mount* paths carry this restriction, not request paths: a request may legitimately
/// name a non-UTF-8 file inside a passthrough store. But a mount point's own name gets
/// *listed* — reported through [`Dirent::name`], which is a `String` — so a non-UTF-8 component
/// could only be shown lossily, and a lossy name does not round-trip. That would produce an
/// entry visible in a listing whose `lookup` then fails.
fn mount_key(path: &Path) -> io::Result<PathBuf> {
    let key = normalize(path)?;
    if key
        .components()
        .any(|component| component.as_os_str().to_str().is_none())
    {
        return Err(io::ErrorKind::InvalidFilename.into());
    }
    Ok(key)
}

/// Canonicalize a request into a root-relative path of `Normal` components only. `.` and a
/// leading root are dropped and `..` pops the previous component; any `..` that would escape the
/// tree's root, and OS prefixes, are rejected.
fn normalize(path: &Path) -> io::Result<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => out.push(name),
            Component::CurDir | Component::RootDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return Err(io::ErrorKind::InvalidFilename.into());
                }
            }
            Component::Prefix(_) => return Err(io::ErrorKind::InvalidFilename.into()),
        }
    }
    Ok(out)
}

impl FileSystem for ContextFs {
    /// Every mounted store, in turn: a reader asking for a refresh is asking the tree, and
    /// which of the stores under it was keeping something is not theirs to know.
    fn forget<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            for store in self.mounts.values() {
                store.forget().await;
            }
        })
    }

    fn stat<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let key = normalize(path)?;
            // Collapsing "no mount claims this" and "a mount claimed it and answered NotFound"
            // into one `Err(NotFound)` is what makes the arm below cover both. Only checking
            // whether routing failed would miss the second, which is the case a root store
            // produces for every path it does not know.
            let answered = match self.route(&key) {
                Some((store, sub)) => store.stat(&sub).await,
                None => Err(not_found()),
            };
            match answered {
                Err(err)
                    if err.kind() == io::ErrorKind::NotFound && self.is_synthesized_dir(&key) =>
                {
                    Ok(self.synthesized_dir())
                }
                // Anything else is the store's answer, including other errors: a
                // `PermissionDenied` must not become an empty directory.
                other => other,
            }
        })
    }

    fn list<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Vec<Dirent>>> {
        Box::pin(async move {
            let key = normalize(path)?;
            let children = self.mount_children(&key);

            // Same collapse as `stat`: no route and a routed `NotFound` are the same question.
            // A directory that exists only in the mount table simply has no store entries to
            // start from.
            let answered = match self.route(&key) {
                Some((store, sub)) => store.list(&sub).await,
                None => Err(not_found()),
            };
            let entries = match answered {
                Ok(entries) => entries,
                Err(err)
                    if err.kind() == io::ErrorKind::NotFound && self.is_synthesized_dir(&key) =>
                {
                    Vec::new()
                }
                other => return other,
            };

            // Split the store's entries by whether the mount table also names them, so each
            // name is emitted exactly once.
            let (mut named_by_mounts, store_only): (Vec<_>, Vec<_>) = entries
                .into_iter()
                .partition(|entry| children.contains_key(&entry.name));
            let mut claimed: BTreeMap<String, Dirent> = named_by_mounts
                .drain(..)
                .map(|entry| (entry.name.clone(), entry))
                .collect();

            // Mount-derived names go first, in sorted order, so their positions depend only on
            // the mount table. `readdir`'s cursor is a position in this list and the kernel
            // resumes by quoting one, so putting them last would let a single file appearing in
            // the store shift every mount point — dropping a whole mount out of an in-progress
            // listing.
            let mut out = Vec::with_capacity(children.len() + store_only.len());
            for (name, mounted_here) in children {
                let store_entry = claimed.remove(&name);
                out.push(match (mounted_here, store_entry) {
                    // A mount at this exact path shadows whatever the store had, the way any
                    // mount hides the directory it is mounted over. No stat: the kernel will
                    // `lookup` for metadata, and that lookup routes to the mount, so the two
                    // answers cannot disagree.
                    (true, _) => Dirent::new(name, DirentKind::Dir),
                    // Merely on the way to a deeper mount, and the store has it — keep its
                    // entry, which carries metadata a synthesized one lacks.
                    (false, Some(entry)) => entry,
                    (false, None) => Dirent::new(name, DirentKind::Dir),
                });
            }
            out.extend(store_only);
            Ok(out)
        })
    }

    fn read_at<'a>(
        &'a self,
        path: &'a Path,
        buf: &'a mut [u8],
        offset: u64,
    ) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move {
            let key = normalize(path)?;
            let (store, sub) = self.route_file(&key)?;
            store.read_at(&sub, buf, offset).await
        })
    }

    fn create<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let key = normalize(path)?;
            // The name is taken by a directory the mount table owns, which is what
            // `AlreadyExists` says — and what an exclusive create owes its caller.
            self.guard_synthesized(&key, io::ErrorKind::AlreadyExists)?;
            match self.route(&key) {
                Some((store, sub)) => store.create(&sub).await,
                None => Err(self.refusal_for_create(&key)),
            }
        })
    }

    fn mkdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let key = normalize(path)?;
            // Already a directory, so this is `EEXIST`.
            self.guard_synthesized(&key, io::ErrorKind::AlreadyExists)?;
            match self.route(&key) {
                Some((store, sub)) => store.mkdir(&sub).await,
                None => Err(self.refusal_for_create(&key)),
            }
        })
    }

    fn unlink<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let key = normalize(path)?;
            self.guard_synthesized(&key, io::ErrorKind::IsADirectory)?;
            match self.route(&key) {
                Some((store, sub)) => store.unlink(&sub).await,
                None => Err(not_found()),
            }
        })
    }

    fn rmdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let key = normalize(path)?;
            // Not empty: it holds the mount points below it, and those are not the caller's to
            // remove through the filesystem.
            self.guard_synthesized(&key, io::ErrorKind::DirectoryNotEmpty)?;
            match self.route(&key) {
                Some((store, sub)) => store.rmdir(&sub).await,
                None => Err(not_found()),
            }
        })
    }

    fn write_at<'a>(
        &'a self,
        path: &'a Path,
        buf: &'a [u8],
        offset: u64,
    ) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move {
            let key = normalize(path)?;
            let (store, sub) = self.route_file(&key)?;
            store.write_at(&sub, buf, offset).await
        })
    }

    fn truncate<'a>(&'a self, path: &'a Path, size: u64) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let key = normalize(path)?;
            let (store, sub) = self.route_file(&key)?;
            store.truncate(&sub, size).await
        })
    }

    fn flush<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let key = normalize(path)?;
            let (store, sub) = self.route_file(&key)?;
            store.flush(&sub).await
        })
    }

    fn rename<'a>(&'a self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let (from_key, to_key) = (normalize(from)?, normalize(to)?);

            // Neither end may be something the mount table owns. Checked before the stores are
            // consulted, because a refusal has to leave the tree untouched and a store asked
            // about its own root would answer something arbitrary.
            if self.is_mount_table_owned(&from_key) || self.is_mount_table_owned(&to_key) {
                return Err(io::ErrorKind::ReadOnlyFilesystem.into());
            }

            let (Some(mount), Some(destination)) =
                (self.mount_for(&from_key), self.mount_for(&to_key))
            else {
                // A move resolves its source before anything else, so a source nothing claims
                // is simply absent. The destination is a *create*, which is why it gets the
                // create refusal instead.
                return if self.mount_for(&from_key).is_none() {
                    Err(not_found())
                } else {
                    Err(self.refusal_for_create(&to_key))
                };
            };

            // Two stores cannot hand an object to each other, so this is `EXDEV` — and `mv`
            // reads that as "copy, then delete", which is the right recovery across a store
            // boundary. Doing the copy here instead could not be made atomic on partial
            // failure, and `mv` already knows how to deal with that.
            if mount != destination {
                return Err(io::ErrorKind::CrossesDevices.into());
            }

            let (store, from_sub) = self
                .route(&from_key)
                .expect("a mount was just found for this key");
            let to_sub = to_key
                .strip_prefix(mount)
                .expect("both paths share this mount");
            store.rename(&from_sub, to_sub).await
        })
    }
}
