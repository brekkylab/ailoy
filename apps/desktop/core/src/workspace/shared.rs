//! The workspace as something that can be mounted, without giving it away.
//!
//! A binding takes ownership of what it serves — `FuseTMount::try_new(fs, …)` moves the
//! `FileSystem` in, keeps a raw pointer into it, and dereferences that on every request for the
//! life of the mount. That is the right shape for a program whose whole job is one mount, and
//! the wrong one for this window: connecting Notion is `ContextFs::mount`, which needs the
//! workspace back afterwards.
//!
//! So the thing handed to the binding is this — a handle onto the workspace rather than the
//! workspace. It is a [`FileSystem`] that forwards, the window keeps a clone, and a mount
//! taken down releases only the handle.
//!
//! **Every method takes the read lock, writes included.** That is not an oversight: the trait's
//! methods all take `&self`, so a store's own interior mutability is what serves a write, and
//! the only `&mut ContextFs` in the crate is the mount table — `mount` and `unmount`. A FUSE
//! request and a connector being attached are therefore a reader and a writer of the *table*,
//! which is exactly what they are.

use std::{io, path::Path, sync::Arc};

use cortex::{
    BoxFuture,
    fs::{ContextFs, Dirent, FileSystem, Stat},
};
use tokio::sync::RwLock;

/// A cloneable handle onto the window's workspace.
#[derive(Clone)]
pub struct SharedFs(Arc<RwLock<ContextFs>>);

impl SharedFs {
    pub fn new(fs: Arc<RwLock<ContextFs>>) -> Self {
        SharedFs(fs)
    }
}

impl FileSystem for SharedFs {
    fn forget<'a>(&'a self) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.forget().await;
        })
    }

    fn stat<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.stat(path).await
        })
    }

    fn list<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Vec<Dirent>>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.list(path).await
        })
    }

    fn read_at<'a>(
        &'a self,
        path: &'a Path,
        buf: &'a mut [u8],
        offset: u64,
    ) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.read_at(path, buf, offset).await
        })
    }

    fn create<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.create(path).await
        })
    }

    fn mkdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<Stat>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.mkdir(path).await
        })
    }

    fn unlink<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.unlink(path).await
        })
    }

    fn rmdir<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.rmdir(path).await
        })
    }

    fn write_at<'a>(
        &'a self,
        path: &'a Path,
        buf: &'a [u8],
        offset: u64,
    ) -> BoxFuture<'a, io::Result<usize>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.write_at(path, buf, offset).await
        })
    }

    fn truncate<'a>(&'a self, path: &'a Path, size: u64) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.truncate(path, size).await
        })
    }

    fn rename<'a>(&'a self, from: &'a Path, to: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.rename(from, to).await
        })
    }

    fn flush<'a>(&'a self, path: &'a Path) -> BoxFuture<'a, io::Result<()>> {
        Box::pin(async move {
            let fs = self.0.read().await;
            fs.flush(path).await
        })
    }
}
