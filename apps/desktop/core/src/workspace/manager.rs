//! The workspace: one `WorkFs`, mounted for the life of the engine.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex as StdMutex},
};

use cortex::fs::{FileSystem, FuseTMount, PassthroughFs, WorkFs};
use tokio::sync::RwLock;

use crate::{
    error::{EngineError, Result},
    types::{MountInfo, MountKind, MountStatus, WorkspaceInfo, WorkspaceStatus},
    workspace::{mount::WorkspaceMount, shared::SharedFs},
};

pub struct WorkspaceManager {
    fs: Arc<RwLock<WorkFs>>,
    files_root: PathBuf,
    mountpoint: PathBuf,
    /// Held for the life of the engine; dropping it unmounts. Behind a std mutex because
    /// `FuseTMount` is dropped on a blocking thread at shutdown.
    mount: StdMutex<Option<FuseTMount>>,
    status: RwLock<WorkspaceStatus>,
    mounts: RwLock<Vec<MountInfo>>,
}

impl WorkspaceManager {
    pub async fn start(files_root: PathBuf, mountpoint: PathBuf, mount: bool) -> WorkspaceManager {
        let status = match prepare_dirs(&files_root, &mountpoint) {
            Ok(()) => WorkspaceStatus::Mounted,
            Err(e) => WorkspaceStatus::Degraded {
                reason: e.to_string(),
            },
        };
        let fs = Arc::new(RwLock::new(
            WorkFs::new()
                .try_with_mount("", PassthroughFs::new(files_root.clone()))
                .expect("an empty path is a valid mount key"),
        ));
        let root = MountInfo {
            id: "root".into(),
            path: "/".into(),
            kind: MountKind::Root,
            label: "Workspace".into(),
            detail: files_root.display().to_string(),
            writable: true,
            status: MountStatus::Ok,
        };
        let manager = WorkspaceManager {
            fs,
            files_root,
            mountpoint,
            mount: StdMutex::new(None),
            status: RwLock::new(if mount {
                status
            } else {
                WorkspaceStatus::Degraded {
                    reason: "mounting disabled".into(),
                }
            }),
            mounts: RwLock::new(vec![root]),
        };
        if mount && matches!(*manager.status.read().await, WorkspaceStatus::Mounted) {
            manager.mount_fuse().await;
        }
        manager
    }

    async fn mount_fuse(&self) {
        let shared = SharedFs::new(self.fs.clone());
        let at = self.mountpoint.clone();
        // Mounting blocks on a helper handshake, so it runs off the runtime.
        let result = tokio::task::spawn_blocking(move || FuseTMount::try_new(shared, &at)).await;
        match result {
            Ok(Ok(m)) => {
                *self.mount.lock().expect("mount mutex") = Some(m);
                tracing::info!("workspace mounted at {}", self.mountpoint.display());
            }
            Ok(Err(e)) => {
                tracing::warn!("workspace mount failed: {e}");
                *self.status.write().await = WorkspaceStatus::Degraded {
                    reason: format!("FUSE-T mount failed: {e}"),
                };
            }
            Err(e) => {
                *self.status.write().await = WorkspaceStatus::Degraded {
                    reason: format!("mount task panicked: {e}"),
                };
            }
        }
    }

    pub fn info(&self) -> WorkspaceInfo {
        WorkspaceInfo {
            mountpoint: self.mountpoint.clone(),
            files_root: self.files_root.clone(),
            status: self.status.try_read().map(|s| s.clone()).unwrap_or(
                WorkspaceStatus::Degraded {
                    reason: "busy".into(),
                },
            ),
        }
    }

    pub fn console_mount(&self) -> WorkspaceMount {
        match self.info().status {
            WorkspaceStatus::Mounted => WorkspaceMount(self.mountpoint.clone()),
            WorkspaceStatus::Degraded { .. } => WorkspaceMount(self.files_root.clone()),
        }
    }

    pub fn fs(&self) -> SharedFs {
        SharedFs::new(self.fs.clone())
    }

    /// Record then mount, in that order: the list is what refuses a duplicate, and the tree
    /// has no undo that leaves the first store where it was.
    pub async fn attach(&self, info: MountInfo, store: Arc<dyn FileSystem>) -> Result<()> {
        {
            let mut mounts = self.mounts.write().await;
            if mounts.iter().any(|m| m.path == info.path) {
                return Err(EngineError::Invalid(format!(
                    "{} 에는 이미 다른 저장소가 연결되어 있습니다",
                    info.path
                )));
            }
            mounts.push(info.clone());
            mounts.sort_by(|a, b| a.path.cmp(&b.path));
        }
        if let Err(e) = self.fs.write().await.mount(Path::new(&info.path), store) {
            self.mounts.write().await.retain(|m| m.path != info.path);
            return Err(EngineError::Workspace(e.to_string()));
        }
        Ok(())
    }

    pub async fn detach(&self, path: &str) -> Result<()> {
        if path == "/" || path.is_empty() {
            return Err(EngineError::Invalid("루트는 분리할 수 없습니다".into()));
        }
        let _ = self.fs.write().await.unmount(Path::new(path));
        self.mounts.write().await.retain(|m| m.path != path);
        Ok(())
    }

    pub async fn mounts(&self) -> Vec<MountInfo> {
        self.mounts.read().await.clone()
    }

    pub async fn set_mount_status(&self, path: &str, status: MountStatus) {
        let mut mounts = self.mounts.write().await;
        if let Some(m) = mounts.iter_mut().find(|m| m.path == path) {
            m.status = status;
        }
    }

    /// Remember a connector that failed to restore, so the sidebar can show it with its error.
    pub async fn remember_failed(&self, info: MountInfo) {
        let mut mounts = self.mounts.write().await;
        mounts.retain(|m| m.path != info.path);
        mounts.push(info);
        mounts.sort_by(|a, b| a.path.cmp(&b.path));
    }

    pub async fn shutdown(&self) {
        let taken = self.mount.lock().expect("mount mutex").take();
        if let Some(m) = taken {
            let _ = tokio::task::spawn_blocking(move || drop(m)).await;
        }
    }
}

/// Create the two directories, and make sure the mount point is free: a mount left over
/// from a crash is unmounted, and anything else inside it is a refusal, not a deletion.
fn prepare_dirs(files_root: &Path, mountpoint: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(files_root)?;
    std::fs::create_dir_all(mountpoint)?;
    if is_mounted(mountpoint) {
        tracing::warn!("stale mount at {}, unmounting", mountpoint.display());
        force_unmount(mountpoint)?;
    }
    if std::fs::read_dir(mountpoint)?.next().is_some() {
        return Err(std::io::Error::other(format!(
            "mount point {} is not empty",
            mountpoint.display()
        )));
    }
    Ok(())
}

pub fn is_mounted(path: &Path) -> bool {
    let Ok(out) = Command::new("mount").output() else {
        return false;
    };
    let table = String::from_utf8_lossy(&out.stdout);
    // Both spellings, because `mount` reports the path the kernel resolved to and the caller
    // holds the one it was configured with — on macOS those differ wherever a symlink is on
    // the way, `/var` → `/private/var` being the one every temp directory goes through.
    let resolved = kernel_path(path);
    [path, resolved.as_path()]
        .iter()
        .any(|p| table.contains(&format!(" on {} (", p.display())))
}

/// `path` with its symlinks resolved, or `path` unchanged when it cannot be.
///
/// The *parent* is what gets resolved, not the path itself: this is asked about mount points,
/// and one left wedged by a crash is exactly the thing that must not be walked into to find
/// out whether it is still there.
fn kernel_path(path: &Path) -> PathBuf {
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => match std::fs::canonicalize(parent) {
            Ok(parent) => parent.join(name),
            Err(_) => path.to_path_buf(),
        },
        _ => path.to_path_buf(),
    }
}

pub fn force_unmount(path: &Path) -> std::io::Result<()> {
    let _ = Command::new("umount").arg(path).status();
    if is_mounted(path) {
        let _ = Command::new("diskutil")
            .args(["unmount", "force"])
            .arg(path)
            .status();
    }
    if is_mounted(path) {
        return Err(std::io::Error::other(format!(
            "could not unmount {}",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use cortex::fs::InMemFs;

    use super::*;
    use crate::{
        types::{MountKind, MountStatus},
        workspace::fsops,
    };

    #[tokio::test]
    async fn degraded_manager_serves_files_root_and_attaches_stores() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        let mp = dir.path().join("workspace");
        let ws = WorkspaceManager::start(files.clone(), mp, false).await;
        assert!(matches!(ws.info().status, WorkspaceStatus::Degraded { .. }));
        assert_eq!(ws.console_mount().0, files);
        assert_eq!(ws.mounts().await.len(), 1, "the root row");

        // Root is the real directory: a write shows up on disk.
        fsops::write(&ws.fs(), "/hello.txt", "hi").await.unwrap();
        assert_eq!(
            std::fs::read_to_string(files.join("hello.txt")).unwrap(),
            "hi"
        );

        let info = MountInfo {
            id: "m".into(),
            path: "/mem".into(),
            kind: MountKind::Local,
            label: "mem".into(),
            detail: "".into(),
            writable: true,
            status: MountStatus::Ok,
        };
        ws.attach(info.clone(), Arc::new(InMemFs::new()))
            .await
            .unwrap();
        assert!(
            ws.attach(info, Arc::new(InMemFs::new())).await.is_err(),
            "duplicate path"
        );
        fsops::write(&ws.fs(), "/mem/a.txt", "x").await.unwrap();
        assert_eq!(fsops::list(&ws.fs(), "/mem").await.unwrap().len(), 1);
        ws.detach("/mem").await.unwrap();
        assert_eq!(ws.mounts().await.len(), 1);
        ws.shutdown().await;
    }
}
