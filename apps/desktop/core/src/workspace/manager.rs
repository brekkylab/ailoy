//! The workspace: one `ContextFs`, mounted for the life of the engine.
//!
//! Two kinds of tree live in it, and cortex keeps them apart on purpose. The user's own —
//! what they put there and what their connectors expose — is a session's *context*, which an
//! agent reads and may not write. What an agent produces goes in its *artifacts*, which is
//! grafted in at [`ARTIFACTS_PATH`] so it is part of the workspace the user sees rather than
//! somewhere else they have to go looking. (A session's third tree, its scratch, is not the
//! workspace's business: it is made per run and thrown away with it — see `run`.)

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex as StdMutex},
};

use cortex::fs::{ContextFs, FileSystem, FuseTMount, PassthroughFs};
use tokio::sync::RwLock;

use crate::{
    error::{EngineError, Result},
    types::{MountInfo, MountKind, MountStatus, WorkspaceInfo, WorkspaceStatus},
    workspace::shared::SharedFs,
};

/// Where the agent's output is grafted into the workspace, one segment under its root.
///
/// A fixed path rather than a setting: it is named in the system preamble, it is what the
/// file panel shows, and a connector may not take it — three places that would have to agree
/// about a configurable one.
pub const ARTIFACTS_PATH: &str = "artifacts";

pub struct WorkspaceManager {
    fs: Arc<RwLock<ContextFs>>,
    /// The host directory behind `/`, which the user can repoint while the app runs.
    ///
    /// A std mutex because `info()` is sync and reads it, and every write is a handful of
    /// instructions with no await inside the guard. It has to agree with the `""` mount in
    /// `fs` at all times, so `set_root` is the only thing that writes it.
    files_root: StdMutex<PathBuf>,
    /// Where the agent's own output goes, on the host.
    ///
    /// Handed to a console as its *artifacts* tree, and grafted into the workspace at
    /// `/artifacts` so the user sees it among their own files. Cortex is given this path
    /// rather than the one inside the mount: it refuses a write anywhere under the context,
    /// and the whole point of this tree is that the agent may write in it.
    artifacts_root: PathBuf,
    mountpoint: PathBuf,
    /// Held for the life of the engine; dropping it unmounts. Behind a std mutex because
    /// `FuseTMount` is dropped on a blocking thread at shutdown.
    mount: StdMutex<Option<FuseTMount>>,
    /// The workspace's own health, as `info()` reports it.
    ///
    /// Two writers, both of them terminal for the state they write: `mount_fuse` (reached
    /// once, from `start`) and `shutdown`. That is what keeps `info()`'s `try_read` fallback
    /// unreachable in practice — outside startup and shutdown nothing contends with a reader.
    /// A third writer (a remount command, a watchdog) would make it reachable, and a healthy
    /// mounted workspace would report `Degraded { reason: "busy" }` — and `console_context()`
    /// would hand back `files_root` — for the length of the write. Adding one means revisiting
    /// `info()` and `console_context()` first.
    status: RwLock<WorkspaceStatus>,
    mounts: RwLock<Vec<MountInfo>>,
}

impl WorkspaceManager {
    pub async fn start(
        files_root: PathBuf,
        artifacts_root: PathBuf,
        mountpoint: PathBuf,
        mount: bool,
    ) -> WorkspaceManager {
        // The files directory is made whether or not the workspace is mounted — it is what the
        // root `PassthroughFs` serves — so it is done first and its failure is reported as
        // itself. Reporting it as "mounting disabled" would leave a passthrough over a
        // directory that does not exist, and every file operation failing with a bare
        // `NotFound` and nothing to say about why. The mount point is only prepared when there
        // is going to be a mount.
        let status = match std::fs::create_dir_all(&files_root)
            .and_then(|()| std::fs::create_dir_all(&artifacts_root))
        {
            Err(e) => WorkspaceStatus::Degraded {
                reason: format!("files directory: {e}"),
            },
            Ok(()) if !mount => WorkspaceStatus::Degraded {
                reason: "mounting disabled".into(),
            },
            Ok(()) => match prepare_mount_point(&mountpoint) {
                Ok(()) => WorkspaceStatus::Mounted,
                Err(e) => WorkspaceStatus::Degraded {
                    reason: e.to_string(),
                },
            },
        };
        // The artifacts tree is grafted in beside the user's files, at a fixed path. It is
        // part of the workspace as the user sees it — output belongs where they are already
        // looking — while being a separate tree as far as a console is concerned.
        let fs = Arc::new(RwLock::new(
            ContextFs::new()
                .try_with_mount("", PassthroughFs::new(files_root.clone()))
                .expect("an empty path is a valid mount key")
                .try_with_mount(ARTIFACTS_PATH, PassthroughFs::new(artifacts_root.clone()))
                .expect("a one-segment path is a valid mount key"),
        ));
        let root = MountInfo {
            id: "root".into(),
            path: "/".into(),
            // A local mount like any other, and it is the user's own machine. What makes it
            // the root is its path: `detachable_path` refuses `/`, so it cannot be taken out
            // of the workspace the way a connector can.
            kind: MountKind::Local,
            // The one source that is the user's own machine, in a list beside Notion and
            // S3 where that is the distinction worth drawing.
            label: "My Computer".into(),
            detail: files_root.display().to_string(),
            writable: true,
            status: MountStatus::Ok,
        };
        let manager = WorkspaceManager {
            fs,
            files_root: StdMutex::new(files_root),
            artifacts_root,
            mountpoint,
            mount: StdMutex::new(None),
            status: RwLock::new(status),
            mounts: RwLock::new(vec![root]),
        };
        // The read guard is a temporary: it lives to the end of the condition and is dropped
        // before the block runs. That is what lets `mount_fuse` take the same lock for writing
        // on its failure path. Rewriting this as a `match` or `if let` over the guard would
        // hold it across the call and deadlock on the first mount failure.
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

    /// The host directory `/` currently serves.
    fn root(&self) -> PathBuf {
        self.files_root
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Point `/` at a different directory on the host.
    ///
    /// The directory is checked before anything is taken apart, because the swap itself has
    /// no failure to report: `mount` refuses a key that escapes the tree or one already
    /// taken, and `""` is neither once the unmount above has vacated it. So by the time the
    /// tree is briefly rootless, the only thing left to do is put the new store in.
    ///
    /// The FUSE mount is untouched. It serves this `ContextFs`, not any one store inside it,
    /// so a window looking at the workspace sees the new tree without a remount — and without
    /// the unmount/remount dance that would invalidate every open handle.
    pub async fn set_root(&self, host_root: PathBuf) -> Result<()> {
        let meta = std::fs::metadata(&host_root)
            .map_err(|e| EngineError::Invalid(format!("{}: {e}", host_root.display())))?;
        if !meta.is_dir() {
            return Err(EngineError::Invalid(format!(
                "{} is not a directory",
                host_root.display()
            )));
        }
        // Canonical, so the path stored and shown is the one the kernel will use — a root
        // given as a symlink would otherwise read back differently from what it serves.
        let root = std::fs::canonicalize(&host_root).unwrap_or(host_root);
        if root == self.root() {
            return Ok(());
        }
        {
            let mut fs = self.fs.write().await;
            let _ = fs.unmount("");
            fs.mount("", PassthroughFs::new(root.clone()))
                .expect("an empty path is a valid mount key, and was just vacated");
        }
        // No await between taking this guard and dropping it.
        *self.files_root.lock().unwrap_or_else(|e| e.into_inner()) = root.clone();
        if let Some(m) = self.mounts.write().await.iter_mut().find(|m| m.path == "/") {
            m.detail = root.display().to_string();
        }
        tracing::info!("workspace root is now {}", root.display());
        Ok(())
    }

    pub fn info(&self) -> WorkspaceInfo {
        WorkspaceInfo {
            mountpoint: self.mountpoint.clone(),
            files_root: self.root(),
            status: self.status.try_read().map(|s| s.clone()).unwrap_or(
                WorkspaceStatus::Degraded {
                    reason: "busy".into(),
                },
            ),
        }
    }

    /// The tree a console is given to *read*: the user's files and every connector under
    /// them, through the kernel when the workspace is mounted and straight off the disk when
    /// it is not. Cortex mounts this read-only — it is the user's, not the agent's.
    ///
    /// Degraded is the lesser tree on purpose: without the mount the connectors exist only
    /// inside this process, so a separate console can reach the passthrough root and nothing
    /// else.
    pub fn console_context(&self) -> PathBuf {
        match self.info().status {
            WorkspaceStatus::Mounted => self.mountpoint.clone(),
            WorkspaceStatus::Degraded { .. } => self.root(),
        }
    }

    /// The tree a console is given to *write*: where its output goes.
    ///
    /// The host path, never the one inside the mount. Cortex refuses a write to anything
    /// under the context, and `/artifacts` is visible there — so a console handed the mounted
    /// spelling would be refused on the one tree it is supposed to fill. The two paths are
    /// the same directory; the user reaches it through the workspace, the agent through this.
    pub fn console_artifacts(&self) -> PathBuf {
        self.artifacts_root.clone()
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
                    "{} already has another store connected",
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
        let path = detachable_path(path)?;
        let _ = self.fs.write().await.unmount(Path::new(&path));
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

    /// Take the mount down, and say so first.
    ///
    /// The status is written *before* the unmount because `console_context()` reads it: a
    /// caller that asked for a console between the unmount and the status change would be
    /// handed a mount point the kernel no longer serves, and every command in it would
    /// fail with `ENOENT` on the working directory rather than with anything a user could
    /// read. Reporting `Degraded` early costs nothing — `files_root` is the honest answer
    /// for the rest of the process's life.
    pub async fn shutdown(&self) {
        *self.status.write().await = WorkspaceStatus::Degraded {
            reason: "shutting down".into(),
        };
        let taken = self.mount.lock().expect("mount mutex").take();
        let Some(m) = taken else { return };
        let mountpoint = self.mountpoint.clone();
        // `FuseTMount::drop` asks the kernel to unmount and does not look at the answer, and
        // the answer here is often `EBUSY`: a `cortex-local-console` spawned for the last run
        // had its working directory *inside* this mount, and the process takes a moment to
        // die after the run that owned it ended. A mount left behind outlives the app — it
        // still shows in Finder, and anything that walks the data directory (a backup, a
        // `remove_dir_all`) blocks in it uninterruptibly — so the drop is checked and
        // escalated rather than trusted.
        let _ = tokio::task::spawn_blocking(move || {
            drop(m);
            for attempt in 1..=5 {
                if !is_mounted(&mountpoint) {
                    return;
                }
                // `force_unmount` tries `umount`, then `diskutil unmount force`, which
                // takes a busy mount down regardless of who is standing in it.
                match force_unmount(&mountpoint) {
                    Ok(()) => return,
                    Err(e) if attempt == 5 => {
                        tracing::error!("could not unmount {}: {e}", mountpoint.display());
                    }
                    // Still busy: the console is on its way out, so give it a moment.
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(200)),
                }
            }
        })
        .await;
    }
}

/// Make sure the mount point exists and is free: a mount left over from a crash is unmounted,
/// and anything else inside it is a refusal, not a deletion.
///
/// The stale-mount check comes *before* the directory is created, not after. `create_dir_all`
/// stats the path it is asked for, and a FUSE-T mount wedged by a crash — the userspace server
/// gone, the kernel still routing to it — is exactly the path whose `stat` never returns. Doing
/// it first would hang the engine at startup on the one case the cleanup below exists to
/// repair. `is_mounted` reads `mount(8)` and canonicalizes the mount point's *parent*, never
/// the mount point, so it is safe to ask about a path that is wedged or not there yet.
fn prepare_mount_point(mountpoint: &Path) -> std::io::Result<()> {
    if is_mounted(mountpoint) {
        tracing::warn!("stale mount at {}, unmounting", mountpoint.display());
        force_unmount(mountpoint)?;
    }
    std::fs::create_dir_all(mountpoint)?;
    if std::fs::read_dir(mountpoint)?.next().is_some() {
        return Err(std::io::Error::other(format!(
            "mount point {} is not empty",
            mountpoint.display()
        )));
    }
    Ok(())
}

/// The path `detach` will act on, or the refusal to touch the root.
///
/// A string compare against `"/"` is not the guard it looks like: `ContextFs::unmount` normalizes
/// its argument before the lookup, so `"//"`, `"/."` and `"/mem/.."` all arrive at the empty
/// root key. Each of those would take the root `PassthroughFs` out of the tree — and say
/// `Ok(())` while doing it, because the `retain` that follows matches no row — leaving the
/// workspace serving an empty root until the app restarts. So the path is put in normal form
/// first, and the guard is the normalizer's own: `normalize_mount_path` rebuilds the path from
/// its components and fails when nothing real is left, which is every spelling of the root.
///
/// Only that failure is rephrased. A `..` segment is refused with the normalizer's own words,
/// because "The root cannot be detached" is not true of `"/mem/.."` — the caller named a path,
/// and what is wrong with it is the `..`.
fn detachable_path(path: &str) -> Result<String> {
    let normalized = crate::workspace::connectors::normalize_mount_path(path).map_err(|e| {
        if path.trim().split('/').any(|s| s == "..") {
            e
        } else {
            EngineError::Invalid("The root cannot be detached".into())
        }
    })?;
    // The artifacts tree is the workspace's own, like the root: a connector can be taken out
    // of the workspace, but the place the agent's output lands cannot, or the next run has
    // nowhere to write.
    if normalized == format!("/{ARTIFACTS_PATH}") {
        return Err(EngineError::Invalid(
            "The artifacts directory cannot be detached".into(),
        ));
    }
    Ok(normalized)
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
    async fn the_root_can_be_repointed_at_another_directory() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first");
        let second = dir.path().join("second");
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(second.join("there.txt"), "b").unwrap();
        let ws = WorkspaceManager::start(
            first.clone(),
            dir.path().join("artifacts"),
            dir.path().join("workspace"),
            false,
        )
        .await;
        std::fs::write(first.join("here.txt"), "a").unwrap();

        let names = |v: Vec<crate::types::Entry>| {
            v.into_iter().map(|e| e.name).collect::<std::collections::BTreeSet<_>>()
        };
        assert!(names(fsops::list(&ws.fs(), "/").await.unwrap()).contains("here.txt"));

        ws.set_root(second.clone()).await.unwrap();

        // The tree serves the new directory, and only it. The artifacts graft is untouched:
        // it is a separate mount and the root swap does not reach it.
        let after = names(fsops::list(&ws.fs(), "/").await.unwrap());
        assert!(after.contains("there.txt"), "{after:?}");
        assert!(!after.contains("here.txt"), "{after:?}");
        assert!(after.contains("artifacts"), "{after:?}");

        // And everything that reports the root agrees with the tree.
        let canonical = std::fs::canonicalize(&second).unwrap();
        assert_eq!(ws.info().files_root, canonical);
        assert_eq!(ws.console_context(), canonical, "degraded serves the root itself");
        let row = ws.mounts().await.into_iter().find(|m| m.path == "/").unwrap();
        assert_eq!(row.detail, canonical.display().to_string());
        assert!(matches!(row.kind, MountKind::Local), "the root is a local mount");

        // A directory that is not one is refused, and refusing leaves the tree alone.
        assert!(ws.set_root(second.join("there.txt")).await.is_err());
        assert!(ws.set_root(dir.path().join("nope")).await.is_err());
        assert_eq!(ws.info().files_root, canonical);
        assert!(names(fsops::list(&ws.fs(), "/").await.unwrap()).contains("there.txt"));
    }

    #[tokio::test]
    async fn degraded_manager_serves_files_root_and_attaches_stores() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        let mp = dir.path().join("workspace");
        let ws =
            WorkspaceManager::start(files.clone(), dir.path().join("artifacts"), mp, false).await;
        assert!(matches!(ws.info().status, WorkspaceStatus::Degraded { .. }));
        assert_eq!(ws.console_context(), files);
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
        // Spelled with the trailing slash a sidebar sends: it has to reach the `/mem` row,
        // not a row of its own that nothing matches.
        ws.detach("/mem/").await.unwrap();
        assert!(
            ws.mounts().await.iter().all(|m| m.path != "/mem"),
            "detach(\"/mem/\") left the /mem row behind"
        );
        assert_eq!(ws.mounts().await.len(), 1);
        ws.shutdown().await;
    }

    /// Every spelling of the root that `ContextFs` normalizes back to its empty mount key has to be
    /// refused, not just the literal `"/"` — and the proof that it was is that the root store is
    /// still there afterwards.
    #[tokio::test]
    async fn detach_refuses_every_spelling_of_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let files = dir.path().join("files");
        let mp = dir.path().join("workspace");
        let ws =
            WorkspaceManager::start(files.clone(), dir.path().join("artifacts"), mp, false).await;

        for spelling in ["/", "", "//", "/.", "/mem/..", "/./", "  /  "] {
            assert!(
                matches!(ws.detach(spelling).await, Err(EngineError::Invalid(_))),
                "detach({spelling:?}) should have refused the root"
            );
        }

        // The root `PassthroughFs` survived all of it: a write still lands in `files_root`.
        fsops::write(&ws.fs(), "/still-here.txt", "root intact")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(files.join("still-here.txt")).unwrap(),
            "root intact"
        );
        assert_eq!(ws.mounts().await.len(), 1, "the root row");
        ws.shutdown().await;
    }
}
