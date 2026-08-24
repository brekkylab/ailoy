//! Where a run is allowed to write.
//!
//! The stores under `live/` refuse writes themselves, so nothing here needs to guard
//! them. What is left is the ordinary part of the tree: the directory the mounts sit
//! in, and everything else the session can see. An agent told to write only under
//! `artifacts/` and `workspace/` might still not, and this is what notices.
//!
//! Detection rather than prevention, because commands run on the host with the
//! session's own rights. Saying that plainly is the point — a check described as
//! enforcement would be believed.

use std::path::{Component, Path, PathBuf};

use anyhow::{Result, bail};

/// The directories a run may write into.
pub struct WriteBoundary {
    allowed: Vec<PathBuf>,
}

impl WriteBoundary {
    pub fn new(allowed: impl IntoIterator<Item = PathBuf>) -> Self {
        Self {
            allowed: allowed.into_iter().map(|p| normalize(&p)).collect(),
        }
    }

    /// Whether `path` falls inside one of the allowed directories.
    pub fn permits(&self, path: &Path) -> bool {
        let p = normalize(path);
        self.allowed.iter().any(|a| p.starts_with(a))
    }

    /// Like [`permits`](Self::permits), but says which path and against what.
    pub fn check(&self, path: &Path) -> Result<()> {
        if !self.permits(path) {
            bail!(
                "{} is outside the writable set {:?}",
                path.display(),
                self.allowed
            );
        }
        Ok(())
    }
}

/// Fold `.` and `..` by hand, without asking the filesystem.
///
/// [`canonicalize`](std::fs::canonicalize) would be the obvious way and cannot be used:
/// it fails on a path that does not exist yet, and the paths being checked are mostly
/// files a run is about to create. Resolving textually also means a `..` cannot escape
/// by pointing through a symlink the check never sees.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boundary() -> WriteBoundary {
        WriteBoundary::new([PathBuf::from("./artifacts/run-1"), PathBuf::from("./workspace/run-1")])
    }

    #[test]
    fn a_path_inside_is_permitted() {
        let b = boundary();
        assert!(b.permits(Path::new("artifacts/run-1/report.md")));
        assert!(b.permits(Path::new("./artifacts/run-1/queries/01.py")));
        assert!(b.permits(Path::new("workspace/run-1/scratch.json")));
    }

    #[test]
    fn traversal_and_siblings_are_refused() {
        let b = boundary();
        // The obvious escape.
        assert!(!b.permits(Path::new("artifacts/run-1/../../etc/passwd")));
        // And the one that looks like the allowed directory without being under it.
        assert!(!b.permits(Path::new("artifacts/run-10/report.md")));
        assert!(!b.permits(Path::new("live/gleif/CATALOG.md")));
        assert!(b.check(Path::new("/tmp/elsewhere")).is_err());
    }

    #[test]
    fn a_path_that_does_not_exist_is_still_checkable() {
        // The whole reason for folding `..` textually: nothing here is on disk, and
        // `canonicalize` would fail rather than answer.
        let b = boundary();
        assert!(b.permits(Path::new("artifacts/run-1/not/created/yet.md")));
    }
}
