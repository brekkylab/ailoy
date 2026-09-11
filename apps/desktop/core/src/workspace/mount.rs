//! Where the console stands.

use std::path::{Path, PathBuf};

use cortex::fs::Mount;

/// Where the console stands: the FUSE-T mount point when the workspace is mounted, the
/// plain files directory when it is not. `Mount` asks for nothing but the path — a
/// directory answers a console the same way a mount point does.
pub struct WorkspaceMount(pub PathBuf);

impl Mount for WorkspaceMount {
    fn mountpoint(&self) -> &Path {
        &self.0
    }
}
