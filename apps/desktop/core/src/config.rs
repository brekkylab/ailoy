use std::path::PathBuf;

/// Where the engine keeps everything, and what it may start.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub data_dir: PathBuf,
    /// The `cortex-local-console` binary. `None` searches `AILOY_CORTEX_BIN_DIR`, then the
    /// sibling `cortex` checkout's `target/` — see `console::resolve_console_bin`.
    pub console_bin: Option<PathBuf>,
    pub catalog_refresh: bool,
    /// `false` skips the FUSE-T mount (tests, machines without FUSE-T). The console then
    /// stands directly in `files_root()`.
    pub mount_workspace: bool,
}

impl EngineConfig {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            console_bin: None,
            catalog_refresh: true,
            mount_workspace: true,
        }
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("ailoy.sqlite")
    }

    pub fn files_root(&self) -> PathBuf {
        self.data_dir.join("files")
    }

    pub fn mountpoint(&self) -> PathBuf {
        self.data_dir.join("workspace")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.data_dir.join("cache")
    }

    /// Where each run's throwaway directory is made.
    ///
    /// Cortex starts a session in its scratch tree, so everything a command writes to a
    /// relative path lands here rather than in the user's workspace. One directory per run,
    /// removed when the run ends; this is only the root they are made under.
    pub fn scratch_root(&self) -> PathBuf {
        self.data_dir.join("scratch")
    }
}
