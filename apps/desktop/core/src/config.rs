use std::path::PathBuf;

/// The store key holding the workspace root the user chose.
///
/// A setting rather than a config field: it is changed from the window, at runtime, and has
/// to survive a restart. [`EngineConfig::default_files_root`] is only what it falls back to.
pub const ROOT_SETTING: &str = "workspace.root";

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

    /// Where the workspace's root points when nothing has been chosen.
    ///
    /// The user's home directory, because the source is called My Computer and that is what
    /// it should be — their machine, not a folder the app made up inside its own data. The
    /// fallback under `data_dir` is for a process with no `HOME`, which on macOS means a
    /// launch context that has no user to have a home; it keeps the workspace working rather
    /// than refusing to start.
    ///
    /// This is a default, not the value: the chosen root lives in the store under
    /// [`ROOT_SETTING`] and can be changed while the app runs.
    pub fn default_files_root(&self) -> PathBuf {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| self.data_dir.join("files"))
    }

    pub fn mountpoint(&self) -> PathBuf {
        self.data_dir.join("workspace")
    }

    pub fn cache_dir(&self) -> PathBuf {
        self.data_dir.join("cache")
    }

    /// Where the agent's own output is kept.
    ///
    /// Its own directory beside `files/` rather than a folder inside it: cortex refuses a
    /// write anywhere under the tree it was given as context, so the place the agent writes
    /// has to be a tree of its own. The workspace shows it to the user all the same — the
    /// manager grafts it in at `/artifacts`.
    pub fn artifacts_root(&self) -> PathBuf {
        self.data_dir.join("artifacts")
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
