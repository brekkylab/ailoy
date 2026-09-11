//! Starting a `cortex-local-console` for one run.

use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

use cortex::console::{Console, stdio::StdioClient};
use tokio::process::Command;

use crate::{
    error::{EngineError, Result},
    workspace::WorkspaceMount,
};

pub const CONSOLE_BIN_NAME: &str = "cortex-local-console";

/// Where the console server binary is. In a bundle it sits beside the app binary (the Tauri
/// layer passes that path explicitly); in development it is the sibling checkout's build.
pub fn resolve_console_bin(explicit: Option<&Path>) -> Result<PathBuf> {
    let env_dir = std::env::var_os("AILOY_CORTEX_BIN_DIR").map(PathBuf::from);
    resolve_console_bin_in(explicit, env_dir.as_deref())
}

/// [`resolve_console_bin`] with `$AILOY_CORTEX_BIN_DIR` already read, so a test can hand in a
/// directory without mutating the process environment.
pub(crate) fn resolve_console_bin_in(
    explicit: Option<&Path>,
    env_dir: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return if p.is_file() {
            Ok(p.to_path_buf())
        } else {
            Err(EngineError::ConsoleUnavailable(format!(
                "{} does not exist",
                p.display()
            )))
        };
    }
    if let Some(dir) = env_dir {
        let p = dir.join(CONSOLE_BIN_NAME);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            for profile in ["debug", "release"] {
                let p = ancestor
                    .join("cortex")
                    .join("target")
                    .join(profile)
                    .join(CONSOLE_BIN_NAME);
                if p.is_file() {
                    return Ok(p);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let p = dir.join(CONSOLE_BIN_NAME);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err(EngineError::ConsoleUnavailable(format!(
        "{CONSOLE_BIN_NAME} not found; set AILOY_CORTEX_BIN_DIR or build it with `cargo build -p cortex-local-console` in ../cortex"
    )))
}

#[derive(Clone, Debug)]
pub struct ConsoleFactory {
    bin: PathBuf,
}

impl ConsoleFactory {
    pub fn new(bin: PathBuf) -> Self {
        Self { bin }
    }

    /// A factory that never spawns: runs proceed without a console (tools that need one fail
    /// saying so). For tests and for a machine with no console binary.
    pub fn disabled() -> Self {
        Self {
            bin: PathBuf::new(),
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.bin.as_os_str().is_empty()
    }

    pub fn bin(&self) -> &Path {
        &self.bin
    }

    /// One console standing in `mount`. Its `PATH` starts with the binary's own directory so
    /// sidecars beside it (`mem`, later) are commands the agent can name.
    pub async fn spawn(&self, mount: WorkspaceMount) -> Result<Console> {
        if self.is_disabled() {
            return Err(EngineError::ConsoleUnavailable("console disabled".into()));
        }
        let mut cmd = Command::new(&self.bin);
        // Deliberately inherited (tokio's default, spelled out): the server's diagnostics go
        // where the app's do. Piping it into `tracing` is the later upgrade, not `null`.
        cmd.stderr(Stdio::inherit());
        if let Some(dir) = self.bin.parent().filter(|d| !d.as_os_str().is_empty()) {
            let mut dirs = vec![dir.to_path_buf()];
            if let Some(existing) = std::env::var_os("PATH") {
                dirs.extend(std::env::split_paths(&existing));
            }
            if let Ok(path) = std::env::join_paths(dirs) {
                cmd.env("PATH", path);
            }
        }
        let client = StdioClient::new(cmd)
            .map_err(|e| EngineError::ConsoleUnavailable(format!("{}: {e}", self.bin.display())))?;
        Console::builder()
            .client(client)
            .mount(mount)
            .build()
            .await
            .map_err(|e| EngineError::ConsoleUnavailable(format!("{}: {e:#}", self.bin.display())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_path_wins_when_it_exists() {
        let f = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(resolve_console_bin(Some(f.path())).unwrap(), f.path());
        assert!(matches!(
            resolve_console_bin(Some(Path::new("/nonexistent/bin"))),
            Err(EngineError::ConsoleUnavailable(_))
        ));
    }

    #[test]
    fn env_dir_is_honoured_and_loses_to_an_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join(CONSOLE_BIN_NAME);
        std::fs::write(&bin, b"").unwrap();
        assert_eq!(resolve_console_bin_in(None, Some(dir.path())).unwrap(), bin);
        let explicit = tempfile::NamedTempFile::new().unwrap();
        assert_eq!(
            resolve_console_bin_in(Some(explicit.path()), Some(dir.path())).unwrap(),
            explicit.path()
        );
    }

    #[test]
    fn a_factory_with_a_binary_is_not_disabled() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let factory = ConsoleFactory::new(f.path().to_path_buf());
        assert!(!factory.is_disabled());
        assert_eq!(factory.bin(), f.path());
    }

    #[tokio::test]
    async fn a_disabled_factory_spawns_nothing() {
        let factory = ConsoleFactory::disabled();
        assert!(factory.is_disabled());
        let dir = tempfile::tempdir().unwrap();
        // `Console` is not `Debug`, so the error comes out of a match rather than `unwrap_err`.
        let Err(err) = factory
            .spawn(WorkspaceMount(dir.path().to_path_buf()))
            .await
        else {
            panic!("a disabled factory spawned a console");
        };
        assert!(
            matches!(&err, EngineError::ConsoleUnavailable(m) if m == "console disabled"),
            "{err}"
        );
    }
}
