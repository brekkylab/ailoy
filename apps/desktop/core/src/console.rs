//! Starting a console for one run.
//!
//! The server is cortex's own — `Backend::local`, which cortex builds and carries inside this
//! binary, and writes out under the app's data directory the first time a run needs it. So
//! there is no program to bundle beside the app, find at runtime, or keep at the version of
//! the cortex this was built against. `$CORTEX_LOCAL_CONSOLE_BIN` runs another build of it,
//! for working on the server itself.

use std::path::PathBuf;

use cortex::console::{Backend, Console, LocalBackend};

use crate::{
    error::{EngineError, Result},
    workspace::WorkspaceMount,
};

#[derive(Clone, Debug)]
pub struct ConsoleFactory {
    /// `None` is a factory that never spawns.
    backend: Option<LocalBackend>,
}

impl ConsoleFactory {
    /// Consoles on the local server, written out under `home` — cortex keeps its program in
    /// `<home>/bin`.
    pub fn new(home: PathBuf) -> Self {
        Self {
            backend: Some(Backend::local().home(home)),
        }
    }

    /// A factory that never spawns: runs proceed without a console (tools that need one fail
    /// saying so). For tests that have no use for one.
    pub fn disabled() -> Self {
        Self { backend: None }
    }

    pub fn is_disabled(&self) -> bool {
        self.backend.is_none()
    }

    /// One console over the three trees a run works with.
    ///
    /// * `context` — the user's own files and their connectors. Cortex refuses a write that
    ///   lands here, which is the point: this tree is managed outside the agent's life and an
    ///   agent reads it.
    /// * `artifacts` — what this agent produces. Part of the workspace the user sees, and the
    ///   one tree here the agent may write.
    /// * `scratch` — the run's `/tmp`. The session *starts* here, so a relative path a command
    ///   writes lands in something thrown away rather than among the user's files.
    ///
    /// The server's stderr is this process's, so its diagnostics go where the app's do.
    pub async fn spawn(
        &self,
        context: WorkspaceMount,
        artifacts: WorkspaceMount,
        scratch: WorkspaceMount,
    ) -> Result<Console> {
        let Some(backend) = self.backend.clone() else {
            return Err(EngineError::ConsoleUnavailable("console disabled".into()));
        };
        Console::builder()
            .backend(backend)
            .context(context)
            .artifacts(artifacts)
            .scratch(scratch)
            .build()
            .await
            .map_err(|e| EngineError::ConsoleUnavailable(format!("{e:#}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_factory_with_a_home_is_not_disabled() {
        let home = tempfile::tempdir().unwrap();
        assert!(!ConsoleFactory::new(home.path().to_path_buf()).is_disabled());
    }

    #[tokio::test]
    async fn a_disabled_factory_spawns_nothing() {
        let factory = ConsoleFactory::disabled();
        assert!(factory.is_disabled());
        let dir = tempfile::tempdir().unwrap();
        // `Console` is not `Debug`, so the error comes out of a match rather than `unwrap_err`.
        let Err(err) = factory
            .spawn(
                WorkspaceMount(dir.path().to_path_buf()),
                WorkspaceMount(dir.path().join("artifacts")),
                WorkspaceMount(dir.path().join("scratch")),
            )
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
