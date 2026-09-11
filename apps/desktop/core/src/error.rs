//! The one error the engine answers in. Serialized as its message, because a webview
//! shows a string and nothing else.

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("session is already running")]
    AlreadyRunning,
    #[error("{0}")]
    Invalid(String),
    #[error("console unavailable: {0}")]
    ConsoleUnavailable(String),
    #[error("workspace: {0}")]
    Workspace(String),
    #[error("storage: {0}")]
    Storage(#[from] rusqlite::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl serde::Serialize for EngineError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;
