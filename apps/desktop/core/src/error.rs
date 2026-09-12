//! The one error the engine answers in.
//!
//! It crosses the IPC boundary as `{ "kind": "...", "message": "..." }`: the `kind` is a
//! stable snake_case tag a webview can branch on (offer a retry, send the user to the
//! settings pane, reload a stale session list), and the `message` is the `Display` text,
//! already written in the user's language. Plan C generates its TypeScript union from
//! [`EngineError::kind`]'s arms, so adding a variant means adding a tag there too.

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

impl EngineError {
    /// The machine-readable tag this error serializes as. Part of the IPC contract: these
    /// strings are what the webview matches on, so renaming one is a breaking change.
    pub fn kind(&self) -> &'static str {
        match self {
            EngineError::NotFound(_) => "not_found",
            EngineError::AlreadyRunning => "already_running",
            EngineError::Invalid(_) => "invalid",
            EngineError::ConsoleUnavailable(_) => "console_unavailable",
            EngineError::Workspace(_) => "workspace",
            EngineError::Storage(_) => "storage",
            EngineError::Io(_) => "io",
            EngineError::Other(_) => "other",
        }
    }
}

impl serde::Serialize for EngineError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct as _;
        let mut st = s.serialize_struct("EngineError", 2)?;
        st.serialize_field("kind", self.kind())?;
        st.serialize_field("message", &self.to_string())?;
        st.end()
    }
}

pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape Plan C generates against: a two-field object, never a bare string.
    #[test]
    fn an_engine_error_serializes_as_kind_and_message() {
        assert_eq!(
            serde_json::to_value(EngineError::NotFound("session s1".into())).unwrap(),
            serde_json::json!({ "kind": "not_found", "message": "not found: session s1" })
        );
        assert_eq!(
            serde_json::to_value(EngineError::AlreadyRunning).unwrap(),
            serde_json::json!({
                "kind": "already_running",
                "message": "session is already running",
            })
        );
        // `Invalid` carries messages the user reads verbatim — the Korean survives the
        // round trip rather than arriving as escapes.
        assert_eq!(
            serde_json::to_value(EngineError::Invalid("제목을 입력해 주세요".into())).unwrap()["message"],
            serde_json::json!("제목을 입력해 주세요")
        );
    }
}
