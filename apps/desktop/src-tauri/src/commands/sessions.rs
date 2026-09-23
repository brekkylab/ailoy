use ailoy_desktop_core::{EngineError, SessionSummary, SessionUsage, StoredMessage};

use super::Eng;

#[tauri::command]
pub async fn session_list(engine: Eng<'_>) -> Result<Vec<SessionSummary>, EngineError> {
    engine.session_list().await
}

#[tauri::command]
pub async fn session_create(
    engine: Eng<'_>,
    model: Option<String>,
) -> Result<SessionSummary, EngineError> {
    engine.session_create(model).await
}

#[tauri::command]
pub async fn session_rename(engine: Eng<'_>, id: String, title: String) -> Result<(), EngineError> {
    engine.session_rename(&id, &title).await
}

#[tauri::command]
pub async fn session_set_model(
    engine: Eng<'_>,
    id: String,
    model: String,
) -> Result<(), EngineError> {
    engine.session_set_model(&id, &model).await
}

#[tauri::command]
pub async fn session_delete(engine: Eng<'_>, id: String) -> Result<(), EngineError> {
    engine.session_delete(&id).await
}

#[tauri::command]
pub async fn message_list(
    engine: Eng<'_>,
    session_id: String,
) -> Result<Vec<StoredMessage>, EngineError> {
    engine.message_list(&session_id).await
}

#[tauri::command]
pub async fn session_usage(
    engine: Eng<'_>,
    session_id: String,
) -> Result<SessionUsage, EngineError> {
    engine.session_usage(&session_id).await
}
