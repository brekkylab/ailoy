use ailoy_desktop_core::{EngineError, ModelInfo, Settings, SettingsPatch};

use super::Eng;

#[tauri::command]
pub async fn settings_get(engine: Eng<'_>) -> Result<Settings, EngineError> {
    engine.settings_get().await
}

#[tauri::command]
pub async fn settings_set(engine: Eng<'_>, patch: SettingsPatch) -> Result<Settings, EngineError> {
    engine.settings_set(patch).await
}

#[tauri::command]
pub fn models_list(engine: Eng<'_>) -> Result<Vec<ModelInfo>, EngineError> {
    engine.models_list()
}

/// Reveal the log directory in Finder.
#[tauri::command]
pub fn open_logs(engine: Eng<'_>) -> Result<(), EngineError> {
    let logs = engine.config().data_dir.join("logs");
    std::process::Command::new("open")
        .arg(logs)
        .spawn()
        .map_err(EngineError::Io)?;
    Ok(())
}
