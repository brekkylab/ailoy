use ailoy_desktop_core::{CatalogStatus, EngineError, ModelInfo, Settings, SettingsPatch};

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

/// Where the model list came from. Changes arrive as the `catalog` event (see `lib.rs`);
/// this is for the window's first paint, which may come before any of them.
#[tauri::command]
pub fn catalog_status(engine: Eng<'_>) -> CatalogStatus {
    engine.catalog_status()
}

/// Fetch the model list now. Resolves once the fetch has ended either way; `error` in the
/// status is how a failure shows.
#[tauri::command]
pub async fn models_refresh(engine: Eng<'_>) -> Result<CatalogStatus, EngineError> {
    Ok(engine.models_refresh().await)
}

/// Reveal the log directory in Finder.
#[tauri::command]
pub fn open_logs(engine: Eng<'_>) -> Result<(), EngineError> {
    let logs = engine.config().data_dir.join("logs");
    let mut child = std::process::Command::new("open")
        .arg(logs)
        .spawn()
        .map_err(EngineError::Io)?;
    // `open` returns at once; reap it off-thread so each click does not leave a zombie.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}
