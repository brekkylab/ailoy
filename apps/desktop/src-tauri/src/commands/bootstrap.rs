use ailoy_desktop_core::BootstrapStatus;

use super::Eng;

/// What is still being downloaded before a chat can run. Changes arrive as the `bootstrap`
/// event (see `lib.rs`); this is for the window's first paint, which may come after some of
/// them.
#[tauri::command]
pub fn bootstrap_status(engine: Eng<'_>) -> BootstrapStatus {
    engine.bootstrap_status()
}

/// Run the steps that have not finished again. Answers at once, with the status as it is;
/// the steps report through the event as they go.
#[tauri::command]
pub fn bootstrap_retry(engine: Eng<'_>) -> BootstrapStatus {
    engine.bootstrap_retry()
}
