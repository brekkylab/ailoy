use std::path::PathBuf;

use ailoy_desktop_core::{
    EngineError, Entry, FileContent, ImportReport, MountInfo, MountRequest, WorkspaceInfo,
};

use super::Eng;

#[tauri::command]
pub async fn workspace_info(engine: Eng<'_>) -> Result<WorkspaceInfo, EngineError> {
    Ok(engine.workspace_info())
}

#[tauri::command]
pub async fn workspace_set_root(
    engine: Eng<'_>,
    path: String,
) -> Result<WorkspaceInfo, EngineError> {
    engine.workspace_set_root(&path).await
}

#[tauri::command]
pub async fn fs_list(engine: Eng<'_>, path: String) -> Result<Vec<Entry>, EngineError> {
    engine.fs_list(&path).await
}

#[tauri::command]
pub async fn fs_read(engine: Eng<'_>, path: String) -> Result<FileContent, EngineError> {
    engine.fs_read(&path).await
}

/// A file's bytes, for the viewers that open a format rather than read characters.
///
/// `tauri::ipc::Response` rather than a `Vec<u8>` return: a plain vector is serialized as a
/// JSON array of numbers, which is roughly six bytes on the wire per byte of file. This
/// sends the buffer as it is, and the window receives an `ArrayBuffer`.
#[tauri::command]
pub async fn fs_read_bytes(
    engine: Eng<'_>,
    path: String,
) -> Result<tauri::ipc::Response, EngineError> {
    Ok(tauri::ipc::Response::new(
        engine.fs_read_bytes(&path).await?,
    ))
}

#[tauri::command]
pub async fn fs_write(engine: Eng<'_>, path: String, text: String) -> Result<(), EngineError> {
    engine.fs_write(&path, &text).await
}

#[tauri::command]
pub async fn fs_mkdir(engine: Eng<'_>, path: String) -> Result<(), EngineError> {
    engine.fs_mkdir(&path).await
}

#[tauri::command]
pub async fn fs_delete(engine: Eng<'_>, path: String) -> Result<(), EngineError> {
    engine.fs_delete(&path).await
}

#[tauri::command]
pub async fn fs_rename(engine: Eng<'_>, from: String, to: String) -> Result<(), EngineError> {
    engine.fs_rename(&from, &to).await
}

#[tauri::command]
pub async fn fs_import(
    engine: Eng<'_>,
    dest: String,
    sources: Vec<PathBuf>,
) -> Result<ImportReport, EngineError> {
    engine.fs_import(&dest, sources).await
}

#[tauri::command]
pub async fn mount_list(engine: Eng<'_>) -> Result<Vec<MountInfo>, EngineError> {
    Ok(engine.mount_list().await)
}

#[tauri::command]
pub async fn mount_add(engine: Eng<'_>, req: MountRequest) -> Result<MountInfo, EngineError> {
    engine.mount_add(req).await
}

#[tauri::command]
pub async fn mount_remove(engine: Eng<'_>, path: String) -> Result<(), EngineError> {
    engine.mount_remove(&path).await
}
