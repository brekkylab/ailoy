//! The window over the engine. Commands are thin; everything they answer comes from
//! `ailoy_desktop_core::Engine`.

mod commands;
mod logging;
mod sidecar;

use std::sync::Arc;

use ailoy_desktop_core::{Engine, EngineConfig};
use tauri::Manager;
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            logging::init(&data_dir);
            let mut cfg = EngineConfig::new(data_dir);
            cfg.console_bin = sidecar::console_bin();
            // Setup runs on the main thread; the engine's start is short (open the DB,
            // mount, register providers) and must finish before any command can arrive.
            let engine = match tauri::async_runtime::block_on(Engine::start(cfg)) {
                Ok(engine) => engine,
                Err(e) => {
                    // A Finder-launched app has no stderr, so the failure has to reach the
                    // log file and the user's eyes before the process gives up. The most
                    // likely cause is a second instance holding the data directory.
                    tracing::error!("engine start failed: {e}");
                    app.dialog()
                        .message(e.to_string())
                        .title("Ailoy")
                        .kind(MessageDialogKind::Error)
                        .blocking_show();
                    return Err(e.into());
                }
            };
            app.manage(engine);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::sessions::session_list,
            commands::sessions::session_create,
            commands::sessions::session_rename,
            commands::sessions::session_set_model,
            commands::sessions::session_delete,
            commands::sessions::message_list,
            commands::sessions::session_usage,
            commands::runs::run_start,
            commands::runs::run_attach,
            commands::runs::run_cancel,
            commands::workspace::workspace_info,
            commands::workspace::fs_list,
            commands::workspace::fs_read,
            commands::workspace::fs_write,
            commands::workspace::fs_mkdir,
            commands::workspace::fs_delete,
            commands::workspace::fs_rename,
            commands::workspace::fs_import,
            commands::workspace::mount_list,
            commands::workspace::mount_add,
            commands::workspace::mount_remove,
            commands::settings::settings_get,
            commands::settings::settings_set,
            commands::settings::models_list,
            commands::settings::open_logs,
        ])
        .build(tauri::generate_context!())
        .expect("the window could not be created")
        .run(|app, event| {
            // Closing the last window raises `ExitRequested`; the macOS Quit menu item
            // (Cmd+Q) goes through `applicationWillTerminate:` and raises only `Exit`.
            // Both must reach the engine, or Cmd+Q leaks the FUSE-T mount and drops the
            // trailing partial message. Running shutdown twice is safe: the mount is
            // `take()`n and the run list is drained on the first pass.
            if matches!(
                event,
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
            ) {
                if let Some(engine) = app.try_state::<Arc<Engine>>() {
                    let engine = engine.inner().clone();
                    // Unmounting joins a thread; do it before the process goes.
                    tauri::async_runtime::block_on(engine.shutdown());
                }
            }
        });
}
