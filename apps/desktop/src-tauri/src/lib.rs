//! The window over the engine. Commands are thin; everything they answer comes from
//! `ailoy_desktop_core::Engine`.

mod commands;
mod logging;
mod sidecar;

use std::sync::Arc;

use ailoy_desktop_core::{Engine, EngineConfig};
use tauri::Manager;

/// Wraps a string as an AppleScript string literal.
///
/// Backslash first, or the escapes added after it would be escaped again. Newlines become
/// `\n` rather than staying literal: the message is passed as one `-e` argument, where a
/// raw newline is a new line of script, and the copy this is used for has two of them.
fn osascript_quoted(s: &str) -> String {
    let body = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{body}\"")
}

/// Says why the app is not starting, then stops the process.
///
/// Not `tauri-plugin-dialog`: its `blocking_show` must not be called from the main thread
/// (its own docs say so), and `setup` *is* the main thread — rfd's alert needs the very
/// event loop this would block, so the app hangs with no window and no dialog, which is
/// the one outcome worse than a silent exit. `osascript` is a separate process with its
/// own event loop, so a blocking `status()` on it is exactly what is wanted here.
///
/// `exit(1)` rather than `Err`: returning from `setup` surfaces as
/// `.expect("the window could not be created")`, a panic that names the wrong cause in
/// the crash report and in the log.
fn fail_to_start(e: &dyn std::fmt::Display) -> ! {
    // A Finder-launched app has no stderr, so the failure has to reach the log file and
    // the user's eyes before the process gives up. The most likely cause is a second
    // instance holding the data directory.
    tracing::error!("engine start failed: {e}");
    let msg = format!("Ailoy could not start.\n\n{e}");
    #[cfg(target_os = "macos")]
    {
        let script = format!(
            "display alert \"Ailoy\" message {} as critical",
            osascript_quoted(&msg)
        );
        let _ = std::process::Command::new("osascript")
            .args(["-e", &script])
            .status();
    }
    #[cfg(not(target_os = "macos"))]
    eprintln!("{msg}");
    std::process::exit(1);
}

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
                Err(e) => fail_to_start(&e),
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
            commands::client::client_log,
            commands::workspace::workspace_info,
            commands::workspace::workspace_set_root,
            commands::workspace::fs_list,
            commands::workspace::fs_read,
            commands::workspace::fs_read_bytes,
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

#[cfg(test)]
mod tests {
    use super::osascript_quoted;

    #[test]
    fn quoting_survives_a_message_full_of_shell_bait() {
        // An engine error carries a path, and a path can carry either character. Getting
        // this wrong would turn the one dialog the user gets into an AppleScript syntax
        // error — a silent exit with no explanation at all.
        assert_eq!(osascript_quoted("plain"), "\"plain\"");
        assert_eq!(osascript_quoted(r#"C:\a "b" \"#), r#""C:\\a \"b\" \\""#);
        // The Korean copy embeds two of these, and `-e` takes the whole script as one
        // argument: a raw newline here would end the statement mid-string.
        assert_eq!(osascript_quoted("a\nb"), r#""a\nb""#);
    }
}
