//! What the window could not tell anyone.
//!
//! A packaged webview has a console nobody can open — no menu, no inspector unless the
//! build enabled one — so a failure in the frontend is invisible in exactly the situation
//! where it matters most: on someone else's machine, in a build they cannot attach to.
//! This is how the window gets a line into the engine's log, which is a file the user can
//! find and send.
//!
//! It is for failures the user has already been shown, not for tracing: the pane says one
//! sentence and this records the reason behind it.

use super::Eng;

/// Records a message from the window, under its own target so it reads as the window's.
///
/// `Eng` is taken and unused on purpose: it is what makes this a command of this app
/// rather than a general-purpose logging endpoint, and it keeps the signature honest if a
/// later version wants the engine's own state alongside the message.
#[tauri::command]
pub async fn client_log(_engine: Eng<'_>, level: String, message: String) -> Result<(), ()> {
    match level.as_str() {
        "warn" => tracing::warn!(target: "window", "{message}"),
        "error" => tracing::error!(target: "window", "{message}"),
        _ => tracing::info!(target: "window", "{message}"),
    }
    Ok(())
}
