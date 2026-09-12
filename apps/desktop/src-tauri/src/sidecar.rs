use std::path::PathBuf;

/// The console server beside our own executable (where Tauri puts a sidecar), or wherever
/// `AILOY_CORTEX_BIN_DIR` says. `None` lets the engine search the sibling checkout.
pub fn console_bin() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("AILOY_CORTEX_BIN_DIR") {
        let p = PathBuf::from(dir).join("cortex-local-console");
        if p.is_file() {
            return Some(p);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let beside = exe.parent()?.join("cortex-local-console");
    beside.is_file().then_some(beside)
}
