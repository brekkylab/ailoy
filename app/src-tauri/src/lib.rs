mod dev_env;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    dev_env::load();
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![dev_env::dev_env_keys])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
