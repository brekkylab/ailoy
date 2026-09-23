use tauri::Manager;

mod agent;
mod cache;
mod context;
mod dev_env;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    dev_env::load();
    let cache = cache::Cache::open().expect("could not open the cache directory");
    let contexts = cache.contexts_dir();
    // The context first: the seeded agent names it.
    if let Err(err) = context::Context::seed(&contexts) {
        eprintln!("could not seed the default context: {err}");
    }
    if let Err(err) = agent::seed(&cache.agents()) {
        eprintln!("could not seed the agents: {err}");
    }
    tauri::Builder::default()
        .manage(cache)
        // The viewers read a context's files through the asset protocol. The scope is
        // granted here rather than in `tauri.conf.json`, because `AILOY_CACHE` can move it.
        .setup(move |app| {
            app.asset_protocol_scope()
                .allow_directory(&contexts, true)?;
            Ok(())
        })
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            dev_env::dev_env_keys,
            context::list_contexts,
            context::create_context,
            context::list_context_files,
            context::make_context_dir,
            context::add_context_files,
            context::remove_context_file,
            context::export_context,
            agent::list_agents,
            agent::get_helper_agent,
            agent::save_agent,
            agent::remove_agent,
            agent::export_agent,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
