use std::sync::Arc;

use ailoy_desktop_core::Engine;

pub mod runs;
pub mod sessions;
pub mod settings;
pub mod workspace;

pub type Eng<'a> = tauri::State<'a, Arc<Engine>>;
