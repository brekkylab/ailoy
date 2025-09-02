#[cfg(not(target_arch = "wasm32"))]
use log;
#[cfg(target_arch = "wasm32")]
use web_sys;

pub fn debug(s: impl Into<String> + std::fmt::Display) {
    #[cfg(not(target_arch = "wasm32"))]
    log::debug!("{}", s);

    #[cfg(target_arch = "wasm32")]
    web_sys::console::debug_1(&s.to_string().into());
}

pub fn info(s: impl Into<String> + std::fmt::Display) {
    #[cfg(not(target_arch = "wasm32"))]
    log::info!("{}", s);

    #[cfg(target_arch = "wasm32")]
    web_sys::console::info_1(&s.to_string().into());
}

pub fn warn(s: impl Into<String> + std::fmt::Display) {
    #[cfg(not(target_arch = "wasm32"))]
    log::warn!("{}", s);

    #[cfg(target_arch = "wasm32")]
    web_sys::console::warn_1(&s.to_string().into());
}

pub fn error(s: impl Into<String> + std::fmt::Display) {
    #[cfg(not(target_arch = "wasm32"))]
    log::error!("{}", s);

    #[cfg(target_arch = "wasm32")]
    web_sys::console::error_1(&s.to_string().into());
}
