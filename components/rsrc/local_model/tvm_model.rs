use std::path::PathBuf;

fn get_current_arch() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "wasm32") {
        "wasm32"
    } else {
        "unknown"
    }
}

fn get_current_os() -> &'static str {
    if cfg!(target_os = "windows") {
        "Windows"
    } else if cfg!(target_os = "linux") {
        "Linux"
    } else if cfg!(target_os = "macos") {
        "Darwin"
    } else {
        "unknown"
    }
}

fn get_current_accelerator() -> &'static str {
    if cfg!(target_os = "windows") {
        "vulkan"
    } else if cfg!(target_os = "linux") {
        "vulkan"
    } else if cfg!(target_os = "macos") {
        "metal"
    } else {
        "webgpu"
    }
}

pub struct TVMModel {
    model_name: String,
}

impl TVMModel {
    pub fn new(model_name: String) -> Self {
        TVMModel { model_name }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn test_tvm_model() {
        let model = TVMModel::new(String::from("Qwen/Qwen3-0.6B"));
    }
}
