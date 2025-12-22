#[cfg(target_family = "wasm")]
pub mod js_bridge;
#[cfg(feature = "nodejs")]
pub(crate) mod node;
#[cfg(feature = "python")]
pub(crate) mod py;
#[cfg(target_family = "wasm")]
pub(crate) mod web;

// #[cfg(any(target_family = "unix", target_family = "windows"))]
// mod dlpack_wrap;
mod faiss_wrap;
pub use faiss_wrap::*;

#[cfg(feature = "python")]
pub mod stub_util {
    use pyo3_stub_gen::{Result, generate::StubInfo};

    pub fn py_stub_info() -> Result<StubInfo> {
        let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
        StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
    }
}
