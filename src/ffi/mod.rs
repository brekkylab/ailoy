#[cfg(feature = "nodejs")]
pub(crate) mod node;
#[cfg(feature = "python")]
pub(crate) mod py;
#[cfg(target_family = "wasm")]
pub(crate) mod web;

pub(crate) mod faiss_wrap;

#[cfg(feature = "python")]
pub use py::py_stub_info;
