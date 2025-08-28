mod base;
mod cache_progress;
mod model;
mod value;

use cache_progress::*;
use model::*;
use value::*;

use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPart>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyMessageOutput>()?;
    m.add_class::<PyCacheProgress>()?;
    m.add_class::<PyCacheProgressIterator>()?;
    m.add_class::<PyCacheProgressSyncIterator>()?;
    m.add_class::<PyLocalLanguageModel>()?;
    m.add_class::<PyAgentRunIterator>()?;
    m.add_class::<PyAgentRunSyncIterator>()?;
    m.add_class::<PyAgentRunIterator>()?;
    Ok(())
}

pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
