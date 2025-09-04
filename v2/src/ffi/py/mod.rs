mod base;
mod cache_progress;
mod model;
mod value;

use cache_progress::{
    PyCacheProgress as CacheProgress, PyCacheProgressIterator as CacheProgressIterator,
    PyCacheProgressSyncIterator as CacheProgressSyncIterator,
};
use model::{
    PyLanguageModelRunIterator as LanguageModelRunIterator,
    PyLanguageModelRunSyncIterator as LanguageModelRunSyncIterator,
    PyLocalLanguageModel as LocalLanguageModel,
};
use pyo3::prelude::*;
use pyo3_stub_gen::{Result, generate::StubInfo};

use crate::value::{FinishReason, Message, MessageAggregator, MessageOutput, Part, Role, ToolDesc};

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    // Add classes in alphabetical order
    m.add_class::<CacheProgress>()?;
    m.add_class::<CacheProgressIterator>()?;
    m.add_class::<CacheProgressSyncIterator>()?;
    m.add_class::<FinishReason>()?;
    m.add_class::<LanguageModelRunIterator>()?;
    m.add_class::<LanguageModelRunSyncIterator>()?;
    m.add_class::<LocalLanguageModel>()?;
    m.add_class::<Message>()?;
    m.add_class::<MessageAggregator>()?;
    m.add_class::<MessageOutput>()?;
    m.add_class::<Part>()?;
    m.add_class::<Role>()?;
    m.add_class::<ToolDesc>()?;
    Ok(())
}

pub fn stub_info() -> Result<StubInfo> {
    let manifest_dir: &::std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("bindings/python/pyproject.toml"))
}
