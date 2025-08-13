mod base;
mod cache_progress;
mod model;
mod value;

use cache_progress::*;
use model::*;
use value::*;

use pyo3::prelude::*;

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPart>()?;
    m.add_class::<PyMessageDelta>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyCacheProgress>()?;
    m.add_class::<PyCacheProgressIterator>()?;
    m.add_class::<PyCacheProgressSyncIterator>()?;
    m.add_class::<PyLocalLanguageModel>()?;
    m.add_class::<PyAgentRunSyncIterator>()?;
    Ok(())
}
