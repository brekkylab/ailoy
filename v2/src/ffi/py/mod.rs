mod model;
mod value;

use model::*;
use value::*;

use pyo3::prelude::*;

#[pymodule(name = "_core")]
fn ailoy_py(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyPart>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyLocalLanguageModel>()?;
    m.add_class::<PyLocalLanguageModelCreateIterator>()?;
    m.add_class::<PyLocalLanguageModelCreateAsyncIterator>()?;
    Ok(())
}
