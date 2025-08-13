use std::sync::Arc;

use pyo3::prelude::*;

use crate::{
    ffi::py::{
        base::PyWrapper,
        cache_progress::{
            PyCacheProgressIterator, PyCacheProgressSyncIterator, create_cache_progress_iterator,
            create_cache_progress_sync_iterator,
        },
        value::PyMessage,
    },
    model::{LanguageModel, LocalLanguageModel},
};

#[pyclass(name = "LocalLanguageModel")]
pub struct PyLocalLanguageModel {
    inner: Arc<LocalLanguageModel>,
}

impl PyWrapper for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn from_inner(inner: Self::Inner) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

#[pymethods]
impl PyLocalLanguageModel {
    // Async version of creation
    #[staticmethod]
    pub fn create(model_name: &str) -> PyCacheProgressIterator {
        create_cache_progress_iterator::<PyLocalLanguageModel>(model_name.to_string())
    }

    // Sync version of creation
    #[staticmethod]
    pub fn create_sync(model_name: &str) -> PyResult<PyCacheProgressSyncIterator> {
        create_cache_progress_sync_iterator::<PyLocalLanguageModel>(model_name)
    }

    pub fn run(&mut self, messages: Vec<PyMessage>) -> PyResult<()> {
        let messages = messages.into_iter().map(|m| m.inner).collect::<Vec<_>>();
        let strm = self.inner.clone().run(Vec::new(), messages);
        Ok(())
    }
}
