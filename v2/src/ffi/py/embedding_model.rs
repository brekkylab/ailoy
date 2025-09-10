use pyo3::{
    PyClass,
    exceptions::{PyNotImplementedError, PyRuntimeError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen_derive::*;

use crate::{
    ffi::py::{
        base::PyWrapper,
        cache_progress::{
            PyCacheProgressIterator, PyCacheProgressSyncIterator, create_cache_progress_iterator,
            create_cache_progress_sync_iterator,
        },
    },
    model::{EmbeddingModel, LocalEmbeddingModel},
};

type Embedding = Vec<f32>;

pub trait PyEmbeddingModelMethods<T: EmbeddingModel + 'static>:
    PyClass<BaseType = PyEmbeddingModel>
{
    fn inner(&mut self) -> &mut T;

    async fn _run(&mut self, message: String) -> PyResult<Embedding> {
        self.inner()
            .run(message)
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn _run_sync(&mut self, message: String) -> PyResult<Embedding> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        rt.block_on(self._run(message))
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "EmbeddingModel", subclass)]
pub struct PyEmbeddingModel {}

#[gen_stub_pymethods]
#[pymethods]
impl PyEmbeddingModel {
    #[allow(unused_variables)]
    async fn run(&mut self, message: String) -> PyResult<Embedding> {
        Err(PyNotImplementedError::new_err(
            "Subclass of EmbeddingModel must implement 'run'",
        ))
    }

    #[allow(unused_variables)]
    fn run_sync(&mut self, message: String) -> PyResult<Embedding> {
        Err(PyNotImplementedError::new_err(
            "Subclass of EmbeddingModel must implement 'run_sync'",
        ))
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "LocalEmbeddingModel", extends = PyEmbeddingModel)]
pub struct PyLocalEmbeddingModel {
    inner: LocalEmbeddingModel,
}

impl PyWrapper for PyLocalEmbeddingModel {
    type Inner = LocalEmbeddingModel;

    fn into_inner(&self) -> PyResult<Self::Inner> {
        Ok(self.inner.clone())
    }

    fn into_py_obj(inner: Self::Inner, py: Python<'_>) -> PyResult<Py<Self>> {
        let base = PyEmbeddingModel {};
        let child = Self { inner };
        let py_obj = Py::new(py, (child, base))?;
        Ok(py_obj)
    }
}

impl PyEmbeddingModelMethods<LocalEmbeddingModel> for PyLocalEmbeddingModel {
    fn inner(&mut self) -> &mut LocalEmbeddingModel {
        &mut self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalEmbeddingModel {
    #[classmethod]
    #[gen_stub(override_return_type(type_repr = "CacheProgressIterator[LocalEmbeddingModel]"))]
    fn create(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model_name: String,
    ) -> PyResult<Py<PyCacheProgressIterator>> {
        create_cache_progress_iterator::<PyLocalEmbeddingModel>(model_name, py)
    }

    #[classmethod]
    #[gen_stub(override_return_type(type_repr = "CacheProgressSyncIterator[LocalEmbeddingModel]"))]
    fn create_sync(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model_name: String,
    ) -> PyResult<Py<PyCacheProgressSyncIterator>> {
        create_cache_progress_sync_iterator::<PyLocalEmbeddingModel>(model_name, py)
    }

    async fn run(&mut self, message: String) -> PyResult<Embedding> {
        self._run(message).await
    }

    fn run_sync(&mut self, message: String) -> PyResult<Embedding> {
        self._run_sync(message)
    }
}
