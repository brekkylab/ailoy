use pyo3::{
    PyClass,
    exceptions::{PyNotImplementedError, PyRuntimeError},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen_derive::*;

use crate::{
    ffi::py::{
        base::{PyWrapper, await_future},
        cache_progress::await_cache_result,
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
        Py::new(py, (Self { inner }, PyEmbeddingModel {}))
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
    #[gen_stub(override_return_type(type_repr = "typing.Awaitable[LocalEmbeddingModel]"))]
    #[pyo3(signature = (model_name, progress_callback = None))]
    fn create<'a>(
        _cls: &Bound<'a, PyType>,
        py: Python<'a>,
        model_name: String,
        #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<Bound<'a, PyAny>> {
        let fut = async move {
            let inner =
                await_cache_result::<LocalEmbeddingModel>(model_name, progress_callback).await?;
            Python::attach(|py| Py::new(py, (PyLocalEmbeddingModel { inner }, PyEmbeddingModel {})))
        };
        pyo3_async_runtimes::tokio::future_into_py(py, fut)
    }

    #[classmethod]
    #[pyo3(signature = (model_name, progress_callback = None))]
    fn create_sync(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model_name: String,
        #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyLocalEmbeddingModel>> {
        let inner = await_future(await_cache_result::<LocalEmbeddingModel>(
            model_name,
            progress_callback,
        ))?;
        Py::new(py, (PyLocalEmbeddingModel { inner }, PyEmbeddingModel {}))
    }

    async fn run(&mut self, message: String) -> PyResult<Embedding> {
        self._run(message).await
    }

    fn run_sync(&mut self, message: String) -> PyResult<Embedding> {
        self._run_sync(message)
    }
}
