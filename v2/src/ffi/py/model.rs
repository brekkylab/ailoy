use futures::StreamExt;
use pyo3::{
    exceptions::{PyRuntimeError, PyStopAsyncIteration, PyStopIteration},
    prelude::*,
    types::PyType,
};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

use crate::{
    ffi::py::{
        base::PyWrapper,
        cache_progress::{
            PyCacheProgressIterator, PyCacheProgressSyncIterator, create_cache_progress_iterator,
            create_cache_progress_sync_iterator,
        },
    },
    model::{LanguageModel, LocalLanguageModel},
    value::{Message, MessageOutput},
};

#[gen_stub_pyclass]
#[pyclass(name = "LocalLanguageModel")]
pub struct PyLocalLanguageModel {
    inner: Option<LocalLanguageModel>,
}

impl PyWrapper for PyLocalLanguageModel {
    type Inner = LocalLanguageModel;

    fn from_inner(inner: Self::Inner) -> Self {
        Self { inner: Some(inner) }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLocalLanguageModel {
    // Async version of creation
    #[classmethod]
    #[gen_stub(override_return_type(type_repr="CacheProgressIterator[LocalLanguageModel]", imports=("typing")))]
    pub fn create(
        _cls: &Bound<'_, PyType>,
        py: Python,
        model_name: &str,
    ) -> PyResult<Py<PyCacheProgressIterator>> {
        create_cache_progress_iterator::<PyLocalLanguageModel>(model_name.to_string(), py)
    }

    // Sync version of creation
    #[classmethod]
    #[gen_stub(override_return_type(type_repr="CacheProgressSyncIterator[LocalLanguageModel]", imports=("typing")))]
    pub fn create_sync(
        _cls: &Bound<'_, PyType>,
        py: Python,
        model_name: &str,
    ) -> PyResult<Py<PyCacheProgressSyncIterator>> {
        create_cache_progress_sync_iterator::<PyLocalLanguageModel>(model_name, py)
    }

    pub fn run(
        mut self_: PyRefMut<'_, Self>,
        messages: Vec<Message>,
    ) -> PyResult<PyLanguageModelRunIterator> {
        let model = self_
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Model already consumed"))?;

        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let mut model = model;
            let mut strm = model.run(messages, Vec::new()).boxed();

            while let Some(item) = strm.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });
        Ok(PyLanguageModelRunIterator { rx })
    }

    pub fn run_sync(
        mut self_: PyRefMut<'_, Self>,
        messages: Vec<Message>,
    ) -> PyResult<PyLanguageModelRunSyncIterator> {
        let model = self_
            .inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Model already consumed"))?;

        let (tx, rx) = tokio::sync::mpsc::channel::<Result<MessageOutput, String>>(16);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        rt.spawn(async move {
            let mut model = model;
            let mut stream = model.run(messages, Vec::new()).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Ok(PyLanguageModelRunSyncIterator { rt, rx })
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "LanguageModelRunIterator")]
pub struct PyLanguageModelRunIterator {
    rx: async_channel::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModelRunIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr="typing.Awaitable[MessageOutput]", imports=("typing")))]
    fn __anext__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let fut = async move {
            match rx.recv().await {
                Ok(Ok(evt)) => Ok(evt),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
                Err(_) => Err(PyStopAsyncIteration::new_err(())),
            }
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "LanguageModelRunSyncIterator")]
pub struct PyLanguageModelRunSyncIterator {
    rt: Runtime,
    rx: tokio::sync::mpsc::Receiver<Result<MessageOutput, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLanguageModelRunSyncIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
        let item = py.allow_threads(|| self.rt.block_on(self.rx.recv()));
        match item {
            Some(Ok(evt)) => Ok(evt),
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Err(PyStopIteration::new_err(())), // StopIteration
        }
    }
}
