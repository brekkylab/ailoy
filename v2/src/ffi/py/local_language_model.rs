use std::sync::Arc;

use async_channel as ach;
use futures::{StreamExt, stream::BoxStream};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopAsyncIteration},
    prelude::*,
    types::PyDict,
};
use tokio::runtime::Runtime;

use crate::{
    cache::{Cache, FromCacheProgress},
    model::LocalLanguageModel,
};

/// Rust -> Python 오류 변환
fn map_err<E: ToString>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass(name = "LanguageModel")]
pub struct PyLocalLanguageModel {
    inner: Arc<LocalLanguageModel>,
}

#[pymethods]
impl PyLocalLanguageModel {
    // Async version of creation
    #[staticmethod]
    pub fn create(
        py: Python<'_>,
        model_name: &str,
    ) -> PyResult<Py<PyLocalLanguageModelCreateAsyncIterator>> {
        let name = model_name.to_string();
        let (tx, rx) = ach::unbounded::<Result<FromCacheProgress<LocalLanguageModel>, String>>();

        pyo3_async_runtimes::tokio::get_runtime().spawn(async move {
            let mut strm = Box::pin(Cache::new().try_create::<LocalLanguageModel>(name));
            while let Some(item) = strm.next().await {
                // 채널 닫혔으면 그냥 종료
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Py::new(py, PyLocalLanguageModelCreateAsyncIterator { rx })
    }

    // Sync version of creation
    #[staticmethod]
    pub fn create_sync(
        py: Python<'_>,
        model_name: &str,
    ) -> PyResult<Py<PyLocalLanguageModelCreateIterator>> {
        // Rust stream
        let stream: BoxStream<'static, Result<FromCacheProgress<LocalLanguageModel>, String>> =
            Box::pin(Cache::new().try_create::<LocalLanguageModel>(model_name));

        // attach current-thread runtime
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(map_err)?;

        Py::new(py, PyLocalLanguageModelCreateIterator { rt, stream })
    }
}

#[pyclass(unsendable, name = "LocalLanguageModelCreateIterator")]
pub struct PyLocalLanguageModelCreateIterator {
    rt: Runtime,
    stream: BoxStream<'static, Result<FromCacheProgress<LocalLanguageModel>, String>>,
}

#[pymethods]
impl PyLocalLanguageModelCreateIterator {
    fn __iter__(self_: PyRef<'_, Self>) -> PyRef<'_, Self> {
        self_
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        // Pull one step
        let item = py.allow_threads(|| {
            self.rt
                .block_on(async { self.stream.as_mut().next().await })
        });

        match item {
            Some(Ok(progress)) => {
                let comment = progress.comment().to_string();
                let current = progress.current_task();
                let total = progress.total_task();
                let maybe_result = if current == total {
                    Some(Py::new(
                        py,
                        PyLocalLanguageModel {
                            inner: Arc::new(progress.take().unwrap()),
                        },
                    )?)
                } else {
                    None
                };

                let d = PyDict::new(py);
                d.set_item("comment", comment)?;
                d.set_item("current", current)?;
                d.set_item("total", total)?;
                d.set_item("result", maybe_result)?;
                Ok(Some(d.into()))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Ok(None), // StopIteration
        }
    }
}

struct ProgressInner {
    comment: String,
    current: usize,
    total: usize,
    result: Option<Arc<LocalLanguageModel>>,
}

impl<'py> IntoPyObject<'py> for ProgressInner {
    type Target = PyDict; // Python type
    type Output = Bound<'py, PyDict>; // Return pointer shape
    type Error = PyErr; // Error type

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Bound<'py, Self::Target>> {
        let d = PyDict::new(py);
        d.set_item("comment", self.comment)?;
        d.set_item("current", self.current)?;
        d.set_item("total", self.total)?;
        if let Some(model_inner) = self.result {
            let py_model: Py<PyLocalLanguageModel> =
                Py::new(py, PyLocalLanguageModel { inner: model_inner })?;
            d.set_item("result", py_model)?;
        } else {
            d.set_item("result", py.None())?;
        }
        Ok(d)
    }
}

#[pyclass(unsendable, name = "LocalLanguageModelCreateAsyncIterator")]
pub struct PyLocalLanguageModelCreateAsyncIterator {
    rx: ach::Receiver<Result<FromCacheProgress<LocalLanguageModel>, String>>,
}

#[pymethods]
impl PyLocalLanguageModelCreateAsyncIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// async iterator: returns awaitable
    fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();

        let fut = async move {
            let progress = rx
                .recv()
                .await
                .map_err(|_| PyStopAsyncIteration::new_err(()))?
                .map_err(PyRuntimeError::new_err)?;

            let comment = progress.comment().to_string();
            let current = progress.current_task();
            let total = progress.total_task();
            let result = if current == total {
                Some(Arc::new(progress.take().unwrap()))
            } else {
                None
            };

            Ok::<ProgressInner, PyErr>(ProgressInner {
                comment,
                current,
                total,
                result,
            })
        };

        let awaitable = pyo3_async_runtimes::tokio::future_into_py(py, fut)?;
        Ok(awaitable.unbind())
    }
}
