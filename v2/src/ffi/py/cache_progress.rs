use std::collections::HashSet;

use futures::{StreamExt as _, stream::BoxStream};
use pyo3::{
    exceptions::{PyRuntimeError, PyStopAsyncIteration, PyStopIteration},
    prelude::*,
};
use pyo3_stub_gen::{PyStubType, TypeInfo, derive::*};
use tokio::runtime::Runtime;

use crate::{
    cache::{Cache, CacheProgress, TryFromCache},
    ffi::py::base::PyWrapper,
};

#[pyclass(subclass)]
pub struct GenericClass {}

impl PyStubType for GenericClass {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: format!("typing.Generic[CacheResultT]"),
            import: HashSet::new(),
        }
    }
}

pub struct CacheResultT(pub Py<PyAny>);

impl<'py> IntoPyObject<'py> for &CacheResultT {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.bind(py).clone())
    }
}

impl PyStubType for CacheResultT {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: format!("CacheResultT"),
            import: HashSet::new(),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(name = "CacheProgress", extends = GenericClass)]
pub struct PyCacheProgress {
    #[pyo3(get)]
    pub comment: String,

    #[pyo3(get)]
    pub current: usize,

    #[pyo3(get)]
    pub total: usize,

    #[pyo3(get)]
    pub result: Option<CacheResultT>,
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "CacheProgressSyncIterator", extends = GenericClass)]
pub struct PyCacheProgressSyncIterator {
    rt: Runtime,

    strm: BoxStream<'static, Result<PyCacheProgress, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCacheProgressSyncIterator {
    #[gen_stub(override_return_type(type_repr="CacheProgressSyncIterator[CacheResultT]", imports=("typing")))]
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr="CacheProgress[CacheResultT]", imports=("typing")))]
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyCacheProgress>> {
        let item = py.detach(|| self.rt.block_on(async { self.strm.next().await }));
        match item {
            Some(Ok(evt)) => {
                let initializer = PyClassInitializer::from(GenericClass {}).add_subclass(evt);
                let obj: Py<PyCacheProgress> = Py::new(py, initializer)?;
                Ok(obj)
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e)),
            None => Err(PyStopIteration::new_err(())), // StopIteration
        }
    }
}

pub fn create_cache_progress_sync_iterator<T>(
    model_key: impl Into<String>,
    py: Python<'_>,
) -> PyResult<Py<PyCacheProgressSyncIterator>>
where
    T: PyWrapper,
    T::Inner: TryFromCache,
{
    // attach current-thread runtime
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    // Rust stream
    let strm: BoxStream<'static, Result<CacheProgress<T::Inner>, String>> =
        Box::pin(Cache::new().try_create::<T::Inner>(model_key));

    // Map CacheProgress<T> -> PyCacheProgressIterator
    let strm = strm
        .then(|item| async move {
            match item {
                Ok(progress) => {
                    let comment = progress.comment.to_string();
                    let current = progress.current_task;
                    let total = progress.total_task;
                    let result = if current == total {
                        let result_inner = progress.result.expect("last event must carry result");
                        let obj: Py<PyAny> = Python::attach(|py| {
                            T::into_py_obj(result_inner, py).map(|o| o.into_any())
                        })
                        .map_err(|e| e.to_string())?;
                        Some(obj)
                    } else {
                        None
                    };
                    Ok(PyCacheProgress {
                        comment,
                        current,
                        total,
                        result: if let Some(result) = result {
                            Some(CacheResultT(result))
                        } else {
                            None
                        },
                    })
                }
                Err(e) => Err(e),
            }
        })
        .boxed();

    let iter = PyCacheProgressSyncIterator { rt, strm };
    let initializer = PyClassInitializer::from(GenericClass {}).add_subclass(iter);
    let obj: Py<PyCacheProgressSyncIterator> = Py::new(py, initializer)?;
    Ok(obj)
}

#[gen_stub_pyclass]
#[pyclass(unsendable, name = "CacheProgressIterator", extends = GenericClass)]
pub struct PyCacheProgressIterator {
    rx: async_channel::Receiver<Result<PyCacheProgress, String>>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCacheProgressIterator {
    #[gen_stub(override_return_type(type_repr="CacheProgressIterator[CacheResultT]", imports=("typing")))]
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr="typing.Awaitable[CacheProgress[CacheResultT]]", imports=("typing")))]
    fn __anext__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let rx = self.rx.clone();
        let fut = async move {
            match rx.recv().await {
                Ok(Ok(evt)) => Python::attach(|py| {
                    let initializer = PyClassInitializer::from(GenericClass {}).add_subclass(evt);
                    let obj: Py<PyCacheProgress> = Py::new(py, initializer)?;
                    Ok(obj)
                }),
                Ok(Err(e)) => Err(PyRuntimeError::new_err(e)),
                Err(_) => Err(PyStopAsyncIteration::new_err(())),
            }
        };
        let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
        Ok(py_fut.into())
    }
}

pub fn create_cache_progress_iterator<T>(
    cache_key: impl Into<String>,
    py: Python<'_>,
) -> PyResult<Py<PyCacheProgressIterator>>
where
    T: PyWrapper,
    T::Inner: TryFromCache,
{
    let rt = pyo3_async_runtimes::tokio::get_runtime();
    let (tx, rx) = async_channel::unbounded::<Result<PyCacheProgress, String>>();
    let cache_key = cache_key.into();

    rt.spawn(async move {
        let mut strm: BoxStream<'static, Result<CacheProgress<T::Inner>, String>> =
            Box::pin(Cache::new().try_create::<T::Inner>(cache_key));

        while let Some(item) = strm.next().await {
            let out = match item {
                Ok(mut progress) => {
                    let comment = progress.comment.to_string();
                    let current = progress.current_task;
                    let total = progress.total_task;
                    let result = if current == total {
                        match progress.result.take() {
                            Some(inner) => {
                                // T::Inner -> Py<PyAny>
                                match Python::attach(|py| {
                                    T::into_py_obj(inner, py).map(|o| o.into_any())
                                }) {
                                    Ok(obj) => Some(obj),
                                    Err(e) => {
                                        let _ = tx.send(Err(e.to_string())).await;
                                        continue;
                                    }
                                }
                            }
                            None => None,
                        }
                    } else {
                        None
                    };

                    Ok(PyCacheProgress {
                        comment,
                        current,
                        total,
                        result: if let Some(result) = result {
                            Some(CacheResultT(result))
                        } else {
                            None
                        },
                    })
                }
                Err(e) => Err(e),
            };

            if tx.send(out).await.is_err() {
                break; // Exit if consumer vanished
            }
            // Add a yield point to allow other tasks to run
            tokio::task::yield_now().await;
        }
    });

    let iter = PyCacheProgressIterator { rx };
    let initializer = PyClassInitializer::from(GenericClass {}).add_subclass(iter);
    let obj: Py<PyCacheProgressIterator> = Py::new(py, initializer)?;
    Ok(obj)
}
