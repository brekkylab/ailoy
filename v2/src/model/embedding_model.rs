use ailoy_macros::multi_platform_async_trait;
use anyhow::Result;
use futures::{Stream, StreamExt as _};

use crate::{
    cache::{Cache, CacheProgress},
    knowledge_base::Embedding,
    model::local::LocalEmbeddingModel,
    utils::{MaybeSend, MaybeSync},
};

#[multi_platform_async_trait]
pub trait EmbeddingModelInference: MaybeSend + MaybeSync {
    async fn infer(self: &mut Self, text: String) -> Result<Embedding>;
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub struct EmbeddingModel {
    inner: LocalEmbeddingModel,
}

impl EmbeddingModel {
    pub async fn try_new(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = Result<CacheProgress<Self>, String>> + 'static {
        let model_name = model_name.into();
        let mut strm = Box::pin(Cache::new().try_create::<LocalEmbeddingModel>(model_name));
        async_stream::try_stream! {
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.current_task,
                    result: result.result.map(|v| EmbeddingModel{inner: v}),
                };
            }
        }
    }
}

#[multi_platform_async_trait]
impl EmbeddingModelInference for EmbeddingModel {
    async fn infer(self: &mut Self, text: String) -> Result<Embedding> {
        self.inner.infer(text).await
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, Py, PyAny, PyResult, Python, exceptions::PyRuntimeError, pymethods, types::PyType,
    };
    use pyo3_stub_gen_derive::*;

    use crate::ffi::py::{base::await_future, cache_progress::await_cache_result};

    use super::*;

    #[gen_stub_pymethods]
    #[pymethods]
    impl EmbeddingModel {
        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[EmbeddingModel]"))]
        #[pyo3(name = "CreateLocal", signature = (model_name, progress_callback = None))]
        fn create_local<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            model_name: String,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'a, PyAny>> {
            let fut = async move {
                let inner =
                    await_cache_result::<LocalEmbeddingModel>(model_name, progress_callback)
                        .await?;
                Python::attach(|py| Py::new(py, EmbeddingModel { inner }))
            };
            pyo3_async_runtimes::tokio::future_into_py(py, fut)
        }

        #[classmethod]
        #[pyo3(name = "CreateLocalSync", signature = (model_name, progress_callback = None))]
        fn create_local_sync(
            _cls: &Bound<'_, PyType>,
            py: Python<'_>,
            model_name: String,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Py<Self>> {
            let inner = await_future(await_cache_result::<LocalEmbeddingModel>(
                model_name,
                progress_callback,
            ))?;
            Py::new(py, EmbeddingModel { inner })
        }

        #[pyo3(signature = (text))]
        async fn run(&mut self, text: String) -> PyResult<Embedding> {
            self.inner
                .infer(text)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }

        #[pyo3(signature = (text))]
        fn run_sync(&mut self, text: String) -> PyResult<Embedding> {
            await_future(self.inner.infer(text)).map_err(|e| PyRuntimeError::new_err(e.to_string()))
        }
    }
}
