use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use futures::{Stream, StreamExt as _};

use crate::{
    cache::{Cache, CacheProgress},
    model::local::LocalEmbeddingModel,
    vector_store::Embedding,
};

#[maybe_send_sync]
#[multi_platform_async_trait]
pub trait EmbeddingModelInference {
    async fn infer(self: &Self, text: String) -> anyhow::Result<Embedding>;
}

#[derive(Debug, Clone)]
enum EmbeddingModelInner {
    Local(LocalEmbeddingModel),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct EmbeddingModel {
    inner: EmbeddingModelInner,
}

impl EmbeddingModel {
    pub async fn new_local(model_name: impl Into<String>) -> anyhow::Result<Self> {
        let cache = crate::cache::Cache::new();
        let mut model_strm = Box::pin(cache.try_create::<LocalEmbeddingModel>(model_name));
        let mut model: Option<LocalEmbeddingModel> = None;
        while let Some(progress) = model_strm.next().await {
            let mut progress = progress.unwrap();
            if progress.current_task == progress.total_task {
                model = progress.result.take();
            }
        }
        Ok(Self {
            inner: EmbeddingModelInner::Local(model.unwrap()),
        })
    }

    pub async fn try_new_local(
        model_name: impl Into<String>,
    ) -> impl Stream<Item = anyhow::Result<CacheProgress<Self>>> + 'static {
        let model_name = model_name.into();
        let mut strm = Box::pin(Cache::new().try_create::<LocalEmbeddingModel>(model_name));
        async_stream::try_stream! {
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.current_task,
                    result: result.result.map(|v| EmbeddingModel{inner: EmbeddingModelInner::Local(v)}),
                };
            }
        }
    }
}

#[multi_platform_async_trait]
impl EmbeddingModelInference for EmbeddingModel {
    async fn infer(&self, text: String) -> anyhow::Result<Embedding> {
        match &self.inner {
            EmbeddingModelInner::Local(model) => model.infer(text).await,
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{Bound, Py, PyAny, PyResult, Python, pymethods, types::PyType};
    use pyo3_stub_gen_derive::*;

    use super::*;
    use crate::ffi::py::{base::await_future, cache_progress::await_cache_result};

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
                Python::attach(|py| {
                    Py::new(
                        py,
                        EmbeddingModel {
                            inner: EmbeddingModelInner::Local(inner),
                        },
                    )
                })
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
            Py::new(
                py,
                EmbeddingModel {
                    inner: EmbeddingModelInner::Local(inner),
                },
            )
        }

        #[pyo3(signature = (text))]
        async fn run(&mut self, text: String) -> PyResult<Embedding> {
            match &mut self.inner {
                EmbeddingModelInner::Local(model) => model.infer(text).await,
            }
            .map_err(Into::into)
        }

        #[pyo3(signature = (text))]
        fn run_sync(&mut self, text: String) -> PyResult<Embedding> {
            let fut = match &mut self.inner {
                EmbeddingModelInner::Local(model) => model.infer(text),
            };
            await_future(fut).map_err(Into::into)
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Status, threadsafe_function::ThreadsafeFunction};
    use napi_derive::napi;

    use super::*;
    use crate::ffi::node::cache::{JsCacheProgress, await_cache_result};

    #[napi]
    impl EmbeddingModel {
        #[napi(js_name = "newLocal")]
        pub async fn new_local_js(
            model_name: String,
            progress_callback: Option<
                ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>,
            >,
        ) -> napi::Result<EmbeddingModel> {
            let inner = await_cache_result::<LocalEmbeddingModel>(model_name, progress_callback)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))?;
            Ok(EmbeddingModel {
                inner: EmbeddingModelInner::Local(inner),
            })
        }

        #[napi(js_name = "infer")]
        pub async fn infer_js(&self, text: String) -> napi::Result<Embedding> {
            self.infer(text)
                .await
                .map_err(|e| napi::Error::new(Status::GenericFailure, e.to_string()))
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::ffi::web::CacheProgressCallbackFn;

    #[wasm_bindgen]
    impl EmbeddingModel {
        #[wasm_bindgen(js_name = "newLocal")]
        pub async fn new_local_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "progressCallback")] progress_callback: Option<
                CacheProgressCallbackFn,
            >,
        ) -> Result<Self, js_sys::Error> {
            let inner = crate::ffi::web::await_cache_result::<LocalEmbeddingModel>(
                model_name,
                progress_callback,
            )
            .await
            .map_err(|e| js_sys::Error::new(&e.to_string()))?;
            Ok(Self {
                inner: EmbeddingModelInner::Local(inner),
            })
        }

        #[wasm_bindgen(js_name = infer)]
        pub async fn infer_js(&mut self, text: String) -> Result<Embedding, js_sys::Error> {
            self.infer(text)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }
    }
}
