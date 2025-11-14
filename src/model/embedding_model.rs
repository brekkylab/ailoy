use std::collections::HashMap;

use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use futures::{Stream, StreamExt as _};

use crate::{
    cache::{Cache, CacheProgress},
    model::{get_cache_context, local::LocalEmbeddingModel},
    value::{Embedding, Value},
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
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct EmbeddingModel {
    inner: EmbeddingModelInner,
}

impl EmbeddingModel {
    pub async fn new_local(
        model_name: impl Into<String>,
        device_id: Option<i32>,
    ) -> anyhow::Result<Self> {
        let cache = crate::cache::Cache::new();
        let ctx = get_cache_context(device_id);
        let mut model_strm =
            Box::pin(cache.try_create::<LocalEmbeddingModel>(model_name, Some(ctx)));
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
        device_id: i32,
    ) -> impl Stream<Item = anyhow::Result<CacheProgress<Self>>> + 'static {
        let mut ctx = HashMap::new();
        ctx.insert("device_id".to_owned(), Value::integer(device_id.into()));
        let mut strm =
            Box::pin(Cache::new().try_create::<LocalEmbeddingModel>(model_name, Some(ctx)));
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
        #[pyo3(name = "new_local", signature = (model_name, device_id = None, progress_callback = None))]
        fn new_local_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            model_name: String,
            device_id: Option<i32>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'a, PyAny>> {
            let fut = async move {
                let ctx = get_cache_context(device_id);
                let inner = await_cache_result::<LocalEmbeddingModel>(
                    model_name,
                    Some(ctx),
                    progress_callback,
                )
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
        #[pyo3(name = "new_local_sync", signature = (model_name, device_id=None, progress_callback = None))]
        fn new_local_sync_py(
            _cls: &Bound<'_, PyType>,
            py: Python<'_>,
            model_name: String,
            device_id: Option<i32>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Py<Self>> {
            let ctx = get_cache_context(device_id);
            let inner = await_future(
                py,
                await_cache_result::<LocalEmbeddingModel>(model_name, Some(ctx), progress_callback),
            )?;
            Py::new(
                py,
                EmbeddingModel {
                    inner: EmbeddingModelInner::Local(inner),
                },
            )
        }

        #[pyo3(signature = (text))]
        async fn infer(&mut self, text: String) -> PyResult<Embedding> {
            match &mut self.inner {
                EmbeddingModelInner::Local(model) => model.infer(text).await,
            }
            .map_err(Into::into)
        }

        #[pyo3(signature = (text))]
        fn infer_sync(&mut self, py: Python<'_>, text: String) -> PyResult<Embedding> {
            let fut = match &mut self.inner {
                EmbeddingModelInner::Local(model) => model.infer(text),
            };
            await_future(py, fut).map_err(Into::into)
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use napi::{Status, threadsafe_function::ThreadsafeFunction};
    use napi_derive::napi;

    use super::*;
    use crate::ffi::node::cache::{JsCacheProgress, await_cache_result};

    #[derive(Default)]
    #[napi(object, object_to_js = false)]
    pub struct EmbeddingModelConfig {
        pub device_id: Option<i32>,
        pub progress_callback:
            Option<ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>>,
    }

    #[napi]
    impl EmbeddingModel {
        #[napi(js_name = "newLocal")]
        pub async fn new_local_js(
            model_name: String,
            config: Option<EmbeddingModelConfig>,
        ) -> napi::Result<EmbeddingModel> {
            let config = config.unwrap_or_default();
            let ctx = get_cache_context(config.device_id);
            let inner = await_cache_result::<LocalEmbeddingModel>(
                model_name,
                Some(ctx),
                config.progress_callback,
            )
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
    use js_sys::{Function, Reflect};
    use wasm_bindgen::prelude::*;

    use super::*;

    #[wasm_bindgen]
    extern "C" {
        #[derive(Clone)]
        #[wasm_bindgen(
            js_name = "EmbeddingModelConfig",
            typescript_type = "{deviceId?: number; progressCallback?: CacheProgressCallbackFn;}"
        )]
        pub type _EmbeddingModelConfig;
    }

    #[derive(Default)]
    pub struct EmbeddingModelConfig {
        pub device_id: Option<i32>,
        pub progress_callback: Option<Function>,
    }

    impl TryFrom<_EmbeddingModelConfig> for EmbeddingModelConfig {
        type Error = js_sys::Error;

        fn try_from(value: _EmbeddingModelConfig) -> Result<Self, Self::Error> {
            let obj = value.obj;
            let mut config = Self::default();

            if let Ok(val) = Reflect::get(&obj, &"deviceId".into())
                && let Some(f64) = val.as_f64()
            {
                let i32 = f64 as i32;
                config.device_id = Some(i32);
            }

            if let Ok(val) = Reflect::get(&obj, &"progressCallback".into())
                && val.is_function()
            {
                config.progress_callback = Some(val.into());
            }

            Ok(config)
        }
    }

    #[wasm_bindgen]
    impl EmbeddingModel {
        #[wasm_bindgen(js_name = "newLocal")]
        pub async fn new_local_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            config: Option<_EmbeddingModelConfig>,
        ) -> Result<Self, js_sys::Error> {
            let config: EmbeddingModelConfig = match config {
                Some(c) => c.try_into()?,
                None => EmbeddingModelConfig::default(),
            };
            let ctx = get_cache_context(config.device_id);
            let inner = crate::ffi::web::await_cache_result::<LocalEmbeddingModel>(
                model_name,
                Some(ctx),
                config.progress_callback,
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
