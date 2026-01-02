use ailoy_macros::{maybe_send_sync, multi_platform_async_trait};
use futures::StreamExt as _;

use crate::{
    cache::CacheProgress,
    model::local::{LocalEmbeddingModel, LocalEmbeddingModelConfig},
    utils::BoxStream,
    value::Embedding,
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
    pub async fn try_new_local(
        model_name: impl Into<String>,
        config: Option<LocalEmbeddingModelConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: EmbeddingModelInner::Local(
                LocalEmbeddingModel::try_new(model_name, config).await?,
            ),
        })
    }

    pub async fn try_new_local_stream<'a>(
        model_name: impl Into<String>,
        config: Option<LocalEmbeddingModelConfig>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<Self>>> {
        let model_name = model_name.into();
        Box::pin(async_stream::try_stream! {
            let mut strm = LocalEmbeddingModel::try_new_stream(model_name, config);
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.total_task,
                    result: result.result.map(|v| EmbeddingModel{inner: EmbeddingModelInner::Local(v)}),
                };
            }
        })
    }

    pub fn download<'a>(
        model: impl Into<String>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<()>>> {
        LocalEmbeddingModel::download(model)
    }

    pub async fn remove(model: impl Into<String>) -> anyhow::Result<()> {
        LocalEmbeddingModel::remove(model).await
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
        #[pyo3(name = "new_local", signature = (model_name, device_id = None, validate_checksum = None, progress_callback = None))]
        fn new_local_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            model_name: String,
            device_id: Option<i32>,
            validate_checksum: Option<bool>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'a, PyAny>> {
            let config = LocalEmbeddingModelConfig {
                device_id,
                validate_checksum,
            };
            let cache_strm = LocalEmbeddingModel::try_new_stream(model_name, Some(config));
            let fut = async move {
                let inner = await_cache_result(cache_strm, progress_callback).await?;
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
        #[pyo3(name = "new_local_sync", signature = (model_name, device_id=None, validate_checksum = None, progress_callback = None))]
        fn new_local_sync_py(
            _cls: &Bound<'_, PyType>,
            py: Python<'_>,
            model_name: String,
            device_id: Option<i32>,
            validate_checksum: Option<bool>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Py<Self>> {
            let config = LocalEmbeddingModelConfig {
                device_id,
                validate_checksum,
            };
            let cache_strm = LocalEmbeddingModel::try_new_stream(model_name, Some(config));
            let inner = await_future(py, await_cache_result(cache_strm, progress_callback))?;
            Py::new(
                py,
                EmbeddingModel {
                    inner: EmbeddingModelInner::Local(inner),
                },
            )
        }

        #[classmethod]
        #[pyo3(name = "download", signature = (model_name, progress_callback = None))]
        fn download_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'_>,
            model_name: String,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<()> {
            let strm = Self::download(model_name);
            await_future(py, await_cache_result(strm, progress_callback))?;
            Ok(())
        }

        #[classmethod]
        #[pyo3(name = "remove", signature = (model_name))]
        fn remove_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'_>,
            model_name: String,
        ) -> PyResult<()> {
            await_future(py, Self::remove(model_name))
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
    #[napi(js_name = "LocalEmbeddingModelConfig", object, object_to_js = false)]
    pub struct JSLocalEmbeddingModelConfig {
        pub device_id: Option<i32>,
        pub validate_checksum: Option<bool>,
        pub progress_callback:
            Option<ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>>,
    }

    #[napi]
    impl EmbeddingModel {
        #[napi(js_name = "newLocal")]
        pub async fn new_local_js(
            model_name: String,
            config: Option<JSLocalEmbeddingModelConfig>,
        ) -> napi::Result<EmbeddingModel> {
            let config = config.unwrap_or_default();
            let cache_strm = LocalEmbeddingModel::try_new_stream(
                model_name,
                Some(LocalEmbeddingModelConfig {
                    device_id: config.device_id,
                    validate_checksum: config.validate_checksum,
                }),
            );
            let inner = await_cache_result(cache_strm, config.progress_callback)
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

        #[napi(js_name = "download")]
        pub async fn download_js(
            model_name: String,
            config: Option<JSLocalEmbeddingModelConfig>,
        ) -> napi::Result<()> {
            let config = config.unwrap_or_default();
            let strm = Self::download(model_name);
            let _ = await_cache_result(strm, config.progress_callback).await;
            Ok(())
        }

        #[napi(js_name = "remove")]
        pub async fn remove_js(model_name: String) -> napi::Result<()> {
            Self::remove(model_name)
                .await
                .map_err(|e| napi::Error::from_reason(e.to_string()))
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use js_sys::{Function, Reflect};
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::ffi::web::cache::await_cache_result;

    #[wasm_bindgen]
    extern "C" {
        #[derive(Clone)]
        #[wasm_bindgen(
            js_name = "LocalEmbeddingModelConfig",
            typescript_type = "{deviceId?: number; validateChecksum?: boolean; progressCallback?: CacheProgressCallbackFn;}"
        )]
        pub type _JSLocalEmbeddingModelConfig;
    }

    #[derive(Default)]
    pub struct JSLocalEmbeddingModelConfig {
        pub device_id: Option<i32>,
        pub validate_checksum: Option<bool>,
        pub progress_callback: Option<Function>,
    }

    impl TryFrom<_JSLocalEmbeddingModelConfig> for JSLocalEmbeddingModelConfig {
        type Error = js_sys::Error;

        fn try_from(value: _JSLocalEmbeddingModelConfig) -> Result<Self, Self::Error> {
            let obj = value.obj;
            let mut config = Self::default();

            if let Ok(val) = Reflect::get(&obj, &"deviceId".into())
                && let Some(f64) = val.as_f64()
            {
                let i32 = f64 as i32;
                config.device_id = Some(i32);
            }

            if let Ok(val) = Reflect::get(&obj, &"validateChecksum".into())
                && let Some(b) = val.as_bool()
            {
                config.validate_checksum = Some(b);
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
            config: Option<_JSLocalEmbeddingModelConfig>,
        ) -> Result<Self, js_sys::Error> {
            let config: JSLocalEmbeddingModelConfig = match config {
                Some(c) => c.try_into()?,
                None => JSLocalEmbeddingModelConfig::default(),
            };
            let cache_strm = LocalEmbeddingModel::try_new_stream(
                model_name,
                Some(LocalEmbeddingModelConfig {
                    device_id: config.device_id,
                    validate_checksum: config.validate_checksum,
                }),
            );
            let inner = await_cache_result(cache_strm, config.progress_callback)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))?;
            Ok(Self {
                inner: EmbeddingModelInner::Local(inner),
            })
        }

        #[wasm_bindgen(js_name = "download")]
        pub async fn download_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "config")] config: Option<_JSLocalEmbeddingModelConfig>,
        ) -> Result<(), js_sys::Error> {
            let config: JSLocalEmbeddingModelConfig = match config {
                Some(c) => c.try_into()?,
                None => JSLocalEmbeddingModelConfig::default(),
            };
            let strm = Self::download(model_name);
            let _ = await_cache_result(strm, config.progress_callback).await;
            Ok(())
        }

        #[wasm_bindgen(js_name = "remove")]
        pub async fn remove_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
        ) -> Result<(), js_sys::Error> {
            Self::remove(model_name)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }

        #[wasm_bindgen(js_name = infer)]
        pub async fn infer_js(&mut self, text: String) -> Result<Embedding, js_sys::Error> {
            self.infer(text)
                .await
                .map_err(|e| js_sys::Error::new(&e.to_string()))
        }
    }
}
