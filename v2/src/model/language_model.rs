use std::sync::Arc;

use ailoy_macros::maybe_send_sync;
use futures::StreamExt as _;
use serde::Serialize;

use crate::{
    cache::CacheProgress,
    model::{
        LocalLangModel, StreamAPILangModel,
        api::APISpecification,
        custom::{CustomLangModel, CustomLangModelInferFunc},
    },
    utils::BoxStream,
    value::{Message, MessageOutput, ToolDesc},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum))]
pub enum ThinkEffort {
    #[default]
    Disable,
    Enable,
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen::derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum))]
pub enum Grammar {
    Plain(),
    JSON(),
    JSONSchema(String),
    Regex(String),
    CFG(String),
}

impl Default for Grammar {
    fn default() -> Self {
        Self::Plain()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
pub struct InferenceConfig {
    pub think_effort: ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,

    pub grammar: Grammar,
}

#[maybe_send_sync]
pub trait LangModelInference {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>>;
}

#[derive(Clone)]
enum LangModelInner {
    Local(LocalLangModel),
    StreamAPI(StreamAPILangModel),
    Custom(CustomLangModel),
}

#[derive(Clone)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
pub struct LangModel {
    inner: LangModelInner,
}

impl LangModel {
    pub async fn try_new_local(model_name: impl Into<String>) -> anyhow::Result<Self> {
        Ok(Self {
            inner: LangModelInner::Local(LocalLangModel::try_new(model_name).await?),
        })
    }

    pub fn try_new_local_stream<'a>(
        model_name: impl Into<String>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<Self>>> {
        let model_name = model_name.into();
        Box::pin(async_stream::try_stream! {
            let mut strm = LocalLangModel::try_new_stream(model_name);
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.total_task,
                    result: result.result.map(|v| LangModel{inner: LangModelInner::Local(v)}),
                };
            }
        })
    }

    pub fn new_stream_api(
        spec: APISpecification,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            inner: LangModelInner::StreamAPI(StreamAPILangModel::new(spec, model, api_key)),
        }
    }

    pub fn new_custom(f: Arc<CustomLangModelInferFunc>) -> Self {
        Self {
            inner: LangModelInner::Custom(CustomLangModel { infer_func: f }),
        }
    }
}

impl LangModelInference for LangModel {
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        match &mut self.inner {
            LangModelInner::Local(model) => model.infer(msgs, tools, config),
            LangModelInner::StreamAPI(model) => model.infer(msgs, tools, config),
            LangModelInner::Custom(model) => model.infer(msgs, tools, config),
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, Py, PyAny, PyRef, PyResult, Python,
        exceptions::{PyRuntimeError, PyStopAsyncIteration, PyStopIteration},
        pyclass, pymethods,
        types::PyType,
    };
    use pyo3_stub_gen_derive::*;

    use super::*;
    use crate::ffi::py::{base::await_future, cache_progress::await_cache_result};

    fn spawn<'a>(
        mut model: LangModel,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> anyhow::Result<(
        &'a tokio::runtime::Runtime,
        async_channel::Receiver<anyhow::Result<MessageOutput>>,
    )> {
        let (tx, rx) = async_channel::unbounded::<anyhow::Result<MessageOutput>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut stream = model.infer(messages, tools, config).boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    break; // Exit if consumer vanished
                }
                // Add a yield point to allow other tasks to run
                tokio::task::yield_now().await;
            }
        });
        Ok((rt, rx))
    }

    #[gen_stub_pyclass]
    #[pyclass(unsendable)]
    pub struct LanguageModelRunIterator {
        rx: async_channel::Receiver<anyhow::Result<MessageOutput>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LanguageModelRunIterator {
        fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
        fn __anext__(&self, py: Python<'_>) -> anyhow::Result<Py<PyAny>> {
            let rx = self.rx.clone();
            let fut = async move {
                match rx.recv().await {
                    Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e.to_string())),
                    Err(_) => Err(PyStopAsyncIteration::new_err(())),
                }
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut.into())
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(unsendable)]
    pub struct LanguageModelRunSyncIterator {
        rt: &'static tokio::runtime::Runtime,
        rx: async_channel::Receiver<anyhow::Result<MessageOutput>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LanguageModelRunSyncIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, py: Python<'_>) -> anyhow::Result<MessageOutput> {
            let item = py.detach(|| self.rt.block_on(self.rx.recv()));
            match item {
                Ok(res) => res.map_err(|e| PyRuntimeError::new_err(e.to_string())),
                Err(_) => Err(PyStopIteration::new_err(())),
            }
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LangModel {
        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[LangModel]"))]
        #[pyo3(name = "CreateLocal", signature = (model_name, progress_callback = None))]
        fn create_local<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            model_name: String,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> Result<Bound<'a, PyAny>> {
            let fut = async move {
                let inner =
                    await_cache_result::<LocalLangModel>(model_name, progress_callback).await?;
                Python::attach(|py| {
                    Py::new(
                        py,
                        LangModel {
                            inner: LangModelInner::Local(inner),
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
        ) -> anyhow::Result<Py<Self>> {
            let inner = await_future(await_cache_result::<LocalLangModel>(
                model_name,
                progress_callback,
            ))?;
            Py::new(
                py,
                LangModel {
                    inner: LangModelInner::Local(inner),
                },
            )
        }

        #[classmethod]
        #[pyo3(name = "CreateStreamAPI", signature = (spec, model_name, api_key))]
        fn create_stream_api<'a>(
            _cls: &Bound<'a, PyType>,
            spec: APISpecification,
            model_name: String,
            api_key: String,
        ) -> LangModel {
            LangModel {
                inner: LangModelInner::StreamAPI(StreamAPILangModel::new(
                    spec, model_name, api_key,
                )),
            }
        }

        #[pyo3(signature = (messages, tools, config))]
        fn run(
            &mut self,
            messages: Vec<Message>,
            tools: Vec<ToolDesc>,
            config: InferenceConfig,
        ) -> anyhow::Result<LanguageModelRunIterator> {
            let (_, rx) = spawn(self.clone(), messages, tools, config)?;
            Ok(LanguageModelRunIterator { rx })
        }

        #[pyo3(signature = (messages, tools, config))]
        fn run_sync(
            &mut self,
            messages: Vec<Message>,
            tools: Vec<ToolDesc>,
            config: InferenceConfig,
        ) -> anyhow::Result<LanguageModelRunSyncIterator> {
            let (rt, rx) = spawn(self.clone(), messages, tools, config)?;
            Ok(LanguageModelRunSyncIterator { rt, rx })
        }
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use std::sync::Arc;

    use futures::{StreamExt, lock::Mutex};
    use napi::{
        Error, JsSymbol, Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction,
    };
    use napi_derive::napi;
    use tokio::sync::mpsc;

    use super::*;

    #[napi(object)]
    pub struct LanguageModelIteratorResult {
        pub value: MessageOutput,
        pub done: bool,
    }

    #[derive(Clone)]
    #[napi]
    pub struct LangModelRunIterator {
        rx: Arc<Mutex<mpsc::UnboundedReceiver<std::result::Result<MessageOutput, anyhow::Error>>>>,
    }

    #[napi]
    impl LangModelRunIterator {
        #[napi(js_name = "[Symbol.asyncIterator]")]
        pub fn async_iterator(&self) -> &Self {
            // This is a dummy function to add typing for Symbol.asyncIterator
            self
        }

        #[napi]
        pub async unsafe fn next(&mut self) -> napi::Result<LanguageModelIteratorResult> {
            let mut rx = self.rx.lock().await;
            match rx.recv().await {
                Some(Ok(output)) => Ok(LanguageModelIteratorResult {
                    value: output.into(),
                    done: false,
                }),
                Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
                None => Ok(LanguageModelIteratorResult {
                    value: MessageOutput::new().into(),
                    done: true,
                }),
            }
        }
    }

    impl LangModelRunIterator {
        /// This returns an object with \[Symbol.asyncIterator\], which is not directly injected by napi-rs.
        fn to_async_iterator<'a>(self, env: Env) -> napi::Result<Object<'a>> {
            let mut obj = Object::new(&env)?;

            let global = env.get_global()?;
            let symbol: Function = global.get_named_property("Symbol")?;
            let symbol_async_iterator: JsSymbol = symbol.get_named_property("asyncIterator")?;

            let func: Function<(), LangModelRunIterator> =
                env.create_function_from_closure("asyncIterator", move |_| Ok(self.clone()))?;

            obj.set_property(symbol_async_iterator, func)?;

            Ok(obj)
        }
    }

    impl FromNapiValue for LangModel {
        unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
            let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
            let inner = ci.as_ref().inner.clone();
            Ok(Self { inner })
        }
    }

    #[napi]
    impl LangModel {
        #[napi]
        pub async fn create_local(
            model_name: String,
            progress_callback: Option<
                ThreadsafeFunction<
                    crate::ffi::node::cache::JsCacheProgress,
                    (),
                    crate::ffi::node::cache::JsCacheProgress,
                    Status,
                    false,
                >,
            >,
        ) -> napi::Result<LangModel> {
            let inner = crate::ffi::node::cache::await_cache_result::<LocalLangModel>(
                model_name,
                progress_callback,
            )
            .await
            .unwrap();
            Ok(LangModel {
                inner: LangModelInner::Local(inner),
            })
        }

        #[napi]
        pub fn create_stream_api(
            spec: APISpecification,
            model_name: String,
            api_key: String,
        ) -> napi::Result<LangModel> {
            let inner = StreamAPILangModel::new(spec, model_name, api_key);
            Ok(LangModel {
                inner: LangModelInner::StreamAPI(inner),
            })
        }

        #[napi(ts_return_type = "LangModelRunIterator")]
        pub fn run<'a>(
            &'a mut self,
            env: Env,
            messages: Vec<Message>,
            tools: Option<Vec<ToolDesc>>,
        ) -> Result<Object<'a>> {
            let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<MessageOutput, _>>();
            let rt = crate::ffi::node::common::get_or_create_runtime();
            let mut model = self.clone();

            rt.spawn(async move {
                // let mut model = inner.model.lock().await;
                let mut stream = model
                    .infer(
                        messages,
                        tools.unwrap_or(vec![]),
                        InferenceConfig::default(),
                    )
                    .boxed();

                while let Some(item) = stream.next().await {
                    if tx.send(item).is_err() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            });

            let it = LangModelRunIterator {
                rx: Arc::new(Mutex::new(rx)),
            };
            it.to_async_iterator(env)
        }
    }
}
