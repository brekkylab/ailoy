use std::sync::Arc;

use ailoy_macros::maybe_send_sync;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};

use crate::{
    cache::CacheProgress,
    model::{
        LocalLangModel, StreamAPILangModel,
        api::APISpecification,
        custom::{CustomLangModel, CustomLangModelInferFunc},
        polyfill::DocumentPolyfill,
    },
    utils::BoxStream,
    value::{Document, Message, MessageOutput, ToolDesc},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(string_enum = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum ThinkEffort {
    #[default]
    Disable,
    Enable,
    Low,
    Medium,
    High,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
#[cfg_attr(
    feature = "python",
    pyo3_stub_gen::derive::gen_stub_pyclass_complex_enum
)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
#[cfg_attr(feature = "nodejs", napi_derive::napi(discriminant_case = "lowercase"))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub enum Grammar {
    Plain {},
    JSON {},
    JSONSchema { schema: String },
    Regex { regex: String },
    CFG { cfg: String },
}

impl Default for Grammar {
    fn default() -> Self {
        Self::Plain {}
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, set_all))]
#[cfg_attr(feature = "nodejs", napi_derive::napi(object))]
#[cfg_attr(feature = "wasm", derive(tsify::Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
pub struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_polyfill: Option<DocumentPolyfill>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub think_effort: Option<ThinkEffort>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Grammar>,
}

#[maybe_send_sync]
pub trait LangModelInference {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
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
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
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
        docs: Vec<Document>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageOutput>> {
        match &mut self.inner {
            LangModelInner::Local(model) => model.infer(msgs, tools, docs, config),
            LangModelInner::StreamAPI(model) => model.infer(msgs, tools, docs, config),
            LangModelInner::Custom(model) => model.infer(msgs, tools, docs, config),
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use pyo3::{
        Bound, Py, PyAny, PyRef, PyResult, Python,
        exceptions::{PyStopAsyncIteration, PyStopIteration},
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
        documents: Vec<Document>,
        config: InferenceConfig,
    ) -> anyhow::Result<(
        &'a tokio::runtime::Runtime,
        async_channel::Receiver<anyhow::Result<MessageOutput>>,
    )> {
        let (tx, rx) = async_channel::unbounded::<anyhow::Result<MessageOutput>>();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.spawn(async move {
            let mut stream = model.infer(messages, tools, documents, config).boxed();

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
    pub struct LangModelRunIterator {
        rx: async_channel::Receiver<anyhow::Result<MessageOutput>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LangModelRunIterator {
        fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
        fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let rx: async_channel::Receiver<Result<MessageOutput, anyhow::Error>> = self.rx.clone();
            let fut = async move {
                match rx.recv().await {
                    Ok(res) => res.map_err(Into::into),
                    Err(_) => Err(PyStopAsyncIteration::new_err(())),
                }
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut.into())
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(unsendable)]
    pub struct LangModelRunSyncIterator {
        rt: &'static tokio::runtime::Runtime,
        rx: async_channel::Receiver<anyhow::Result<MessageOutput>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LangModelRunSyncIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
            let item = py.detach(|| self.rt.block_on(self.rx.recv()));
            match item {
                Ok(res) => res.map_err(Into::into),
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
        ) -> PyResult<Bound<'a, PyAny>> {
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
        ) -> PyResult<Py<Self>> {
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

        #[pyo3(signature = (messages, tools=None, documents=None, config=None))]
        fn run(
            &mut self,
            messages: Vec<Message>,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<LangModelRunIterator> {
            let (_, rx) = spawn(
                self.clone(),
                messages,
                tools.unwrap_or_default(),
                documents.unwrap_or_default(),
                config.unwrap_or_default(),
            )?;
            Ok(LangModelRunIterator { rx })
        }

        #[pyo3(signature = (messages, tools=None, documents=None, config=None))]
        fn run_sync(
            &mut self,
            messages: Vec<Message>,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<LangModelRunSyncIterator> {
            let (rt, rx) = spawn(
                self.clone(),
                messages,
                tools.unwrap_or_default(),
                documents.unwrap_or_default(),
                config.unwrap_or_default(),
            )?;
            Ok(LangModelRunSyncIterator { rt, rx })
        }

        pub fn __repr__(&self) -> String {
            // FIXME: provide model name or sth?
            let s = match &self.inner {
                LangModelInner::Local(_) => "LocalLangModel()",
                LangModelInner::StreamAPI(_) => "StreamAPILangModel()",
                LangModelInner::Custom(_) => "CustomLangModel()",
            };
            format!("LangModel({})", s)
        }
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl InferenceConfig {
        #[new]
        // #[pyo3(signature = (document_polyfill=None, think_effort=None, temperature=None, top_p=None, max_tokens=None, grammar=None))]
        #[pyo3(signature = (document_polyfill=None, think_effort=None, temperature=None, top_p=None, max_tokens=None))]
        fn __new__(
            document_polyfill: Option<DocumentPolyfill>,
            think_effort: Option<ThinkEffort>,
            temperature: Option<f64>,
            top_p: Option<f64>,
            max_tokens: Option<i32>,
            // grammar: Option<Grammar>,
        ) -> InferenceConfig {
            Self {
                document_polyfill: document_polyfill,
                think_effort: think_effort,
                temperature,
                top_p,
                max_tokens,
                grammar: Some(Grammar::default()),
            }
        }

        // TODO: initialize from {"think_effort": "enable", ...}
        // fn from_dict(d: Py<PyDict>)
    }
}

#[cfg(feature = "nodejs")]
mod node {
    use std::sync::Arc;

    use futures::{StreamExt, lock::Mutex};
    use napi::{
        Error, JsSymbol, Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction,
    };
    use tokio::sync::mpsc;

    use super::*;

    #[napi_derive::napi(object)]
    pub struct LanguageModelIteratorResult {
        pub value: MessageOutput,
        pub done: bool,
    }

    #[derive(Clone)]
    #[napi_derive::napi]
    pub struct LangModelRunIterator {
        rx: Arc<Mutex<mpsc::UnboundedReceiver<std::result::Result<MessageOutput, anyhow::Error>>>>,
    }

    #[napi_derive::napi]
    impl LangModelRunIterator {
        #[napi_derive::napi(js_name = "[Symbol.asyncIterator]")]
        pub fn async_iterator(&self) -> &Self {
            // This is a dummy function to add typing for Symbol.asyncIterator
            self
        }

        #[napi_derive::napi]
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

    #[napi_derive::napi]
    impl LangModel {
        #[napi_derive::napi]
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

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::{ffi::web::stream_to_async_iterable, model::api::APISpecification};

    #[wasm_bindgen]
    impl LangModel {
        #[wasm_bindgen]
        pub async fn create_local(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(
                js_name = "progressCallback",
                unchecked_param_type = "(progress: CacheProgress) => void"
            )]
            progress_callback: Option<js_sys::Function>,
        ) -> Result<LangModel, js_sys::Error> {
            let inner = crate::ffi::web::await_cache_result::<LocalLangModel>(
                model_name,
                progress_callback,
            )
            .await
            .map_err(|e| js_sys::Error::new(&e.to_string()))?;
            Ok(LangModel {
                inner: LangModelInner::Local(inner),
            })
        }

        #[wasm_bindgen]
        pub async fn create_stream_api(
            spec: APISpecification,
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "apiKey")] api_key: String,
        ) -> LangModel {
            LangModel::new_stream_api(spec, model_name, api_key)
        }

        #[wasm_bindgen(js_name = infer, unchecked_return_type = "AsyncIterable<MessageOutput>")]
        pub fn infer_js(
            &mut self,
            msgs: Vec<Message>,
            tools: Option<Vec<ToolDesc>>,
            config: Option<InferenceConfig>,
        ) -> JsValue {
            let mut model = self.clone();
            let stream = async_stream::stream! {
                let mut inner_stream = model.infer(msgs, tools.unwrap_or(vec![]), config.unwrap_or_default());
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|result| {
                result
                    .map(|output| output.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            stream_to_async_iterable(js_stream).into()
        }
    }
}
