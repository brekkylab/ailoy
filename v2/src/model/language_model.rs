use std::sync::Arc;

use ailoy_macros::maybe_send_sync;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use crate::{
    cache::CacheProgress,
    model::{
        LocalLangModel, StreamAPILangModel,
        api::APISpecification,
        custom::{CustomLangModel, CustomLangModelInferFunc},
        polyfill::DocumentPolyfill,
    },
    utils::{BoxFuture, BoxStream},
    value::{
        Delta, Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput,
        ToolDesc,
    },
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, EnumString, Display)]
#[serde(rename_all = "lowercase")]
#[strum(serialize_all = "lowercase")]
#[cfg_attr(feature = "python", derive(ailoy_macros::PyStringEnum))]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            document_polyfill: Some(DocumentPolyfill::default()),
            think_effort: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            grammar: None,
        }
    }
}

#[maybe_send_sync]
pub trait LangModelInference {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn infer_delta<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>>;

    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: InferenceConfig,
    ) -> BoxFuture<'a, anyhow::Result<MessageOutput>> {
        Box::pin(async move {
            let mut strm = self.infer_delta(msgs, tools, docs, config);
            let mut acc_delta = MessageDelta::new();
            let mut acc_finish_reason: Option<FinishReason> = None;
            while let Some(out_opt) = strm.next().await {
                let MessageDeltaOutput {
                    delta,
                    finish_reason,
                } = out_opt?;
                if finish_reason.is_some() {
                    acc_finish_reason = finish_reason;
                }
                acc_delta = acc_delta.accumulate(delta)?;
            }
            Ok(MessageOutput {
                message: acc_delta.finish()?,
                finish_reason: acc_finish_reason
                    .ok_or_else(|| anyhow::anyhow!("Inference finished without reason"))?,
            })
        })
    }
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
    fn infer_delta<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        docs: Vec<Document>,
        config: InferenceConfig,
    ) -> BoxStream<'a, anyhow::Result<MessageDeltaOutput>> {
        match &mut self.inner {
            LangModelInner::Local(model) => model.infer_delta(msgs, tools, docs, config),
            LangModelInner::StreamAPI(model) => model.infer_delta(msgs, tools, docs, config),
            LangModelInner::Custom(model) => model.infer_delta(msgs, tools, docs, config),
        }
    }
}

#[cfg(feature = "python")]
mod py {
    use futures::lock::Mutex;
    use pyo3::{Bound, Py, PyAny, PyResult, Python, pymethods, types::PyType};
    use pyo3_stub_gen_derive::*;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        ffi::py::{base::await_future, cache_progress::await_cache_result},
        value::{
            Messages,
            py::{MessageDeltaOutputIterator, MessageDeltaOutputSyncIterator},
        },
    };

    fn spawn_delta<'a>(
        mut model: LangModel,
        messages: Vec<Message>,
        tools: Vec<ToolDesc>,
        documents: Vec<Document>,
        config: InferenceConfig,
    ) -> anyhow::Result<(
        tokio::runtime::Runtime,
        mpsc::UnboundedReceiver<anyhow::Result<MessageDeltaOutput>>,
    )> {
        let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageDeltaOutput>>();
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.spawn(async move {
            let mut stream = model
                .infer_delta(messages, tools, documents, config)
                .boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break; // Exit if consumer vanished
                }
            }
        });
        Ok((rt, rx))
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LangModel {
        #[classmethod]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[LangModel]"))]
        #[pyo3(name = "new_local", signature = (model_name, progress_callback = None))]
        fn new_local_py<'a>(
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
        #[pyo3(name = "new_local_sync", signature = (model_name, progress_callback = None))]
        fn new_local_sync_py(
            _cls: &Bound<'_, PyType>,
            py: Python<'_>,
            model_name: String,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Py<Self>> {
            let inner = await_future(
                py,
                await_cache_result::<LocalLangModel>(model_name, progress_callback),
            )?;
            Py::new(
                py,
                LangModel {
                    inner: LangModelInner::Local(inner),
                },
            )
        }

        #[classmethod]
        #[pyo3(name = "new_stream_api", signature = (spec, model_name, api_key))]
        fn new_stream_api_py<'a>(
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

        #[pyo3(name="infer_delta", signature = (messages, tools=None, documents=None, config=None))]
        fn infer_delta_py(
            &mut self,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<MessageDeltaOutputIterator> {
            let (_rt, rx) = spawn_delta(
                self.clone(),
                messages.into(),
                tools.unwrap_or_default(),
                documents.unwrap_or_default(),
                config.unwrap_or_default(),
            )?;
            Ok(MessageDeltaOutputIterator {
                _rt,
                rx: Arc::new(Mutex::new(rx)),
            })
        }

        #[pyo3(name="infer_delta_sync", signature = (messages, tools=None, documents=None, config=None))]
        fn infer_delta_sync_py(
            &mut self,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> anyhow::Result<MessageDeltaOutputSyncIterator> {
            let (_rt, rx) = spawn_delta(
                self.clone(),
                messages.into(),
                tools.unwrap_or_default(),
                documents.unwrap_or_default(),
                config.unwrap_or_default(),
            )?;
            Ok(MessageDeltaOutputSyncIterator { _rt, rx })
        }

        #[pyo3(name="infer", signature = (messages, tools=None, documents=None, config=None))]
        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
        fn infer_py<'py>(
            &'py self,
            py: Python<'py>,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> PyResult<Py<PyAny>> {
            let mut this = self.clone();
            let fut = async move {
                let out = this
                    .infer(
                        messages.into(),
                        tools.unwrap_or_default(),
                        documents.unwrap_or_default(),
                        config.unwrap_or_default(),
                    )
                    .await?;
                Python::attach(|py| Py::new(py, out))
            };
            let py_fut = pyo3_async_runtimes::tokio::future_into_py(py, fut)?.unbind();
            Ok(py_fut)
        }

        #[pyo3(name="infer_sync", signature = (messages, tools=None, documents=None, config=None))]
        fn infer_sync_py<'py>(
            &'py self,
            py: Python<'_>,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            documents: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> PyResult<Py<MessageOutput>> {
            let mut this = self.clone();
            let fut = async move {
                let out = this
                    .infer(
                        messages.into(),
                        tools.unwrap_or_default(),
                        documents.unwrap_or_default(),
                        config.unwrap_or_default(),
                    )
                    .await?;
                Python::attach(|py| Py::new(py, out))
            };
            await_future(py, fut)
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
    use napi::{Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction};
    use napi_derive::napi;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        ffi::node::{
            cache::{JsCacheProgress, await_cache_result},
            common::get_or_create_runtime,
        },
        value::Messages,
    };

    #[napi]
    impl LangModel {
        #[napi(js_name = "newLocal")]
        pub async fn new_local_js(
            model_name: String,
            progress_callback: Option<
                ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>,
            >,
        ) -> napi::Result<LangModel> {
            let inner = await_cache_result::<LocalLangModel>(model_name, progress_callback)
                .await
                .unwrap();
            Ok(LangModel {
                inner: LangModelInner::Local(inner),
            })
        }

        #[napi(js_name = "newStreamAPI")]
        pub async fn new_stream_api_js(
            spec: APISpecification,
            model_name: String,
            api_key: String,
        ) -> napi::Result<LangModel> {
            let inner = StreamAPILangModel::new(spec, model_name, api_key);
            Ok(LangModel {
                inner: LangModelInner::StreamAPI(inner),
            })
        }

        #[napi(js_name = "inferDelta", ts_return_type = "MessageDeltaOutputIterator")]
        pub fn infer_delta_node<'a>(
            &'a mut self,
            env: Env,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            docs: Option<Vec<Document>>,
        ) -> Result<Object<'a>> {
            let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<MessageDeltaOutput, _>>();
            let rt = get_or_create_runtime();
            let mut model = self.clone();

            rt.spawn(async move {
                let mut stream = model
                    .infer_delta(
                        messages.into(),
                        tools.unwrap_or(vec![]),
                        docs.unwrap_or(vec![]),
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

            let it = crate::value::node::MessageDeltaOutputIterator {
                rx: Arc::new(Mutex::new(rx)),
            };
            it.to_async_iterator(env)
        }

        #[napi(js_name = "infer", ts_return_type = "MessageDeltaOutput")]
        pub async unsafe fn infer_node<'a>(
            &mut self,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            docs: Option<Vec<Document>>,
        ) -> napi::Result<MessageOutput> {
            let result = self
                .infer(
                    messages.into(),
                    tools.unwrap_or(vec![]),
                    docs.unwrap_or(vec![]),
                    InferenceConfig::default(),
                )
                .await
                .map_err(|v| napi::Error::from_reason(v.to_string()));
            result
        }
    }
}

#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::{
        ffi::web::{CacheProgressCallbackFn, stream_to_async_iterable},
        model::api::APISpecification,
        value::Messages,
    };

    #[wasm_bindgen]
    impl LangModel {
        #[wasm_bindgen(js_name = "newLocal")]
        pub async fn new_local_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "progressCallback")] progress_callback: Option<
                CacheProgressCallbackFn,
            >,
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

        #[wasm_bindgen(js_name = "newStreamAPI")]
        pub async fn new_stream_api_js(
            spec: APISpecification,
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "apiKey")] api_key: String,
        ) -> LangModel {
            LangModel::new_stream_api(spec, model_name, api_key)
        }

        #[wasm_bindgen(js_name = inferDelta, unchecked_return_type = "AsyncIterable<MessageDeltaOutput>")]
        pub fn infer_delta_js(
            &mut self,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            docs: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> Result<JsValue, js_sys::Error> {
            let mut model = self.clone();
            let messages: Vec<Message> = messages.try_into()?;
            let stream = async_stream::stream! {
                let mut inner_stream = model.infer_delta(messages, tools.unwrap_or(vec![]), docs.unwrap_or(vec![]), config.unwrap_or_default());
                while let Some(item) = inner_stream.next().await {
                    yield item;
                }
            };
            let js_stream = Box::pin(stream.map(|result| {
                result
                    .map(|output| output.into())
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }));

            Ok(stream_to_async_iterable(js_stream).into())
        }

        #[wasm_bindgen(js_name = infer)]
        pub async fn infer_js(
            &mut self,
            messages: Messages,
            tools: Option<Vec<ToolDesc>>,
            docs: Option<Vec<Document>>,
            config: Option<InferenceConfig>,
        ) -> Result<MessageOutput, JsValue> {
            let messages = messages.try_into()?;
            self.infer(
                messages,
                tools.unwrap_or(vec![]),
                docs.unwrap_or(vec![]),
                config.unwrap_or_default(),
            )
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}
