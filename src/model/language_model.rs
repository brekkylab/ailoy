use std::sync::Arc;

use ailoy_macros::maybe_send_sync;
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use crate::{
    cache::CacheProgress,
    model::{
        LocalLangModel, LocalLangModelConfig, StreamAPILangModel,
        api::APISpecification,
        custom::{CustomLangModel, CustomLangModelInferFunc},
        polyfill::DocumentPolyfill,
    },
    utils::{BoxFuture, BoxStream},
    value::{
        Delta, Document, FinishReason, Message, MessageDelta, MessageDeltaOutput, MessageOutput,
        PartDelta, PartDeltaFunction, ToolDesc,
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
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
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

/// Configuration parameters that control the behavior of model inference.
///
/// `InferenceConfig` encapsulates all the configuration, controlling behavior of `LanguageModel`` inference.
///
/// # Fields
///
/// ## `document_polyfill`
/// Configuration describing how retrieved documents are embedded into the model input.
/// If `None`, it does not perform any polyfill, (ignoring documents).
///
/// ## `think_effort`
/// Controls the model’s reasoning intensity.
/// In local models, `low`, `medium`, `high` is ignored.
/// In API models, it is up to it's API. See API parameters.
///
/// Possible values: `disable`, `enable`, `low`, `medium`, `high`.
///
/// ## `temperature`
/// Sampling temperature controlling randomness of output.
/// Lower values make output more deterministic; higher values increase diversity.
///
/// ## `top_p`
/// Nucleus sampling parameter (probability mass cutoff).
/// Limits token sampling to a cumulative probability ≤ `top_p`.`
///
/// ## `max_tokens`
/// Maximum number of tokens to generate for a single inference.
///
/// ## `grammar`
/// Optional grammar constraint that restricts valid output forms.  
/// Supported types include:
/// - `Plain`: unconstrained text  
/// - `JSON`: ensures valid JSON output  
/// - `JSONSchema { schema }`: validates JSON against the given schema  
/// - `Regex { regex }`: constrains generation by a regular expression  
/// - `CFG { cfg }`: uses a context-free grammar definition
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass)]
#[cfg_attr(
    feature = "python",
    pyo3::pyclass(module = "ailoy._core", get_all, set_all)
)]
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
                acc_delta = acc_delta.accumulate(delta)?;
                if let Some(finish_reason) = finish_reason {
                    acc_finish_reason = Some(finish_reason.clone());
                    if let FinishReason::ToolCall {} = finish_reason {
                        let last_tool_call = acc_delta.tool_calls.pop().unwrap();
                        let (id, name, arguments) = last_tool_call.to_parsed_function().unwrap();
                        acc_delta.tool_calls.push(PartDelta::Function {
                            id,
                            function: PartDeltaFunction::WithParsedArgs { name, arguments },
                        })
                    }
                }
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
#[cfg_attr(feature = "python", pyo3::pyclass(module = "ailoy._core"))]
#[cfg_attr(feature = "nodejs", napi_derive::napi)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct LangModel {
    inner: LangModelInner,
}

impl LangModel {
    pub async fn try_new_local(
        model_name: impl Into<String>,
        config: Option<LocalLangModelConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: LangModelInner::Local(LocalLangModel::try_new(model_name, config).await?),
        })
    }

    pub fn try_new_local_stream<'a>(
        model_name: impl Into<String>,
        config: Option<LocalLangModelConfig>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<Self>>> {
        let model_name = model_name.into();
        Box::pin(async_stream::try_stream! {
            let mut strm = LocalLangModel::try_new_stream(model_name, config);
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

    pub fn download<'a>(
        model: impl Into<String>,
    ) -> BoxStream<'a, anyhow::Result<CacheProgress<()>>> {
        LocalLangModel::download(model)
    }

    pub async fn remove(model: impl Into<String>) -> anyhow::Result<()> {
        LocalLangModel::remove(model).await
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
    use pyo3::{
        Bound, Py, PyAny, PyResult, Python, pymethods,
        types::{PyDict, PyDictMethods, PyType},
    };
    use pyo3_stub_gen_derive::*;
    use tokio::sync::mpsc;

    use super::*;
    use crate::{
        boxed,
        ffi::py::{
            base::{await_future, python_to_value},
            cache_progress::await_cache_result,
        },
        model::KvCacheConfig,
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
            let mut stream = boxed!(model.infer_delta(messages, tools, documents, config));

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
        #[pyo3(name = "new_local", signature = (model_name, device_id = None, validate_checksum = None, kv_cache = None, progress_callback = None))]
        fn new_local_py<'a>(
            _cls: &Bound<'a, PyType>,
            py: Python<'a>,
            model_name: String,
            device_id: Option<i32>,
            validate_checksum: Option<bool>,
            kv_cache: Option<KvCacheConfig>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Bound<'a, PyAny>> {
            let config = LocalLangModelConfig {
                device_id,
                validate_checksum,
                kv_cache,
            };
            let cache_strm = LocalLangModel::try_new_stream(model_name, Some(config));
            let fut = async move {
                let inner = await_cache_result(cache_strm, progress_callback).await?;
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
        #[pyo3(name = "new_local_sync", signature = (model_name, device_id = None, validate_checksum = None, kv_cache = None, progress_callback = None))]
        fn new_local_sync_py(
            _cls: &Bound<'_, PyType>,
            py: Python<'_>,
            model_name: String,
            device_id: Option<i32>,
            validate_checksum: Option<bool>,
            kv_cache: Option<KvCacheConfig>,
            #[gen_stub(override_type(type_repr = "typing.Callable[[CacheProgress], None]"))]
            progress_callback: Option<Py<PyAny>>,
        ) -> PyResult<Py<Self>> {
            let config = LocalLangModelConfig {
                device_id,
                validate_checksum,
                kv_cache,
            };
            let cache_strm = LocalLangModel::try_new_stream(model_name, Some(config));
            let inner = await_future(py, await_cache_result(cache_strm, progress_callback))?;
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

        #[classmethod]
        #[pyo3(signature = (config))]
        pub fn from_dict(
            _cls: &Bound<'_, PyType>,
            config: &Bound<'_, PyDict>,
        ) -> PyResult<InferenceConfig> {
            use std::str::FromStr;

            use crate::model::DocumentPolyfillKind;

            let document_polyfill = config
                .get_item("document_polyfill")?
                .and_then(|kind| python_to_value(&kind).ok())
                .and_then(|kind| {
                    DocumentPolyfillKind::from_str(kind.as_str().unwrap_or_default()).ok()
                })
                .and_then(|kind| DocumentPolyfill::get(kind).ok());
            let think_effort = config
                .get_item("think_effort")?
                .and_then(|think_effort| python_to_value(&think_effort).ok())
                .and_then(|think_effort| {
                    ThinkEffort::from_str(think_effort.as_str().unwrap_or_default()).ok()
                });
            let temperature = config
                .get_item("temperature")?
                .and_then(|temperature| python_to_value(&temperature).ok())
                .and_then(|temperature| temperature.as_float());
            let top_p = config
                .get_item("top_p")?
                .and_then(|top_p| python_to_value(&top_p).ok())
                .and_then(|top_p| top_p.as_float());
            let max_tokens = config
                .get_item("max_tokens")?
                .and_then(|max_tokens| python_to_value(&max_tokens).ok())
                .and_then(|max_tokens| max_tokens.as_integer())
                .map(|max_tokens| max_tokens as i32);
            // Grammar is not yet supported, so configuring grammar is disabled for now.
            // let grammar = config
            //     .get_item("grammar")?
            //     .and_then(|grammar| python_to_value(&grammar).ok())
            //     .and_then(|grammar| {
            //         serde_json::from_str::<Grammar>(grammar.as_str().unwrap_or("{}")).ok()
            //     });
            Ok(InferenceConfig {
                document_polyfill,
                think_effort,
                temperature,
                top_p,
                max_tokens,
                grammar: Some(Grammar::default()),
            })
        }
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
        boxed,
        ffi::node::{
            cache::{JsCacheProgress, await_cache_result},
            common::get_or_create_runtime,
        },
        model::KvCacheConfig,
        value::Messages,
    };

    #[derive(Default)]
    #[napi(js_name = "LocalLangModelConfig", object, object_to_js = false)]
    pub struct JSLocalLangModelConfig {
        pub device_id: Option<i32>,
        pub validate_checksum: Option<bool>,
        pub kv_cache: Option<KvCacheConfig>,
        pub progress_callback:
            Option<ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>>,
    }

    #[napi]
    impl LangModel {
        #[napi(js_name = "newLocal")]
        pub async fn new_local_js(
            model_name: String,
            config: Option<JSLocalLangModelConfig>,
        ) -> napi::Result<LangModel> {
            let config = config.unwrap_or_default();
            let cache_strm = LocalLangModel::try_new_stream(
                model_name,
                Some(LocalLangModelConfig {
                    device_id: config.device_id,
                    validate_checksum: config.validate_checksum,
                    kv_cache: config.kv_cache,
                }),
            );
            let inner = await_cache_result(cache_strm, config.progress_callback)
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

        #[napi(js_name = "download")]
        pub async fn download_js(
            model_name: String,
            config: Option<JSLocalLangModelConfig>,
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
                let mut stream = boxed!(model.infer_delta(
                    messages.into(),
                    tools.unwrap_or(vec![]),
                    docs.unwrap_or(vec![]),
                    InferenceConfig::default(),
                ));

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
    use js_sys::{Function, Reflect};
    use tsify::Tsify;
    use wasm_bindgen::prelude::*;

    use super::*;
    use crate::{
        ffi::web::cache::await_cache_result,
        ffi::web::common::stream_to_async_iterable,
        model::{KVCacheConfig, api::APISpecification},
        value::Messages,
    };

    #[wasm_bindgen]
    extern "C" {
        #[derive(Clone)]
        #[wasm_bindgen(
            js_name = "LocalLangModelConfig",
            typescript_type = "{
                deviceId?: number; 
                validateChecksum?: boolean; 
                kvCache?: KVCacheConfig;
                progressCallback?: CacheProgressCallbackFn;
            }"
        )]
        pub type _JSLocalLangModelConfig;
    }

    #[derive(Default)]
    pub struct JSLocalLangModelConfig {
        pub device_id: Option<i32>,
        pub validate_checksum: Option<bool>,
        pub kv_cache: Option<KVCacheConfig>,
        pub progress_callback: Option<Function>,
    }

    impl TryFrom<_JSLocalLangModelConfig> for JSLocalLangModelConfig {
        type Error = js_sys::Error;

        fn try_from(value: _JSLocalLangModelConfig) -> Result<Self, Self::Error> {
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
                config.validate_checksum = Some(b)
            }

            if let Ok(val) = Reflect::get(&obj, &"kvCache".into()) {
                let kv_cache_config = KVCacheConfig::from_js(val).unwrap_or_default();
                config.kv_cache = Some(kv_cache_config);
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
    impl LangModel {
        #[wasm_bindgen(js_name = "newLocal")]
        pub async fn new_local_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            config: Option<_JSLocalLangModelConfig>,
        ) -> Result<LangModel, js_sys::Error> {
            let config: JSLocalLangModelConfig = match config {
                Some(c) => c.try_into()?,
                None => JSLocalLangModelConfig::default(),
            };
            let cache_strm = LocalLangModel::try_new_stream(
                model_name,
                Some(LocalLangModelConfig {
                    device_id: config.device_id,
                    validate_checksum: config.validate_checksum,
                    kv_cache: config.kv_cache,
                }),
            );
            let inner = await_cache_result(cache_strm, config.progress_callback)
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

        #[wasm_bindgen(js_name = "download")]
        pub async fn download_js(
            #[wasm_bindgen(js_name = "modelName")] model_name: String,
            #[wasm_bindgen(js_name = "config")] config: Option<_JSLocalLangModelConfig>,
        ) -> Result<(), js_sys::Error> {
            let config: JSLocalLangModelConfig = match config {
                Some(c) => c.try_into()?,
                None => JSLocalLangModelConfig::default(),
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
