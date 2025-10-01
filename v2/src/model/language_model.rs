use std::sync::Arc;

use futures::StreamExt as _;
use serde::Serialize;

use crate::{
    cache::CacheProgress,
    model::{CustomLangModel, LocalLangModel, StreamAPILangModel, api::APISpecification},
    utils::{BoxStream, MaybeSend, MaybeSync},
    value::{Message, MessageOutput, ToolDesc},
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "python", pyo3_stub_gen::derive::gen_stub_pyclass_enum)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
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
pub struct InferenceConfig {
    pub think_effort: ThinkEffort,

    pub temperature: Option<f64>,

    pub top_p: Option<f64>,

    pub max_tokens: Option<i32>,

    pub grammar: Grammar,
}

pub trait LangModelInference: MaybeSend + MaybeSync {
    /// Runs the language model with the given tools and messages, returning a stream of `MessageOutput`s.
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>>;
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
pub struct LangModel {
    inner: LangModelInner,
}

impl LangModel {
    pub async fn try_new_local(model_name: impl Into<String>) -> Result<Self, String> {
        Ok(Self {
            inner: LangModelInner::Local(LocalLangModel::try_new(model_name).await?),
        })
    }

    pub fn try_new_local_stream<'a>(
        model_name: impl Into<String>,
    ) -> BoxStream<'a, Result<CacheProgress<Self>, String>> {
        let model_name = model_name.into();
        Box::pin(async_stream::try_stream! {
            let mut strm = LocalLangModel::try_new_stream(model_name);
            while let Some(result) = strm.next().await {
                let result = result?;
                yield CacheProgress {
                    comment: result.comment,
                    current_task: result.current_task,
                    total_task: result.current_task,
                    result: result.result.map(|v| LangModel{inner: LangModelInner::Local(v)}),
                };
            }
        })
    }

    pub fn new_stream_api(
        provider: APISpecification,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            inner: LangModelInner::StreamAPI(StreamAPILangModel::new(provider, model, api_key)),
        }
    }

    pub fn new_custom(
        f: Arc<
            dyn Fn(
                    Vec<Message>,
                    Vec<ToolDesc>,
                    InferenceConfig,
                ) -> BoxStream<'static, Result<MessageOutput, String>>
                + MaybeSend
                + MaybeSync,
        >,
    ) -> Self {
        Self {
            inner: LangModelInner::Custom(CustomLangModel { run: f }),
        }
    }
}

impl LangModelInference for LangModel {
    fn infer<'a>(
        &'a mut self,
        msgs: Vec<Message>,
        tools: Vec<ToolDesc>,
        config: InferenceConfig,
    ) -> BoxStream<'a, Result<MessageOutput, String>> {
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
    ) -> PyResult<(
        &'a tokio::runtime::Runtime,
        async_channel::Receiver<Result<MessageOutput, String>>,
    )> {
        let (tx, rx) = async_channel::unbounded::<Result<MessageOutput, String>>();
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
        rx: async_channel::Receiver<Result<MessageOutput, String>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LanguageModelRunIterator {
        fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        #[gen_stub(override_return_type(type_repr = "typing.Awaitable[MessageOutput]"))]
        fn __anext__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
        rx: async_channel::Receiver<Result<MessageOutput, String>>,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl LanguageModelRunSyncIterator {
        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, py: Python<'_>) -> PyResult<MessageOutput> {
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
        #[pyo3(name = "CreateStreamAPI", signature = (model_name, api_key))]
        fn create_stream_api<'a>(
            _cls: &Bound<'a, PyType>,
            model_name: String,
            api_key: String,
        ) -> LangModel {
            LangModel {
                inner: LangModelInner::StreamAPI(StreamAPILangModel::new(model_name, api_key)),
            }
        }

        #[pyo3(signature = (messages, tools, config))]
        fn run(
            &mut self,
            messages: Vec<Message>,
            tools: Vec<ToolDesc>,
            config: InferenceConfig,
        ) -> PyResult<LanguageModelRunIterator> {
            let (_, rx) = spawn(self.clone(), messages, tools, config)?;
            Ok(LanguageModelRunIterator { rx })
        }

        #[pyo3(signature = (messages, tools, config))]
        fn run_sync(
            &mut self,
            messages: Vec<Message>,
            tools: Vec<ToolDesc>,
            config: InferenceConfig,
        ) -> PyResult<LanguageModelRunSyncIterator> {
            let (rt, rx) = spawn(self.clone(), messages, tools, config)?;
            Ok(LanguageModelRunSyncIterator { rt, rx })
        }
    }
}
