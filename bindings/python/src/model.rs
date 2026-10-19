//! `LangModel`, and the per-org providers models are built through.
//!
//! An org's provider is a backend (`"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"`)
//! or a factory of Python's own, as [`LangModelProvider::custom`] is in Rust: called with the
//! org and the name after it (`None` for a desc naming only the org), it returns a model. That
//! is a `LangModel`, used as is, or any object with
//!
//! - `infer(messages, tools, options)`, returning a `MessageOutput` dict or an awaitable of
//!   one, and optionally
//! - `infer_stream(messages, tools, options)`, returning an async iterable of
//!   `MessageDeltaOutput` dicts. Without it, streaming gives `infer`'s output as one delta.
//!
//! Each takes the messages, tool descriptions and options in their dict forms. The factory is
//! called synchronously, whenever an agent or `LangModel.create` builds a model of the org;
//! the methods off the event loop when synchronous and on the turn's when async, as tools are.

use std::sync::Arc;

use ailoy::{
    message::{Message, MessageDeltaOutput, MessageOutput},
    model::{
        Backend, InferLangModel, LangModelOptions, LangModelProvider, create_lang_model,
        remove_lang_model_provider as remove_lang_model_provider_rs,
        set_lang_model_provider as set_lang_model_provider_rs,
    },
    tool::ToolDesc,
};
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use pyo3::{
    exceptions::{PyStopAsyncIteration, PyValueError},
    prelude::*,
};
use pyo3_async_runtimes::{
    into_future_with_locals,
    tokio::{future_into_py, get_current_locals},
};

use crate::{
    agent::{OnRuntime, PyAgentRun, item},
    convert::{from_py, to_py},
    error,
    tool::{call_py, event_loop},
};

/// A model built from its desc, as an `Agent` builds one, run on a conversation directly.
#[pyclass(name = "LangModel", module = "ailoy", frozen)]
pub struct PyLangModel(OnRuntime<Arc<dyn InferLangModel>>);

impl PyLangModel {
    fn model(&self) -> Arc<dyn InferLangModel> {
        self.0
            .0
            .clone()
            .expect("a model is only let go of when dropped")
    }
}

/// The arguments of `infer` and `infer_stream`, from their Python forms.
fn infer_args(
    messages: &Bound<'_, PyAny>,
    tools: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Vec<Message>, Vec<ToolDesc>, LangModelOptions)> {
    let messages = from_py(messages, "a list of messages")?;
    let tools = match tools {
        Some(tools) => from_py(tools, "a list of tool descriptions")?,
        None => Vec::new(),
    };
    let options = match options {
        Some(options) => from_py(options, "language model options")?,
        None => LangModelOptions::default(),
    };
    Ok((messages, tools, options))
}

#[pymethods]
impl PyLangModel {
    /// The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
    /// `<org>`. Awaitable.
    #[staticmethod]
    fn create(py: Python<'_>, desc: String) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            let model = create_lang_model(&desc).await.map_err(error::anyhow)?;
            Ok(PyLangModel(OnRuntime(Some(model))))
        })
    }

    /// Run the model on `messages` to completion. Awaitable; answers with the whole output.
    #[pyo3(signature = (messages, *, tools = None, options = None))]
    fn infer<'py>(
        &self,
        py: Python<'py>,
        messages: Bound<'py, PyAny>,
        tools: Option<Bound<'py, PyAny>>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (messages, tools, options) = infer_args(&messages, tools.as_ref(), options.as_ref())?;
        let model = self.model();
        future_into_py(py, async move {
            item(model.infer(&messages, &tools, &options).await)
        })
    }

    /// Run the model on `messages` and iterate its output as deltas while it is generated.
    #[pyo3(signature = (messages, *, tools = None, options = None))]
    fn infer_stream(
        &self,
        messages: Bound<'_, PyAny>,
        tools: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyAgentRun> {
        let (messages, tools, options) = infer_args(&messages, tools.as_ref(), options.as_ref())?;
        let model = self.model();
        let stream = async_stream::stream! {
            let mut deltas = model.infer_stream(&messages, &tools, &options);
            while let Some(delta) = deltas.next().await {
                yield item(delta);
            }
        };
        Ok(PyAgentRun::new(stream.boxed()))
    }
}

/// A model of Python's own, as a factory returned it.
struct PyModel(Arc<Py<PyAny>>);

/// The positional arguments of the model's methods, in their dict forms.
fn args<'py>(
    py: Python<'py>,
    messages: &[Message],
    tools: &[ToolDesc],
    options: &LangModelOptions,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    Ok((
        to_py(py, &messages)?,
        to_py(py, &tools)?,
        to_py(py, options)?,
    ))
}

impl InferLangModel for PyModel {
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let model = self.0.clone();
        let (messages, tools, options) = (messages.to_vec(), tools.to_vec(), options.clone());
        Box::pin(async move {
            let returned = call_py(move |py| {
                let args = args(py, &messages, &tools, &options)?;
                model.call_method1(py, "infer", args)
            })
            .await?;
            Ok(Python::attach(|py| {
                from_py::<MessageOutput>(returned.bind(py), "a message output")
            })?)
        })
    }

    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let streams = Python::attach(|py| self.0.bind(py).hasattr("infer_stream"));
        if !matches!(streams, Ok(true)) {
            let output = self.infer(messages, tools, options);
            return stream::once(async move { output.await.map(MessageDeltaOutput::from) }).boxed();
        }

        let model = self.0.clone();
        let (messages, tools, options) = (messages.to_vec(), tools.to_vec(), options.clone());
        async_stream::try_stream! {
            let locals = Python::attach(|py| get_current_locals(py).ok());
            let iterator = call_py(move |py| {
                let args = args(py, &messages, &tools, &options)?;
                let iterable = model.call_method1(py, "infer_stream", args)?;
                Ok(iterable.bind(py).call_method0("__aiter__")?.unbind())
            })
            .await?;
            loop {
                let next = Python::attach(|py| {
                    let awaitable = iterator.bind(py).call_method0("__anext__")?;
                    into_future_with_locals(&event_loop(locals.as_ref())?, awaitable)
                })?;
                let delta = match next.await {
                    Ok(delta) => delta,
                    Err(e) if Python::attach(|py| e.is_instance_of::<PyStopAsyncIteration>(py)) => {
                        break;
                    }
                    Err(e) => Err(e)?,
                };
                yield Python::attach(|py| {
                    from_py::<MessageDeltaOutput>(delta.bind(py), "a message delta output")
                })?;
            }
        }
        .boxed()
    }
}

/// The models of `org` from what `factory` returns for it.
fn custom(factory: Py<PyAny>) -> LangModelProvider {
    LangModelProvider::custom(move |org, name| {
        Python::attach(|py| -> PyResult<Arc<dyn InferLangModel>> {
            let model = factory.call1(py, (org, name))?;
            if let Ok(model) = model.bind(py).cast::<PyLangModel>() {
                return Ok(model.get().model());
            }
            Ok(Arc::new(PyModel(Arc::new(model))))
        })
        .map_err(Into::into)
    })
}

/// Build the models of `org` (`"anthropic"`, `"openai"`, `"google"`, `"deepseek-ai"`, …)
/// through `provider`: a backend — `"cli"`, `"api"`, `"bedrock"`, `"openrouter"` or `"vllm"` —
/// or a factory, as the module says. Replaces the provider the org had.
#[pyfunction]
fn set_lang_model_provider(org: String, provider: Bound<'_, PyAny>) -> PyResult<()> {
    let provider = if provider.is_instance_of::<pyo3::types::PyString>() {
        LangModelProvider::from(from_py::<Backend>(&provider, "a backend")?)
    } else if provider.is_callable() {
        custom(provider.unbind())
    } else {
        return Err(PyValueError::new_err(
            "a lang model provider is a backend's name or a factory of models",
        ));
    };
    set_lang_model_provider_rs(org, provider);
    Ok(())
}

/// Stop building the models of `org`. Answers whether it had a provider.
#[pyfunction]
fn remove_lang_model_provider(org: String) -> bool {
    remove_lang_model_provider_rs(&org).is_some()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLangModel>()?;
    m.add_function(wrap_pyfunction!(set_lang_model_provider, m)?)?;
    m.add_function(wrap_pyfunction!(remove_lang_model_provider, m)?)?;
    Ok(())
}
