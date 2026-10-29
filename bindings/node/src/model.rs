//! `LangModel`, and the per-org providers models are built through.
//!
//! An org's provider is a backend (`'cli'`, `'api'`, `'bedrock'`, `'openrouter'`, `'vllm'` or
//! `'openai-compatible'`), a server of your own as an object (`{ backend: 'openai-compatible',
//! base_url: 'http://localhost:11434/v1' }`, with an `api_key` if the server wants one, as
//! [`LangModelServer`] is in Rust), or a factory of JavaScript's own, as
//! [`LangModelProvider::custom`] is in Rust: called with the org and the name after it (`null`
//! for a desc naming only the org), it returns a model. That is a `LangModel`, used as is, or
//! any object with
//!
//! - `infer(messages, tools, options)`, returning a `MessageOutput` or a promise of one, and
//!   optionally
//! - `inferStream(messages, tools, options)`, returning an async iterable of
//!   `MessageDeltaOutput`s. Without it, streaming gives `infer`'s output as one delta.
//!
//! The methods are called with `this` as the model. The factory is called whenever an agent or
//! `LangModel.create` builds a model of the org, from the runtime's thread, which waits for the
//! JavaScript thread to run it; it must return the model, not a promise of it.

use std::{ptr, sync::Arc};

use ailoy::{
    message::{Message, MessageDeltaOutput, MessageOutput},
    model::{
        Backend, InferLangModel, LangModelOptions, LangModelProvider, LangModelServer,
        create_lang_model, get_lang_model_provider as get_lang_model_provider_rs,
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
use napi::{
    Env, JsValue, Status, ValueType,
    bindgen_prelude::{
        ClassInstance, FnArgs, FromNapiValue, Function, Object, PromiseRaw, Unknown,
    },
    sys,
    threadsafe_function::ThreadsafeFunction,
};
use napi_derive::napi;
use serde::Deserialize;
use tokio::runtime::Handle;

use crate::{
    agent::{Item, JsAgentRun, OnRuntime},
    convert::{Json, from_js},
    error::{self, Result, invalid, promise, thrown},
    tool::{Returned, reason},
};

/// A model built from its desc, as an `Agent` builds one, run on a conversation directly.
#[napi(js_name = "LangModel")]
pub struct JsLangModel {
    model: OnRuntime<Arc<dyn InferLangModel>>,
}

impl JsLangModel {
    fn model(&self) -> Arc<dyn InferLangModel> {
        self.model
            .value
            .clone()
            .expect("a model is only let go of when dropped")
    }
}

/// The arguments of `infer` and `inferStream`, from their JavaScript forms.
fn infer_args(
    env: &Env,
    messages: Unknown<'_>,
    tools: Option<Unknown<'_>>,
    options: Option<Unknown<'_>>,
) -> Result<(Vec<Message>, Vec<ToolDesc>, LangModelOptions)> {
    let messages = from_js(env, messages, "a list of messages")?;
    let tools = match tools {
        Some(tools) => from_js(env, tools, "a list of tool descriptions")?,
        None => Vec::new(),
    };
    let options = match options {
        Some(options) => from_js(env, options, "language model options")?,
        None => LangModelOptions::default(),
    };
    Ok((messages, tools, options))
}

#[napi]
impl JsLangModel {
    /// The model `desc` names, `<org>` or `<org>/<name>`, built through the provider set for
    /// `<org>`.
    #[napi(ts_return_type = "Promise<LangModel>")]
    pub fn create<'env>(
        env: &'env Env,
        desc: String,
    ) -> napi::Result<PromiseRaw<'env, JsLangModel>> {
        promise(env, async move {
            let model = create_lang_model(&desc).await.map_err(error::anyhow)?;
            Ok(JsLangModel {
                model: OnRuntime::new(model, Handle::current()),
            })
        })
    }

    /// Run the model on `messages` to completion, settling with the whole output.
    #[napi(ts_return_type = "Promise<MessageOutput>")]
    pub fn infer<'env>(
        &self,
        env: &'env Env,
        #[napi(ts_arg_type = "Array<Message>")] messages: Unknown<'_>,
        #[napi(ts_arg_type = "Array<ToolDesc>")] tools: Option<Unknown<'_>>,
        #[napi(ts_arg_type = "LangModelOptions")] options: Option<Unknown<'_>>,
    ) -> napi::Result<PromiseRaw<'env, Json<MessageOutput>>> {
        let (messages, tools, options) =
            infer_args(env, messages, tools, options).map_err(|e| thrown(env, e))?;
        let model = self.model();
        promise(env, async move {
            model
                .infer(&messages, &tools, &options)
                .await
                .map(Json)
                .map_err(error::anyhow)
        })
    }

    /// Run the model on `messages` and iterate its output as deltas while it is generated.
    #[napi(ts_return_type = "AsyncIterableIterator<MessageDeltaOutput>")]
    pub fn infer_stream(
        &self,
        env: &Env,
        #[napi(ts_arg_type = "Array<Message>")] messages: Unknown<'_>,
        #[napi(ts_arg_type = "Array<ToolDesc>")] tools: Option<Unknown<'_>>,
        #[napi(ts_arg_type = "LangModelOptions")] options: Option<Unknown<'_>>,
    ) -> Result<JsAgentRun> {
        let (messages, tools, options) = infer_args(env, messages, tools, options)?;
        let model = self.model();
        let stream = async_stream::stream! {
            let mut deltas = model.infer_stream(&messages, &tools, &options);
            while let Some(delta) = deltas.next().await {
                yield delta.map(Item::Delta).map_err(error::anyhow);
            }
        };
        Ok(JsAgentRun::new(stream.boxed(), self.model.runtime.clone()))
    }
}

/// The arguments a custom model's methods are called with.
type InferArgs = FnArgs<(
    Json<Vec<Message>>,
    Json<Vec<ToolDesc>>,
    Json<LangModelOptions>,
)>;

/// A JavaScript function callable from the runtime's threads; weak, so a model does not keep
/// the process alive.
type Tsfn<Args, Return> = ThreadsafeFunction<Args, Return, Args, Status, false, true>;

/// A JavaScript function, bound to `this`, as a [`Tsfn`]. On the JavaScript thread.
fn bound<Args, Return>(
    function: Function<'_, Args, Return>,
    this: Object<'_>,
) -> napi::Result<Tsfn<Args, Return>>
where
    Args: 'static + napi::bindgen_prelude::JsValuesTupleIntoVec,
    Return: 'static + FromNapiValue,
{
    function
        .bind(this)?
        .build_threadsafe_function()
        .weak::<true>()
        .build()
}

/// A model of JavaScript's own, as a factory returned it.
struct JsModel {
    infer: Arc<Tsfn<InferArgs, Returned<MessageOutput>>>,
    infer_stream: Option<Arc<Tsfn<InferArgs, Iterator>>>,
}

fn infer_args_of(
    messages: &[Message],
    tools: &[ToolDesc],
    options: &LangModelOptions,
) -> InferArgs {
    FnArgs {
        data: (
            Json(messages.to_vec()),
            Json(tools.to_vec()),
            Json(options.clone()),
        ),
    }
}

impl InferLangModel for JsModel {
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let infer = self.infer.clone();
        let args = infer_args_of(messages, tools, options);
        Box::pin(async move {
            let returned = infer
                .call_async_catch(args)
                .await
                .map_err(|e| anyhow::anyhow!(reason(e)))?;
            returned.settle().await.map_err(anyhow::Error::msg)
        })
    }

    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let Some(infer_stream) = self.infer_stream.clone() else {
            let output = self.infer(messages, tools, options);
            return stream::once(async move { output.await.map(MessageDeltaOutput::from) }).boxed();
        };
        let args = infer_args_of(messages, tools, options);
        async_stream::try_stream! {
            let Iterator(next) = infer_stream
                .call_async_catch(args)
                .await
                .map_err(|e| anyhow::anyhow!(reason(e)))?;
            loop {
                let step = next
                    .call_async_catch(())
                    .await
                    .map_err(|e| anyhow::anyhow!(reason(e)))?
                    .settle()
                    .await
                    .map_err(anyhow::Error::msg)?;
                if step.done {
                    break;
                }
                yield step
                    .value
                    .ok_or_else(|| anyhow::anyhow!("a stream yielded no delta"))?;
            }
        }
        .boxed()
    }
}

/// An async iterator's result, `{ value, done }`.
#[derive(Deserialize)]
struct Step {
    #[serde(default)]
    done: bool,
    value: Option<MessageDeltaOutput>,
}

/// The iterator `inferStream` returned, as its `next` method.
struct Iterator(Tsfn<(), Returned<Step>>);

impl FromNapiValue for Iterator {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        let iterable = unsafe { Object::from_napi_value(env, napi_val)? };
        // An async generator is its own iterator; another iterable makes one.
        let iterator = if iterable.get::<Unknown>("next")?.is_some() {
            iterable
        } else {
            let key = unsafe { async_iterator_symbol(env)? };
            let mut method = ptr::null_mut();
            napi::check_status!(unsafe {
                sys::napi_get_property(env, napi_val, key, &mut method)
            })?;
            let method = unsafe { Function::<(), Unknown>::from_napi_value(env, method)? };
            let iterator = method.apply(iterable, ())?;
            unsafe { Object::from_napi_value(env, iterator.raw())? }
        };
        let next: Function<(), Returned<Step>> = iterator
            .get("next")?
            .ok_or_else(|| napi::Error::new(Status::InvalidArg, "not an async iterable"))?;
        Ok(Iterator(bound(next, iterator)?))
    }
}

/// `Symbol.asyncIterator`.
unsafe fn async_iterator_symbol(env: sys::napi_env) -> napi::Result<sys::napi_value> {
    let mut global = ptr::null_mut();
    napi::check_status!(unsafe { sys::napi_get_global(env, &mut global) })?;
    let global = unsafe { Object::from_napi_value(env, global)? };
    let symbol: Object = global
        .get("Symbol")?
        .ok_or_else(|| napi::Error::new(Status::GenericFailure, "no Symbol"))?;
    let key: Unknown = symbol
        .get("asyncIterator")?
        .ok_or_else(|| napi::Error::new(Status::GenericFailure, "no Symbol.asyncIterator"))?;
    Ok(key.raw())
}

/// What a factory returned: a `LangModel`, or a model of JavaScript's own.
struct Made(Arc<dyn InferLangModel>);

impl FromNapiValue for Made {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> napi::Result<Self> {
        if let Ok(model) = unsafe { ClassInstance::<JsLangModel>::from_napi_value(env, napi_val) } {
            return Ok(Made(model.model()));
        }
        let model = unsafe { Object::from_napi_value(env, napi_val)? };
        let infer: Function<InferArgs, Returned<MessageOutput>> = model
            .get("infer")?
            .ok_or_else(|| napi::Error::new(Status::InvalidArg, "a model has an `infer` method"))?;
        let infer_stream = match model.get::<Function<InferArgs, Iterator>>("inferStream")? {
            Some(infer_stream) => Some(Arc::new(bound(infer_stream, model)?)),
            None => None,
        };
        Ok(Made(Arc::new(JsModel {
            infer: Arc::new(bound(infer, model)?),
            infer_stream,
        })))
    }
}

type Factory = Tsfn<FnArgs<(String, Option<String>)>, Made>;

/// The models of an org from what `factory` returns for it.
fn custom(factory: Factory) -> LangModelProvider {
    LangModelProvider::custom(move |org, name| {
        let args = FnArgs {
            data: (org.to_owned(), name.map(str::to_owned)),
        };
        // Called on the runtime's thread, never the JavaScript one, which is free to run it.
        futures::executor::block_on(factory.call_async_catch(args))
            .map(|made| made.0)
            .map_err(|e| anyhow::anyhow!(reason(e)))
    })
}

/// Build the models of `org` (`'anthropic'`, `'openai'`, `'google'`, `'deepseek-ai'`, …)
/// through `provider`: a backend, a server object, or a factory, as the module says. Replaces
/// the provider the org had.
#[napi]
pub fn set_lang_model_provider(
    env: &Env,
    org: String,
    #[napi(
        ts_arg_type = "Backend | LangModelServer | ((org: string, name: string | null) => LangModel | CustomLangModel)"
    )]
    provider: Unknown<'_>,
) -> Result<()> {
    let provider = match provider.get_type().map_err(|e| invalid(e.reason))? {
        ValueType::String => {
            LangModelProvider::from(from_js::<Backend>(env, provider, "a backend")?)
        }
        ValueType::Object => LangModelProvider::from(from_js::<LangModelServer>(
            env,
            provider,
            "a lang model server",
        )?),
        ValueType::Function => {
            let factory = unsafe { Factory::from_napi_value(env.raw(), provider.raw()) }
                .map_err(|e| invalid(e.reason))?;
            custom(factory)
        }
        _ => {
            return Err(invalid(
                "a lang model provider is a backend's name, a server object or a factory of models",
            ));
        }
    };
    set_lang_model_provider_rs(org, provider);
    Ok(())
}

/// How the model `desc` names (`'<org>/<name>'`, or `'<org>'`) is built: a backend's name
/// (that of the server's backend, for a server of your own), `'custom'` for a factory, or
/// `null` when its org has no provider.
#[napi(ts_return_type = "Backend | 'custom' | null")]
pub fn get_lang_model_provider(desc: String) -> Option<String> {
    get_lang_model_provider_rs(&desc).map(|provider| match provider {
        LangModelProvider::Backend(backend) => backend.to_string(),
        LangModelProvider::Server(server) => server.backend().to_string(),
        LangModelProvider::Custom(_) => "custom".to_string(),
    })
}

/// Stop building the models of `org`. Answers whether it had a provider.
#[napi]
pub fn remove_lang_model_provider(org: String) -> bool {
    remove_lang_model_provider_rs(&org).is_some()
}
