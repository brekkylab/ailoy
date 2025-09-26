use std::sync::Arc;

use futures::{StreamExt, lock::Mutex};
use napi::{Error, JsSymbol, Status, bindgen_prelude::*, threadsafe_function::ThreadsafeFunction};
use napi_derive::napi;
use tokio::sync::mpsc;

use crate::{
    ffi::node::{
        cache::{JsCacheProgress, await_cache_result},
        common::{await_future, get_or_create_runtime},
        value::{JsMessage, JsMessageOutput},
    },
    model::{
        ArcMutexLanguageModel, LocalLanguageModel, anthropic::AnthropicLanguageModel,
        gemini::GeminiLanguageModel, openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    value::{MessageOutput, ToolDesc},
};

#[napi(object)]
pub struct LanguageModelIteratorResult {
    pub value: JsMessageOutput,
    pub done: bool,
}

#[derive(Clone)]
#[napi]
pub struct LanguageModelRunIterator {
    rx: Arc<Mutex<mpsc::UnboundedReceiver<std::result::Result<MessageOutput, String>>>>,
}

#[napi]
impl LanguageModelRunIterator {
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

impl LanguageModelRunIterator {
    /// This returns an object with \[Symbol.asyncIterator\], which is not directly injected by napi-rs.
    fn to_async_iterator<'a>(self, env: Env) -> napi::Result<Object<'a>> {
        let mut obj = Object::new(&env)?;

        let global = env.get_global()?;
        let symbol: Function = global.get_named_property("Symbol")?;
        let symbol_async_iterator: JsSymbol = symbol.get_named_property("asyncIterator")?;

        let func: Function<(), LanguageModelRunIterator> =
            env.create_function_from_closure("asyncIterator", move |_| Ok(self.clone()))?;

        obj.set_property(symbol_async_iterator, func)?;

        Ok(obj)
    }
}

pub trait LanguageModelMethods {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel>;

    fn create_iterator<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> napi::Result<Object<'a>> {
        let inner = self.get_inner()?;
        let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<MessageOutput, String>>();

        let rt = get_or_create_runtime();

        rt.spawn(async move {
            let mut model = inner.model.lock().await;
            let mut stream = model
                .run(
                    messages.into_iter().map(|msg| msg.into()).collect(),
                    tools.unwrap_or(vec![]),
                )
                .boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        });

        let it = LanguageModelRunIterator {
            rx: Arc::new(Mutex::new(rx)),
        };
        it.to_async_iterator(env)
    }
}

#[napi(js_name = "LocalLanguageModel")]
pub struct JsLocalLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl FromNapiValue for JsLocalLanguageModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

impl LanguageModelMethods for JsLocalLanguageModel {
    fn get_inner(&self) -> napi::Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JsLocalLanguageModel {
    #[napi]
    pub async fn create(
        model_name: String,
        progress_callback: Option<
            ThreadsafeFunction<JsCacheProgress, (), JsCacheProgress, Status, false>,
        >,
    ) -> napi::Result<JsLocalLanguageModel> {
        let inner = await_cache_result::<LocalLanguageModel>(model_name, progress_callback)
            .await
            .unwrap();
        Ok(JsLocalLanguageModel {
            inner: ArcMutexLanguageModel::new(inner),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> Result<Object<'a>> {
        self.create_iterator(env, messages, tools)
    }

    #[napi]
    pub fn enable_reasoning(&self) -> napi::Result<()> {
        let inner = self.inner.into_inner::<LocalLanguageModel>().unwrap();
        await_future(async { Ok::<_, napi::Error>(inner.lock().await.enable_reasoning()) })
    }

    #[napi]
    pub fn disable_reasoning(&self) -> napi::Result<()> {
        let inner = self.inner.into_inner::<LocalLanguageModel>().unwrap();
        await_future(async { Ok::<_, napi::Error>(inner.lock().await.disable_reasoning()) })
    }
}

#[napi(js_name = "OpenAILanguageModel")]
pub struct JsOpenAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl FromNapiValue for JsOpenAILanguageModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

impl LanguageModelMethods for JsOpenAILanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JsOpenAILanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(OpenAILanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> Result<Object<'a>> {
        self.create_iterator(env, messages, tools)
    }
}

#[napi(js_name = "GeminiLanguageModel")]
pub struct JsGeminiLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl FromNapiValue for JsGeminiLanguageModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

impl LanguageModelMethods for JsGeminiLanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JsGeminiLanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(GeminiLanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> Result<Object<'a>> {
        self.create_iterator(env, messages, tools)
    }
}

#[napi(js_name = "AnthropicLanguageModel")]
pub struct JsAnthropicLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl FromNapiValue for JsAnthropicLanguageModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

impl LanguageModelMethods for JsAnthropicLanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JsAnthropicLanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(AnthropicLanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> Result<Object<'a>> {
        self.create_iterator(env, messages, tools)
    }
}

#[napi(js_name = "XAILanguageModel")]
pub struct JsXAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl FromNapiValue for JsXAILanguageModel {
    unsafe fn from_napi_value(env: sys::napi_env, napi_val: sys::napi_value) -> Result<Self> {
        let ci = unsafe { ClassInstance::<Self>::from_napi_value(env, napi_val) }?;
        let inner = ci.as_ref().inner.clone();
        Ok(Self { inner })
    }
}

impl LanguageModelMethods for JsXAILanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JsXAILanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(XAILanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(
        &'a self,
        env: Env,
        messages: Vec<JsMessage>,
        tools: Option<Vec<ToolDesc>>,
    ) -> Result<Object<'a>> {
        self.create_iterator(env, messages, tools)
    }
}

impl TryFrom<Unknown<'_>> for ArcMutexLanguageModel {
    type Error = napi::Error;

    fn try_from(value: Unknown<'_>) -> std::result::Result<Self, Self::Error> {
        if let Ok(model) =
            unsafe { JsLocalLanguageModel::from_napi_value(value.env(), value.raw()) }
        {
            model.get_inner()
        } else if let Ok(model) =
            unsafe { JsOpenAILanguageModel::from_napi_value(value.env(), value.raw()) }
        {
            model.get_inner()
        } else if let Ok(model) =
            unsafe { JsGeminiLanguageModel::from_napi_value(value.env(), value.raw()) }
        {
            model.get_inner()
        } else if let Ok(model) =
            unsafe { JsAnthropicLanguageModel::from_napi_value(value.env(), value.raw()) }
        {
            model.get_inner()
        } else if let Ok(model) =
            unsafe { JsXAILanguageModel::from_napi_value(value.env(), value.raw()) }
        {
            model.get_inner()
        } else {
            Err(napi::Error::new(
                Status::InvalidArg,
                "Invalid language model object provided",
            ))
        }
    }
}
