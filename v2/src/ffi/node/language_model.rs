use std::sync::Arc;

use futures::{StreamExt, lock::Mutex};
use napi::{
    Error, JsSymbol, Result as NapiResult, Status, bindgen_prelude::*,
    threadsafe_function::ThreadsafeFunction,
};
use napi_derive::napi;
use tokio::sync::mpsc;

use crate::{
    ffi::node::{
        cache::{JsCacheProgress, await_cache_result},
        common::get_or_create_runtime,
        value::{Message, MessageOutput},
    },
    model::{
        ArcMutexLanguageModel, LocalLanguageModel, anthropic::AnthropicLanguageModel,
        gemini::GeminiLanguageModel, openai::OpenAILanguageModel, xai::XAILanguageModel,
    },
    value::{Message as _Message, MessageOutput as _MessageOutput},
};

#[napi(object)]
pub struct LanguageModelIteratorResult {
    pub value: Option<MessageOutput>,
    pub done: bool,
}

#[derive(Clone)]
#[napi]
pub struct LanguageModelRunIterator {
    rx: Arc<Mutex<mpsc::UnboundedReceiver<std::result::Result<_MessageOutput, String>>>>,
}

#[napi]
impl LanguageModelRunIterator {
    #[napi(js_name = "[Symbol.asyncIterator]")]
    pub fn async_iterator(&self) -> &Self {
        // This is a dummy function to add typing for Symbol.asyncIterator
        self
    }

    #[napi]
    pub async unsafe fn next(&mut self) -> Result<LanguageModelIteratorResult> {
        let mut rx = self.rx.lock().await;
        match rx.recv().await {
            Some(Ok(output)) => Ok(LanguageModelIteratorResult {
                value: Some(output.into()),
                done: false,
            }),
            Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
            None => Ok(LanguageModelIteratorResult {
                value: None,
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

    fn create_iterator<'a>(&'a self, env: Env, messages: Vec<_Message>) -> NapiResult<Object<'a>> {
        let inner = self.get_inner()?;
        let (tx, rx) = mpsc::unbounded_channel();

        let rt = get_or_create_runtime();

        rt.spawn(async move {
            let mut model = inner.model.lock().await;
            let mut stream = model.run(messages, vec![]).boxed();

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

impl LanguageModelMethods for JsLocalLanguageModel {
    fn get_inner(&self) -> NapiResult<ArcMutexLanguageModel> {
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
    pub fn run<'a>(&'a self, env: Env, messages: Vec<Message>) -> Result<Object<'a>> {
        self.create_iterator(env, messages.into_iter().map(|msg| msg.into()).collect())
    }
}

#[napi(js_name = "OpenAILanguageModel")]
pub struct JSOpenAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl LanguageModelMethods for JSOpenAILanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JSOpenAILanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(OpenAILanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(&'a self, env: Env, messages: Vec<Message>) -> Result<Object<'a>> {
        self.create_iterator(env, messages.into_iter().map(|msg| msg.into()).collect())
    }
}

#[napi(js_name = "GeminiLanguageModel")]
pub struct JSGeminiLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl LanguageModelMethods for JSGeminiLanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JSGeminiLanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(GeminiLanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(&'a self, env: Env, messages: Vec<Message>) -> Result<Object<'a>> {
        self.create_iterator(env, messages.into_iter().map(|msg| msg.into()).collect())
    }
}

#[napi(js_name = "AnthropicLanguageModel")]
pub struct JSAnthropicLanguageModel {
    inner: ArcMutexLanguageModel,
}

impl LanguageModelMethods for JSAnthropicLanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JSAnthropicLanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(AnthropicLanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(&'a self, env: Env, messages: Vec<Message>) -> Result<Object<'a>> {
        self.create_iterator(env, messages.into_iter().map(|msg| msg.into()).collect())
    }
}

#[napi(js_name = "XAILanguageModel")]
pub struct JSXAILanguageModel {
    inner: ArcMutexLanguageModel,
}

impl LanguageModelMethods for JSXAILanguageModel {
    fn get_inner(&self) -> Result<ArcMutexLanguageModel> {
        Ok(self.inner.clone())
    }
}

#[napi]
impl JSXAILanguageModel {
    #[napi(constructor)]
    pub fn new(model_name: String, api_key: String) -> Result<Self> {
        Ok(Self {
            inner: ArcMutexLanguageModel::new(XAILanguageModel::new(model_name, api_key)),
        })
    }

    #[napi(ts_return_type = "LanguageModelRunIterator")]
    pub fn run<'a>(&'a self, env: Env, messages: Vec<Message>) -> Result<Object<'a>> {
        self.create_iterator(env, messages.into_iter().map(|msg| msg.into()).collect())
    }
}
