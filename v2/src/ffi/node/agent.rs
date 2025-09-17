use std::sync::Arc;

use futures::{StreamExt, lock::Mutex};
use napi::{Error, JsSymbol, Result, Status, bindgen_prelude::*};
use napi_derive::napi;
use tokio::sync::mpsc;

use crate::{
    agent::Agent,
    ffi::node::{
        common::get_or_create_runtime,
        language_model::{
            JsAnthropicLanguageModel, JsGeminiLanguageModel, JsLocalLanguageModel,
            JsOpenAILanguageModel, JsXAILanguageModel, LanguageModelMethods,
        },
        value::{JsMessageOutput, JsPart},
    },
    value::MessageOutput,
};

#[napi(object)]
pub struct AgentRunIteratorResult {
    pub value: Option<JsMessageOutput>,
    pub done: bool,
}

#[derive(Clone)]
#[napi]
pub struct AgentRunIterator {
    rx: Arc<Mutex<mpsc::UnboundedReceiver<std::result::Result<MessageOutput, String>>>>,
}

#[napi]
impl AgentRunIterator {
    #[napi(js_name = "[Symbol.asyncIterator]")]
    pub fn async_iterator(&self) -> &Self {
        // This is a dummy function to add typing for Symbol.asyncIterator
        self
    }

    #[napi]
    pub async unsafe fn next(&mut self) -> napi::Result<AgentRunIteratorResult> {
        let mut rx = self.rx.lock().await;
        match rx.recv().await {
            Some(Ok(output)) => Ok(AgentRunIteratorResult {
                value: Some(output.into()),
                done: false,
            }),
            Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
            None => Ok(AgentRunIteratorResult {
                value: None,
                done: true,
            }),
        }
    }
}

impl AgentRunIterator {
    /// This returns an object with \[Symbol.asyncIterator\], which is not directly injected by napi-rs.
    fn to_async_iterator<'a>(self, env: Env) -> napi::Result<Object<'a>> {
        let mut obj = Object::new(&env)?;

        let global = env.get_global()?;
        let symbol: Function = global.get_named_property("Symbol")?;
        let symbol_async_iterator: JsSymbol = symbol.get_named_property("asyncIterator")?;

        let func: Function<(), AgentRunIterator> =
            env.create_function_from_closure("asyncIterator", move |_| Ok(self.clone()))?;

        obj.set_property(symbol_async_iterator, func)?;

        Ok(obj)
    }
}

#[napi(js_name = "Agent")]
pub struct JsAgent {
    inner: Arc<Mutex<Agent>>,
}

#[napi]
impl JsAgent {
    #[napi(
        constructor,
        ts_args_type = "lm: LocalLanguageModel | OpenAILanguageModel | GeminiLanguageModel | AnthropicLanguageModel | XAILanguageModel"
    )]
    pub fn new(lm: Unknown<'_>) -> napi::Result<JsAgent> {
        let agent = if let Ok(model) =
            unsafe { JsLocalLanguageModel::from_napi_value(lm.env(), lm.raw()) }
        {
            Ok(Agent::new_from_arc(model.get_inner()?, vec![]))
        } else if let Ok(model) =
            unsafe { JsOpenAILanguageModel::from_napi_value(lm.env(), lm.raw()) }
        {
            Ok(Agent::new_from_arc(model.get_inner()?, vec![]))
        } else if let Ok(model) =
            unsafe { JsGeminiLanguageModel::from_napi_value(lm.env(), lm.raw()) }
        {
            Ok(Agent::new_from_arc(model.get_inner()?, vec![]))
        } else if let Ok(model) =
            unsafe { JsAnthropicLanguageModel::from_napi_value(lm.env(), lm.raw()) }
        {
            Ok(Agent::new_from_arc(model.get_inner()?, vec![]))
        } else if let Ok(model) = unsafe { JsXAILanguageModel::from_napi_value(lm.env(), lm.raw()) }
        {
            Ok(Agent::new_from_arc(model.get_inner()?, vec![]))
        } else {
            Err(Error::new(Status::InvalidArg, "Invalid lm provided"))
        }?;

        Ok(Self {
            inner: Arc::new(Mutex::new(agent)),
        })
    }

    fn create_iterator<'a>(&'a self, env: Env, parts: Vec<JsPart>) -> napi::Result<Object<'a>> {
        let inner = self.inner.clone();
        let (tx, rx) = mpsc::unbounded_channel::<std::result::Result<MessageOutput, String>>();

        let rt = get_or_create_runtime();

        rt.spawn(async move {
            let mut agent = inner.lock().await;
            let mut stream = agent
                .run(parts.into_iter().map(|p| p.into()).collect())
                .boxed();

            while let Some(item) = stream.next().await {
                if tx.send(item).is_err() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        });

        let it = AgentRunIterator {
            rx: Arc::new(Mutex::new(rx)),
        };
        it.to_async_iterator(env)
    }

    #[napi(ts_return_type = "AgentRunIterator")]
    pub fn run<'a>(&'a self, env: Env, parts: Vec<JsPart>) -> Result<Object<'a>> {
        self.create_iterator(env, parts)
    }
}
