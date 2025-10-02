use std::sync::Arc;

use futures::{StreamExt, lock::Mutex};
use napi::{Error, JsSymbol, Result, Status, bindgen_prelude::*};
use napi_derive::napi;
use tokio::sync::mpsc;

use crate::{
    agent::Agent,
    ffi::node::{
        common::{await_future, get_or_create_runtime},
        value::{JsMessageOutput, JsPart},
    },
    model::ArcMutexLanguageModel,
    tool::ArcTool,
    value::MessageOutput,
};

#[napi(object)]
pub struct AgentRunIteratorResult {
    pub value: JsMessageOutput,
    pub done: bool,
}

#[derive(Clone)]
#[napi]
pub struct AgentRunIterator {
    rx: Arc<Mutex<mpsc::UnboundedReceiver<anyhow::Result<MessageOutput>>>>,
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
                value: output.into(),
                done: false,
            }),
            Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
            None => Ok(AgentRunIteratorResult {
                value: MessageOutput::new().into(),
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
    #[napi(constructor, ts_args_type = "lm: LanguageModel, tools?: Array<Tool>")]
    pub fn new(lm: Unknown<'_>, tools: Option<Vec<Unknown<'_>>>) -> napi::Result<JsAgent> {
        let lm: ArcMutexLanguageModel = lm.try_into()?;
        let tools = if let Some(tools) = tools {
            tools
                .into_iter()
                .map(|tool| {
                    let arc_tool: ArcTool = tool.try_into()?;
                    Ok(arc_tool.inner)
                })
                .collect::<napi::Result<Vec<_>>>()
        } else {
            Ok(vec![])
        }?;
        Ok(Self {
            inner: Arc::new(Mutex::new(Agent::new_from_arc(lm, tools))),
        })
    }

    fn create_iterator<'a>(&'a self, env: Env, parts: Vec<JsPart>) -> napi::Result<Object<'a>> {
        let inner = self.inner.clone();
        let (tx, rx) = mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();

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
    pub fn run<'a>(
        &'a self,
        env: Env,
        message: Either3<Vec<Either<JsPart, String>>, JsPart, String>,
    ) -> Result<Object<'a>> {
        match message {
            Either3::A(messages) => {
                let parts: Vec<JsPart> = messages
                    .into_iter()
                    .map(|msg| match msg {
                        Either::A(part) => part,
                        Either::B(text) => JsPart::new_text(text.clone()),
                    })
                    .collect::<Vec<JsPart>>();
                self.create_iterator(env, parts)
            }
            Either3::B(part) => self.create_iterator(env, vec![part]),
            Either3::C(text) => self.create_iterator(env, vec![JsPart::new_text(text)]),
        }
    }

    #[napi(ts_args_type = "tool: Tool")]
    pub fn add_tool(&mut self, tool: Unknown<'_>) -> napi::Result<()> {
        let tool: ArcTool = tool.try_into()?;
        await_future(async { self.inner.lock().await.add_tool(tool.inner).await })
    }

    #[napi(ts_args_type = "tools: Array<Tool>")]
    pub fn add_tools(&mut self, tools: Vec<Unknown<'_>>) -> napi::Result<()> {
        let tools = tools
            .into_iter()
            .map(|tool| {
                let arc_tool: ArcTool = tool.try_into()?;
                Ok(arc_tool.inner)
            })
            .collect::<napi::Result<Vec<_>>>()?;
        await_future(async { self.inner.lock().await.add_tools(tools).await })
    }

    #[napi]
    pub fn remove_tool(&self, tool_name: String) -> napi::Result<()> {
        await_future(async { self.inner.lock().await.remove_tool(tool_name).await })
    }

    #[napi]
    pub fn remove_tools(&self, tool_names: Vec<String>) -> napi::Result<()> {
        await_future(async { self.inner.lock().await.remove_tools(tool_names).await })
    }
}
