//! `Agent`, and the turn an `Agent` runs: [`ailoy::Agent`], reached over HTTP from a server in
//! this process.
//!
//! A promise's future must be `'static`, but a turn's stream borrows the agent. So the agent sits
//! behind an `Arc`, which each turn holds a clone of. The server runs one turn of an agent at a
//! time and refuses another while one runs.
//!
//! `Agent.fromSpec`'s `sandbox` gives the agent a `@brekkylab/virtx` `Sandbox` (see
//! [`crate::console`]), which stays the caller's: the agent runs in it over the same client, and
//! never removes it.
//!
//! `AgentRun`'s `next()` and `return()` are an async iterator's, and the package's `index.js`
//! gives the class `Symbol.asyncIterator` so `for await` takes it. napi's own async-iterator
//! support is not used because it rejects with napi's status as the `code`, and the codes a
//! caller acts on are ailoy's and virtx's.

use std::sync::Arc;

use ailoy::{
    Agent,
    agent::AgentSpec,
    message::{Message, MessageDeltaOutput, MessageOutput},
};
use futures::{StreamExt as _, stream::BoxStream};
use napi::{
    Env,
    bindgen_prelude::{Object, PromiseRaw, Unknown},
};
use napi_derive::napi;
use serde::Serialize;
use tokio::{runtime::Handle, sync::Mutex};

use crate::{
    console,
    convert::{Json, from_js, query},
    error::{self, Result, invalid, promise, thrown},
};

/// A value let go of on the runtime it was made on: what an agent or a turn's stream holds may
/// let go of itself on a task, and a garbage-collection finalizer is not on a runtime.
pub struct OnRuntime<T> {
    pub value: Option<T>,
    pub runtime: Handle,
}

impl<T> OnRuntime<T> {
    pub fn new(value: T, runtime: Handle) -> Self {
        OnRuntime {
            value: Some(value),
            runtime,
        }
    }
}

impl<T> Drop for OnRuntime<T> {
    fn drop(&mut self) {
        let _entered = self.runtime.enter();
        self.value.take();
    }
}

/// An agent's or a turn's error, as JavaScript is thrown it.
fn failed(e: ailoy::Error) -> napi::Error<String> {
    error::anyhow(e.into())
}

type Held = Arc<OnRuntime<Agent>>;

#[napi(js_name = "Agent")]
pub struct JsAgent {
    /// None once closed.
    agent: std::sync::Mutex<Option<Held>>,

    /// The runtime the agent was made on, which a turn's stream is let go of on too.
    runtime: Handle,
}

impl JsAgent {
    /// Made inside a promise's future, so the current runtime is the one to keep.
    fn new(agent: Agent) -> Self {
        let runtime = Handle::current();
        let agent = Arc::new(OnRuntime::new(agent, runtime.clone()));
        JsAgent {
            agent: std::sync::Mutex::new(Some(agent)),
            runtime,
        }
    }

    fn held(&self) -> Result<Held> {
        self.agent.lock().unwrap().clone().ok_or_else(closed)
    }

    /// The turn's stream; one that rejects at once, for an agent closed.
    fn start(&self, query: Message, mode: Mode) -> JsAgentRun {
        let stream = match self.held() {
            Ok(agent) => turn(agent, query, mode),
            Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
        };
        JsAgentRun::new(stream, self.runtime.clone())
    }
}

fn closed() -> napi::Error<String> {
    error::ailoy("this agent has been closed")
}

/// Which of the agent's two streams a turn is iterated from.
#[derive(Clone, Copy)]
enum Mode {
    Messages,
    Deltas,
}

/// What a turn yields: a message whole, or a delta of one.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Item {
    Message(MessageOutput),
    Delta(MessageDeltaOutput),
}

/// One turn's stream, holding the agent: `'static`, so a promise can hold it.
fn turn(agent: Held, query: Message, mode: Mode) -> BoxStream<'static, Result<Item>> {
    async_stream::stream! {
        let agent = agent.value.as_ref().expect("held until dropped");
        match mode {
            Mode::Messages => {
                let mut stream = std::pin::pin!(agent.run(&query));
                while let Some(output) = stream.next().await {
                    yield output.map(Item::Message).map_err(error::anyhow);
                }
            }
            Mode::Deltas => {
                let mut stream = std::pin::pin!(agent.run_stream(&query));
                while let Some(delta) = stream.next().await {
                    yield delta.map(Item::Delta).map_err(failed);
                }
            }
        }
    }
    .boxed()
}

#[napi]
impl JsAgent {
    /// An agent from a spec, the object an `AgentSpec` serializes to, with the history it
    /// starts with and the `Sandbox` it works in.
    #[napi(
        ts_args_type = "spec: AgentSpec, options?: { history?: Array<Message>; sandbox?: Sandbox }",
        ts_return_type = "Promise<Agent>"
    )]
    pub fn from_spec<'env>(
        env: &'env Env,
        spec: Unknown<'_>,
        options: Option<Object<'_>>,
    ) -> napi::Result<PromiseRaw<'env, JsAgent>> {
        let prepared = (|| -> Result<_> {
            let spec: AgentSpec = from_js(env, spec, "an agent spec")?;
            let mut history = Vec::new();
            let mut sandbox = None;
            if let Some(options) = options {
                let get = |e: napi::Error| invalid(e.reason);
                if let Some(given) = options.get::<Unknown>("history").map_err(get)? {
                    history = from_js::<Vec<Message>>(env, given, "a list of messages")?;
                }
                if let Some(given) = options.get::<Object>("sandbox").map_err(get)? {
                    sandbox = Some(console::sandbox(env, &given)?);
                }
            }
            Ok((spec, history, sandbox))
        })();
        let (spec, history, sandbox) = prepared.map_err(|e| thrown(env, e))?;
        promise(env, async move {
            let agent = Agent::from_spec(spec, history, sandbox.as_ref())
                .await
                .map_err(failed)?;
            Ok(JsAgent::new(agent))
        })
    }

    /// The conversation so far, as a list of messages, as the server keeps it.
    #[napi(ts_return_type = "Promise<Array<Message>>")]
    pub fn history<'env>(
        &self,
        env: &'env Env,
    ) -> napi::Result<PromiseRaw<'env, Json<Vec<Message>>>> {
        let agent = self.held().map_err(|e| thrown(env, e))?;
        promise(env, async move {
            let agent = agent.value.as_ref().expect("held until dropped");
            agent.history().await.map(Json).map_err(failed)
        })
    }

    /// Run one turn on `query` — a message, or a string as the user's text — and iterate what
    /// it produces: each complete message, the model's and each tool's.
    #[napi(ts_return_type = "AsyncIterableIterator<MessageOutput>")]
    pub fn run(
        &self,
        env: &Env,
        #[napi(ts_arg_type = "string | Message")] query: Unknown<'_>,
    ) -> Result<JsAgentRun> {
        Ok(self.start(self::query(env, query)?, Mode::Messages))
    }

    /// Run one turn as `run` does, but iterate the model's output as it is generated: deltas
    /// for each message the model writes, and each tool's result whole.
    #[napi(ts_return_type = "AsyncIterableIterator<MessageDeltaOutput>")]
    pub fn run_stream(
        &self,
        env: &Env,
        #[napi(ts_arg_type = "string | Message")] query: Unknown<'_>,
    ) -> Result<JsAgentRun> {
        Ok(self.start(self::query(env, query)?, Mode::Deltas))
    }

    /// Let go of the agent and its session on the server, and of its hold of its sandbox's
    /// client. A turn being iterated still ends. Closing twice is the same as closing once.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close<'env>(&self, env: &'env Env) -> napi::Result<PromiseRaw<'env, ()>> {
        let agent = self.agent.lock().unwrap().take();
        promise(env, async move {
            if let Some(agent) = agent {
                let agent = agent.value.as_ref().expect("held until dropped");
                agent.close().await.map_err(failed)?;
            }
            Ok(())
        })
    }
}

type Stream = Arc<Mutex<OnRuntime<BoxStream<'static, Result<Item>>>>>;

/// A turn, iterated with `for await`.
///
/// Nothing runs until the first `next()`, and the turn only goes as far as it has been
/// iterated: breaking out of the loop, or `return()`, ends it where it stands.
#[napi(js_name = "AgentRun")]
pub struct JsAgentRun {
    stream: Stream,
}

impl JsAgentRun {
    /// Iterate `stream`, let go of on `runtime`.
    pub fn new(stream: BoxStream<'static, Result<Item>>, runtime: Handle) -> Self {
        JsAgentRun {
            stream: Arc::new(Mutex::new(OnRuntime::new(stream, runtime))),
        }
    }
}

/// An iterator's result, `{ value, done }`.
#[derive(Serialize)]
pub struct Step {
    #[serde(skip_serializing_if = "Option::is_none")]
    value: Option<Item>,
    done: bool,
}

const DONE: Step = Step {
    value: None,
    done: true,
};

#[napi]
impl JsAgentRun {
    #[napi(
        ts_return_type = "Promise<IteratorResult<MessageOutput | MessageDeltaOutput, undefined>>"
    )]
    pub fn next<'env>(&self, env: &'env Env) -> napi::Result<PromiseRaw<'env, Json<Step>>> {
        let stream = self.stream.clone();
        promise(env, async move {
            let mut guard = stream.lock().await;
            let Some(stream) = guard.value.as_mut() else {
                return Ok(Json(DONE));
            };
            match stream.next().await {
                Some(item) => Ok(Json(Step {
                    value: Some(item?),
                    done: false,
                })),
                None => {
                    // Let go of the agent now rather than when this object is collected.
                    guard.value.take();
                    Ok(Json(DONE))
                }
            }
        })
    }

    /// End the turn where it stands, as breaking out of `for await` does.
    #[napi(
        js_name = "return",
        ts_return_type = "Promise<IteratorResult<MessageOutput | MessageDeltaOutput, undefined>>"
    )]
    pub fn finish<'env>(&self, env: &'env Env) -> napi::Result<PromiseRaw<'env, Json<Step>>> {
        let stream = self.stream.clone();
        promise(env, async move {
            stream.lock().await.value.take();
            Ok(Json(DONE))
        })
    }
}
