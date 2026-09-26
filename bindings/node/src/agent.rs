//! `AgentBuilder`, `Agent`, and the turn an `Agent` runs.
//!
//! # Where the agent lives
//!
//! A promise's future has to be `'static`, so it cannot borrow the object it was started
//! from — and [`Agent::run`] borrows the agent for as long as the turn's stream lives. The
//! agent is therefore behind an `Arc<Mutex<..>>`, and a turn holds the lock's *owned* guard
//! inside its stream. Which also makes turns take turns, as `&mut self` does in Rust: a second
//! `run` started while one is being iterated waits for it to finish before it begins.
//!
//! # Where the console lives
//!
//! `AgentBuilder.console` takes a cortex `ConsoleClient` and puts its slot in the agent's state —
//! the slot itself, not what is in it. The `ConsoleClient` object stays usable: its calls and the
//! agent's tools take turns on the one lock, and the agent starts and stops its backend
//! around each batch of tool calls, as the Rust agent does. `ConsoleClient.close()` ends the session
//! for both.
//!
//! # How a turn is iterated
//!
//! `run` hands back an `AgentRun`, whose `next()` and `return()` are an async iterator's; the
//! package's `index.js` gives the class its `Symbol.asyncIterator`, so `for await` takes it.
//! Not napi's own async-iterator support, because that rejects with napi's status as the
//! error's `code`, and the codes a caller acts on are ailoy's and cortex's.
//!
//! # How it ends
//!
//! Dropping an agent may drop the last hold on its console, and a console says `quit` only
//! when it is dropped on a runtime — which a garbage-collection finalizer is not on. So
//! [`OnRuntime`] keeps the handle of the runtime the agent was built on and enters it before
//! letting go, for the agent and for a turn's stream alike.

use std::sync::Arc;

use ailoy::{
    agent::{Agent, AgentBuilder, AgentSpec, AgentState, ContextManager},
    datatype::Value,
    lang_model::ResponseFormat,
    memory::Memory,
    message::{Message, MessageDeltaOutput, MessageOutput},
    tool::{ToolDesc, WebSearchEngineKind},
};
use cortex_node::console::{JsConsoleClient, promise, thrown};
use futures::{StreamExt as _, stream::BoxStream};
use napi::{
    Env,
    bindgen_prelude::{ClassInstance, Object, PromiseRaw, This, Unknown},
};
use napi_derive::napi;
use serde::Serialize;
use tokio::{runtime::Handle, sync::Mutex};

use crate::{
    convert::{Json, from_js, query},
    error::{self, Result, invalid, unsigned},
};

/// A value let go of on the runtime it was made on — see the module docs.
struct OnRuntime<T> {
    value: Option<T>,
    runtime: Handle,
}

impl<T> OnRuntime<T> {
    fn new(value: T, runtime: Handle) -> Self {
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

/// An [`AgentBuilder`], filled in place and emptied by `build()`.
///
/// In place rather than by value, as cortex's `ConsoleClientBuilder` is: the Rust builder is
/// consumed by each call and is not `Clone`, so there is exactly one of it to hand along.
/// Each method returns the same object so calls chain as they do in Rust.
#[napi(js_name = "AgentBuilder")]
pub struct JsAgentBuilder(Option<AgentBuilder>);

impl JsAgentBuilder {
    fn update<'env>(
        &mut self,
        this: This<'env>,
        f: impl FnOnce(AgentBuilder) -> Result<AgentBuilder>,
    ) -> Result<This<'env>> {
        let builder = self.0.take().ok_or_else(built)?;
        self.0 = Some(f(builder)?);
        Ok(this)
    }
}

fn built() -> napi::Error<String> {
    invalid("this AgentBuilder has already been built")
}

#[napi(object)]
pub struct ContextManagerOptions {
    pub max_input_tokens: Option<i64>,
    pub preserve_recent_turns: Option<u32>,
}

#[napi]
impl JsAgentBuilder {
    /// A builder for `model`, such as `'anthropic/claude-sonnet-5'`: a name some registered
    /// language-model provider serves.
    #[napi(constructor)]
    pub fn new(model: String) -> Self {
        JsAgentBuilder(Some(AgentBuilder::new(model)))
    }

    #[napi]
    pub fn agent_provider<'env>(&mut self, this: This<'env>, name: String) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.agent_provider(name)))
    }

    #[napi]
    pub fn instruction<'env>(
        &mut self,
        this: This<'env>,
        instruction: String,
    ) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.instruction(instruction)))
    }

    #[napi]
    pub fn tool<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "ToolDesc")] desc: Unknown<'_>,
    ) -> Result<This<'env>> {
        let desc: ToolDesc = from_js(env, desc, "a tool description")?;
        self.update(this, |b| Ok(b.tool(desc)))
    }

    #[napi]
    pub fn tools<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "Array<ToolDesc>")] descs: Unknown<'_>,
    ) -> Result<This<'env>> {
        let descs: Vec<ToolDesc> = from_js(env, descs, "a list of tool descriptions")?;
        self.update(this, |b| Ok(b.tools(descs)))
    }

    #[napi]
    pub fn system_tools<'env>(&mut self, this: This<'env>) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.system_tools()))
    }

    #[napi]
    pub fn shell_tool<'env>(&mut self, this: This<'env>) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.shell_tool()))
    }

    /// `engines` by name — `'Google'`, `'DuckDuckGo'`, … — or every engine when empty.
    #[napi]
    pub fn web_search_tool<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "Array<WebSearchEngine>")] engines: Option<Unknown<'_>>,
    ) -> Result<This<'env>> {
        let engines: Vec<WebSearchEngineKind> = match engines {
            Some(engines) => from_js(env, engines, "a list of web search engines")?,
            None => Vec::new(),
        };
        self.update(this, |b| Ok(b.web_search_tool(engines)))
    }

    #[napi]
    pub fn web_fetch_tool<'env>(&mut self, this: This<'env>) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.web_fetch_tool()))
    }

    /// A sub-agent, as the object an `AgentSpec` serializes to. It has to carry a `card`.
    #[napi]
    pub fn subagent<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "AgentSpec")] spec: Unknown<'_>,
    ) -> Result<This<'env>> {
        let spec: AgentSpec = from_js(env, spec, "an agent spec")?;
        self.update(this, |b| Ok(b.subagent(spec)))
    }

    #[napi]
    pub fn history<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "Array<Message>")] history: Unknown<'_>,
    ) -> Result<This<'env>> {
        let history: Vec<Message> = from_js(env, history, "a list of messages")?;
        self.update(this, |b| Ok(b.history(history)))
    }

    /// Run the agent's console tools in `console`'s session, which the agent then shares.
    #[napi]
    pub fn console<'env>(
        &mut self,
        this: This<'env>,
        #[napi(ts_arg_type = "ConsoleClient")] console: &JsConsoleClient,
    ) -> Result<This<'env>> {
        let slot = console.slot();
        self.update(this, |b| Ok(b.shared_console(slot)))
    }

    /// Remember into the memory store at `memfile`, a path in the console.
    #[napi]
    pub fn memory<'env>(&mut self, this: This<'env>, memfile: String) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.memory(Memory::new(memfile))))
    }

    #[napi]
    pub fn skill<'env>(&mut self, this: This<'env>, dir: String) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.skill(dir)))
    }

    #[napi]
    pub fn context_manager<'env>(
        &mut self,
        this: This<'env>,
        options: Option<ContextManagerOptions>,
    ) -> Result<This<'env>> {
        let default = ContextManager::default();
        let options = options.unwrap_or(ContextManagerOptions {
            max_input_tokens: None,
            preserve_recent_turns: None,
        });
        let cm = ContextManager {
            max_input_tokens: unsigned(options.max_input_tokens, "maxInputTokens")?
                .unwrap_or(default.max_input_tokens),
            preserve_recent_turns: options
                .preserve_recent_turns
                .map_or(default.preserve_recent_turns, |n| n as usize),
        };
        self.update(this, |b| Ok(b.context_manager(cm)))
    }

    #[napi]
    pub fn max_tokens<'env>(&mut self, this: This<'env>, max_tokens: i64) -> Result<This<'env>> {
        let max_tokens = unsigned(Some(max_tokens), "maxTokens")?.unwrap_or_default();
        self.update(this, |b| Ok(b.max_tokens(max_tokens)))
    }

    #[napi]
    pub fn temperature<'env>(&mut self, this: This<'env>, temperature: f64) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.temperature(temperature)))
    }

    #[napi]
    pub fn top_p<'env>(&mut self, this: This<'env>, top_p: f64) -> Result<This<'env>> {
        self.update(this, |b| Ok(b.top_p(top_p)))
    }

    #[napi]
    pub fn top_k<'env>(&mut self, this: This<'env>, top_k: i64) -> Result<This<'env>> {
        let top_k = unsigned(Some(top_k), "topK")?.unwrap_or_default();
        self.update(this, |b| Ok(b.top_k(top_k)))
    }

    /// Constrain replies to the JSON schema `schema`, which is checked here.
    #[napi]
    pub fn response_format<'env>(
        &mut self,
        env: &Env,
        this: This<'env>,
        #[napi(ts_arg_type = "Record<string, unknown>")] schema: Unknown<'_>,
    ) -> Result<This<'env>> {
        let schema: Value = from_js(env, schema, "a JSON schema")?;
        let fmt = ResponseFormat::json_schema(schema).map_err(|e| invalid(format!("{e:#}")))?;
        self.update(this, |b| Ok(b.response_format(fmt)))
    }

    /// Make the agent, settling with it — a promise, as `AgentBuilder::build` is a future: it
    /// reads the skills through the console.
    #[napi(ts_return_type = "Promise<Agent>")]
    pub fn build<'env>(&mut self, env: &'env Env) -> napi::Result<PromiseRaw<'env, JsAgent>> {
        let builder = self
            .0
            .take()
            .ok_or_else(built)
            .map_err(|e| thrown(env, e))?;
        promise(env, async move {
            let agent = builder.build().await.map_err(error::anyhow)?;
            Ok(JsAgent::new(agent))
        })
    }
}

type Held = Arc<Mutex<OnRuntime<Agent>>>;

#[napi(js_name = "Agent")]
pub struct JsAgent {
    agent: Held,

    /// The runtime the agent was built on, which a turn's stream is let go of on too.
    runtime: Handle,
}

impl JsAgent {
    /// Built inside a promise's future, so the current runtime is the one to keep.
    fn new(agent: Agent) -> Self {
        let runtime = Handle::current();
        JsAgent {
            agent: Arc::new(Mutex::new(OnRuntime::new(agent, runtime.clone()))),
            runtime,
        }
    }

    fn start(&self, query: Message, mode: Mode) -> JsAgentRun {
        let stream = turn(self.agent.clone(), query, mode);
        JsAgentRun {
            stream: Arc::new(Mutex::new(OnRuntime::new(stream, self.runtime.clone()))),
        }
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
enum Item {
    Message(MessageOutput),
    Delta(MessageDeltaOutput),
}

/// One turn's stream, from the agent's owned guard: `'static`, so a promise can hold it.
fn turn(agent: Held, query: Message, mode: Mode) -> BoxStream<'static, Result<Item>> {
    async_stream::stream! {
        let mut guard = agent.lock_owned().await;
        let Some(agent) = guard.value.as_mut() else {
            yield Err(closed());
            return;
        };
        match mode {
            Mode::Messages => {
                let mut stream = agent.run(query);
                while let Some(output) = stream.next().await {
                    yield output.map(Item::Message).map_err(error::anyhow);
                }
            }
            Mode::Deltas => {
                let mut stream = agent.run_stream(query);
                while let Some(delta) = stream.next().await {
                    yield delta.map(Item::Delta).map_err(error::anyhow);
                }
            }
        }
    }
    .boxed()
}

#[napi]
impl JsAgent {
    /// An agent from a spec — the object an `AgentSpec` serializes to — for a caller who holds
    /// one rather than building it up.
    ///
    /// `options` may name the `agentProvider` (`'default'` unless it does), the `history` to
    /// start from, the `console` to share and the `memory` file to remember into.
    #[napi(
        ts_args_type = "spec: AgentSpec, options?: { agentProvider?: string; history?: Array<Message>; console?: ConsoleClient; memory?: string }",
        ts_return_type = "Promise<Agent>"
    )]
    pub fn from_spec<'env>(
        env: &'env Env,
        spec: Unknown<'_>,
        options: Option<Object<'_>>,
    ) -> napi::Result<PromiseRaw<'env, JsAgent>> {
        let prepared = (|| -> Result<_> {
            let spec: AgentSpec = from_js(env, spec, "an agent spec")?;
            let mut state = AgentState::new();
            let mut agent_provider = "default".to_string();
            if let Some(options) = options {
                let get = |e: napi::Error| invalid(e.reason);
                if let Some(name) = options.get::<String>("agentProvider").map_err(get)? {
                    agent_provider = name;
                }
                if let Some(history) = options.get::<Unknown>("history").map_err(get)? {
                    state = state.with_history(from_js::<Vec<Message>>(
                        env,
                        history,
                        "a list of messages",
                    )?);
                }
                if let Some(console) = options
                    .get::<ClassInstance<JsConsoleClient>>("console")
                    .map_err(get)?
                {
                    state = state.with_console_slot(console.slot());
                }
                if let Some(memfile) = options.get::<String>("memory").map_err(get)? {
                    state = state.with_memory(Memory::new(memfile));
                }
            }
            Ok((spec, agent_provider, state))
        })();
        let (spec, agent_provider, state) = prepared.map_err(|e| thrown(env, e))?;
        promise(env, async move {
            let agent = Agent::try_with_provider_and_state(spec, &agent_provider, state)
                .await
                .map_err(error::anyhow)?;
            Ok(JsAgent::new(agent))
        })
    }

    /// The conversation so far, as a list of messages.
    ///
    /// Read without waiting, so it throws while a turn holds the agent rather than making
    /// this getter a promise.
    #[napi(getter, ts_return_type = "Array<Message>")]
    pub fn history(&self) -> Result<Json<Vec<Message>>> {
        let guard = self
            .agent
            .try_lock()
            .map_err(|_| error::ailoy("this agent is in the middle of a turn"))?;
        let agent = guard.value.as_ref().ok_or_else(closed)?;
        Ok(Json(agent.get_history().to_vec()))
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

    /// Let go of the agent now, and of its console if nothing else holds it. Waits for a turn
    /// being iterated to finish. Closing twice is the same as closing once.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn close<'env>(&self, env: &'env Env) -> napi::Result<PromiseRaw<'env, ()>> {
        let agent = self.agent.clone();
        promise(env, async move {
            // Dropped here, on the runtime.
            agent.lock().await.value.take();
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
