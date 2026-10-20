use std::{collections::HashMap, panic::AssertUnwindSafe, path::Path, sync::Arc};

use anyhow::Context as _;
use futures::{
    FutureExt as _, StreamExt as _,
    stream::{self, BoxStream},
};
use tokio::sync::Mutex;
use virtx::console::ConsoleClient;

use super::{
    AgentSpec, AgentState, ContextManager, get_agent_providers, skill::render_skills,
    spec::split_desc,
};
use crate::{
    datatype::Value,
    message::{
        Assembler, Delta as _, FinishReason, Message, MessageDelta, MessageDeltaOutput,
        MessageOutput, Part, Role, cap_tool_delta,
    },
    model::{InferLangModel, LangModelOptions, create_lang_model},
    tool::{
        ToolDesc, ToolFunc, get_tool_providers,
        r#impl::{
            get_mem_insert_tool_desc, get_mem_insert_tool_func, get_mem_search_tool_desc,
            get_mem_search_tool_func, get_web_search_tool_factory,
        },
    },
};

/// An agent that drives an [`InferLangModel`] through multi-turn, tool-augmented
/// conversations.
///
/// Pairs an [`AgentSpec`] (model + instruction + tools) with an
/// [`AgentProvider`](super::AgentProvider) (tool sources) and an [`AgentState`]
/// (history + console). [`run`](Self::run) streams one turn, calling tools until the model
/// answers, and appends every message to the history.
pub struct Agent {
    model: Arc<dyn InferLangModel>,

    options: LangModelOptions,

    tool_descs: Vec<ToolDesc>,

    tools: HashMap<String, ToolFunc>,

    pub state: AgentState,

    /// Cuts old tool results out of the history once a call's input grows past its limit.
    context_manager: Option<ContextManager>,
}

impl Agent {
    /// Create an agent with the `"default"` [`AgentProvider`](super::AgentProvider) and a
    /// fresh [`AgentState`].
    pub async fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", AgentState::new()).await
    }

    /// Create an agent with the [`AgentProvider`](super::AgentProvider) registered under
    /// `provider` and a fresh [`AgentState`].
    pub async fn try_with_provider(
        spec: AgentSpec,
        provider: impl AsRef<str>,
    ) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, provider, AgentState::new()).await
    }

    /// Create an agent with the `"default"` [`AgentProvider`](super::AgentProvider) and
    /// `state`.
    pub async fn try_with_state(spec: AgentSpec, state: AgentState) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", state).await
    }

    /// Create an agent with the [`AgentProvider`](super::AgentProvider) registered under
    /// `provider` and `state`.
    ///
    /// The model is built from `spec.model` through
    /// [`create_lang_model`]. Unless `state.history` already has a [`Role::System`]
    /// message, one built from `spec.instruction` and [`AgentSpec::skills`] is put first.
    ///
    /// Async because skills are read through the console.
    pub async fn try_with_provider_and_state(
        spec: AgentSpec,
        provider: impl AsRef<str>,
        mut state: AgentState,
    ) -> anyhow::Result<Self> {
        let provider = provider.as_ref();
        let provider = get_agent_providers()
            .get(provider)
            .cloned()
            .with_context(|| format!("agent_provider '{provider}' not registered"))?;

        let model = create_lang_model(&spec.model).await?;

        // Errors if any spec tool is missing.
        let mut tools = {
            let registry = get_tool_providers();
            let tool_provider = registry.get(&provider.tool_provider).with_context(|| {
                format!("tool_provider '{}' not registered", provider.tool_provider)
            })?;
            match &spec.web_search_engines {
                Some(engines) => {
                    let mut tool_provider = tool_provider.clone();
                    tool_provider.insert_func_factory(
                        "web_search",
                        get_web_search_tool_factory(engines.clone()),
                    );
                    tool_provider.provide(&spec.tools)?
                }
                None => tool_provider.provide(&spec.tools)?,
            }
        };
        let mut tool_descs = spec.tools.clone();

        // The store is this agent's `Memory`, not a registry name, so these are built here.
        if let Some(memory) = state.memory.clone() {
            for (desc, func) in [
                (
                    get_mem_search_tool_desc(),
                    get_mem_search_tool_func(memory.clone()),
                ),
                (get_mem_insert_tool_desc(), get_mem_insert_tool_func(memory)),
            ] {
                tools.insert(desc.name.clone(), func);
                tool_descs.push(desc);
            }
        }

        if !state.history.iter().any(|m| m.role == Role::System) {
            let skills = if spec.skills.is_empty() {
                None
            } else {
                let org = split_desc(&spec.model).0;
                Some(render_skills(&spec.skills, &state.console, org).await?)
            };
            let text = [spec.instruction, skills]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("\n\n");
            if !text.is_empty() {
                state.history.insert(
                    0,
                    Message::new(Role::System).with_contents([Part::text(text)]),
                );
            }
        }

        Ok(Self {
            model,
            options: spec.model_options.unwrap_or_default(),
            tool_descs,
            tools,
            state,
            context_manager: None,
        })
    }

    /// The whole conversation so far.
    pub fn history(&self) -> &[Message] {
        &self.state.history
    }

    pub fn context_manager(&self) -> Option<&ContextManager> {
        self.context_manager.as_ref()
    }

    pub fn set_context_manager(&mut self, context_manager: Option<ContextManager>) {
        self.context_manager = context_manager;
    }

    /// Truncate the history if the last call's input went past the context manager's limit.
    fn manage_context(&mut self) {
        if let Some(cm) = &self.context_manager
            && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens
        {
            cm.truncate_history(&mut self.state.history);
        }
    }

    /// Commit the model's `output` to the history, noting its input size for
    /// [`manage_context`](Self::manage_context).
    fn commit(&mut self, output: &MessageOutput) {
        if let Some(usage) = &output.usage {
            self.state.last_input_tokens = Some(usage.input_tokens);
        }
        self.state.history.push(output.message.clone());
    }

    /// Run one turn, yielding each model message and tool result as it completes.
    ///
    /// If the model fails before its first message of the turn, `query` is taken back out
    /// of the history, so the agent can be run again.
    pub fn run(&mut self, query: Message) -> BoxStream<'_, anyhow::Result<MessageOutput>> {
        Box::pin(async_stream::try_stream! {
            self.begin_turn(query).await;
            let mut committed = false;
            loop {
                self.manage_context();
                let output = match self
                    .model
                    .infer(&self.state.history, &self.tool_descs, &self.options)
                    .await
                {
                    Ok(output) => output,
                    Err(e) => {
                        if !committed {
                            self.state.history.pop();
                        }
                        Err(e)?
                    }
                };
                self.commit(&output);
                committed = true;

                let calls = tool_calls(&output);
                yield output;
                if calls.is_empty() {
                    break;
                }

                let mut results = self.call_tools(calls);
                let mut assembler = Assembler::default();
                while let Some(delta) = results.next().await {
                    if let Some(output) = assembler.push(delta?)? {
                        self.state.history.push(output.message.clone());
                        yield output;
                    }
                }
            }
        })
    }

    /// Like [`run`](Self::run), but yields each message as deltas while it is produced:
    /// the model's as it is generated, and each tool result as its tool yields it. A
    /// `finish_reason` marks the end of a message.
    ///
    /// The calls of one batch run concurrently, so their results' deltas interleave; each
    /// carries its call's id. A tool result's deltas are clamped one by one, and the result
    /// they make up is clamped again before it goes into the history.
    pub fn run_stream(
        &mut self,
        query: Message,
    ) -> BoxStream<'_, anyhow::Result<MessageDeltaOutput>> {
        Box::pin(async_stream::try_stream! {
            self.begin_turn(query).await;
            let mut committed = false;
            loop {
                self.manage_context();
                let mut deltas =
                    self.model
                        .infer_stream(&self.state.history, &self.tool_descs, &self.options);
                let mut acc = MessageDeltaOutput::new();
                let mut failure = None;
                while let Some(delta) = deltas.next().await {
                    let step = delta.and_then(|delta| {
                        let prev = std::mem::replace(&mut acc, MessageDeltaOutput::new());
                        acc = prev.accumulate(delta.clone())?;
                        Ok(delta)
                    });
                    match step {
                        Ok(delta) => yield delta,
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                let output = match failure.map_or_else(|| acc.finish(), Err) {
                    Ok(output) => output,
                    Err(e) => {
                        if !committed {
                            self.state.history.pop();
                        }
                        Err(e)?
                    }
                };
                self.commit(&output);
                committed = true;

                // Already streamed as deltas.
                let calls = tool_calls(&output);
                if calls.is_empty() {
                    break;
                }

                let mut results = self.call_tools(calls);
                let mut assembler = Assembler::default();
                while let Some(delta) = results.next().await {
                    let delta = delta?;
                    if let Some(output) = assembler.push(delta.clone())? {
                        self.state.history.push(output.message);
                    }
                    yield delta;
                }
            }
        })
    }

    /// Add `query` to the history and tell the model where its console mounts are.
    async fn begin_turn(&mut self, query: Message) {
        close_dangling_calls(&mut self.state.history);
        self.state.history.push(query);
        self.seed_mounts().await;
    }

    /// List the console's mounts in the system message, read afresh each turn.
    ///
    /// Only paths are listed; what a mount is for belongs in the instruction. The section
    /// goes into a caller-written system message too, since mounts are not known ahead of
    /// time, and only once: it is its own marker.
    async fn seed_mounts(&mut self) {
        let mounts: Vec<std::path::PathBuf> = match self.state.console.lock().await.as_ref() {
            Some(console) => console.mounts().map(Path::to_path_buf).collect(),
            None => return,
        };
        if mounts.is_empty() {
            return;
        }

        let mut section = String::from(
            "# Mounts\n\nThe directories you have been given, in the order they were mounted:\n",
        );
        for path in &mounts {
            section.push_str(&format!("\n- `{}`", path.display()));
        }

        let Some(system) = self
            .state
            .history
            .iter_mut()
            .find(|m| m.role == Role::System)
        else {
            self.state.history.insert(
                0,
                Message::new(Role::System).with_contents([Part::text(section)]),
            );
            return;
        };
        // Into the first text part, which every model reads as the system prompt.
        let text = system.contents.iter_mut().find_map(|p| match p {
            Part::Text { text } => Some(text),
            _ => None,
        });
        match text {
            Some(text) if text.contains(&section) => {}
            Some(text) => {
                text.push_str("\n\n");
                text.push_str(&section);
            }
            None => system.contents.insert(0, Part::text(section)),
        }
    }

    /// Run `calls` concurrently, yielding each one's [`Role::Tool`] result as deltas while
    /// its tool produces them.
    ///
    /// The console is started for the batch only if a console tool is in it, and stopped
    /// after, so its backend (possibly a whole micro-VM) is not kept up while the model
    /// thinks. Stopping keeps the session, so the next batch resumes the same console.
    fn call_tools(
        &self,
        calls: Vec<Part>,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let calls: Vec<_> = calls
            .iter()
            .filter_map(Part::as_function)
            .map(|(id, name, args)| {
                (
                    id.to_owned(),
                    name.to_owned(),
                    args.clone(),
                    self.tools.get(name).cloned(),
                )
            })
            .collect();
        let needs_console = calls
            .iter()
            .any(|(.., tool)| tool.as_ref().is_some_and(ToolFunc::needs_console));
        let console = self.state.console.clone();

        Box::pin(async_stream::try_stream! {
            if needs_console && let Some(console) = console.lock().await.as_mut() {
                console.start().await?;
            }

            let mut pending = stream::select_all(
                calls
                    .into_iter()
                    .map(|(id, name, args, tool)| call_tool(id, name, args, tool, console.clone())),
            );
            while let Some(delta) = pending.next().await {
                yield delta;
            }

            if needs_console && let Some(console) = console.lock().await.as_mut() {
                console.stop().await?;
            }
        })
    }
}

/// The tool calls `output` stops at; empty when the model has answered.
fn tool_calls(output: &MessageOutput) -> Vec<Part> {
    match output.finish_reason {
        FinishReason::ToolCall {} => output.message.tool_calls.clone().unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Answer every call of the last assistant message that has no result yet, as when the
/// caller dropped the stream mid-turn or the console failed to start. Without this the
/// next request carries calls without results, which APIs reject (Claude with a 400).
fn close_dangling_calls(history: &mut Vec<Message>) {
    let Some(at) = history.iter().rposition(|m| m.role == Role::Assistant) else {
        return;
    };
    let answered = history[at + 1..]
        .iter()
        .take_while(|m| m.role == Role::Tool)
        .count();
    let results: Vec<Message> = history[at]
        .tool_calls
        .iter()
        .flatten()
        .filter_map(Part::as_function)
        .map(|(id, ..)| id)
        .filter(|id| {
            !history[at + 1..at + 1 + answered]
                .iter()
                .any(|m| m.id.as_deref() == Some(*id))
        })
        .map(|id| tool_error(id.to_owned(), "the call was interrupted before it returned"))
        .collect();
    let end = at + 1 + answered;
    history.splice(end..end, results);
}

fn tool_error(id: String, text: impl Into<String>) -> Message {
    Message::new(Role::Tool)
        .with_contents([Part::text(text)])
        .with_id(id)
}

/// Run one tool call, yielding its result as deltas as the tool produces them; the last
/// carries `finish_reason`.
///
/// Each item the tool's stream yields is a piece of the result, sent on as soon as it is
/// yielded. Whatever goes wrong (an unknown tool, a missing console, a panic) becomes part
/// of the result, so the model gets exactly one per call and can recover.
fn call_tool(
    id: String,
    name: String,
    args: Value,
    tool: Option<ToolFunc>,
    console: Arc<Mutex<Option<ConsoleClient>>>,
) -> BoxStream<'static, MessageDeltaOutput> {
    Box::pin(async_stream::stream! {
        let Some(tool) = tool else {
            yield tool_delta(&id, [Part::text(format!("there is no tool named '{name}'"))], true);
            return;
        };

        let mut guard = None;
        let pieces = if tool.needs_console() {
            let Some(console) = guard.insert(console.lock().await).as_mut() else {
                let text = format!(
                    "tool '{name}' failed: it needs a console, and the agent has none; \
                     give it one with `AgentState::with_console`"
                );
                yield tool_delta(&id, [Part::text(text)], true);
                return;
            };
            std::panic::catch_unwind(AssertUnwindSafe(|| tool.call(args, id.clone(), console)))
        } else {
            std::panic::catch_unwind(AssertUnwindSafe(|| tool.call_pure(args, id.clone())))
                .map(|pieces| pieces.expect("a tool that needs no console is called pure"))
        };
        let Ok(pieces) = pieces else {
            yield tool_delta(&id, [Part::text(format!("tool '{name}' panicked"))], true);
            return;
        };

        let mut pieces = AssertUnwindSafe(pieces).catch_unwind();
        let mut next = pieces.next().await;
        let mut produced = false;
        loop {
            let parts = match next {
                Some(Ok(output)) => output.message.contents,
                Some(Err(_)) => {
                    let text = format!("{}tool '{name}' panicked", if produced { "\n\n" } else { "" });
                    yield tool_delta(&id, [Part::text(text)], true);
                    return;
                }
                None if produced => {
                    yield tool_delta(&id, [], true);
                    return;
                }
                None => {
                    let text = format!("tool '{name}' produced no output");
                    yield tool_delta(&id, [Part::text(text)], true);
                    return;
                }
            };
            produced = true;
            // A piece the tool has already ended after goes out as the finished result,
            // so a tool that returns once streams one complete delta.
            next = match pieces.next().now_or_never() {
                Some(None) => {
                    yield tool_delta(&id, parts, true);
                    return;
                }
                Some(after) => {
                    yield tool_delta(&id, parts, false);
                    after
                }
                None => {
                    yield tool_delta(&id, parts, false);
                    pieces.next().await
                }
            };
        }
    })
}

/// A piece of call `id`'s result, clamped; `done` when it ends the result.
fn tool_delta(id: &str, parts: impl IntoIterator<Item = Part>, done: bool) -> MessageDeltaOutput {
    MessageDeltaOutput {
        delta: cap_tool_delta(
            MessageDelta::new()
                .with_role(Role::Tool)
                .with_id(id)
                .with_contents(parts),
        ),
        finish_reason: done.then_some(FinishReason::Stop {}),
        usage: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex as StdMutex};

    use futures::{
        future::BoxFuture,
        stream::{self, BoxStream},
    };

    use super::*;
    use crate::{
        agent::{AgentProvider, get_agent_providers_mut},
        message::middle_truncate,
        message::{Role, TokenUsage},
        model::{LangModelProvider, set_lang_model_provider},
        suppress_panics, to_value,
        tool::{ToolDescBuilder, ToolProvider, get_tool_providers_mut},
        tool_func,
    };

    /// Answers with `outputs` in order, and keeps every history it was given.
    #[derive(Default)]
    struct ScriptedModel {
        outputs: StdMutex<VecDeque<anyhow::Result<MessageOutput>>>,
        seen: StdMutex<Vec<Vec<Message>>>,
    }

    impl ScriptedModel {
        fn next(&self, messages: &[Message]) -> anyhow::Result<MessageOutput> {
            self.seen.lock().unwrap().push(messages.to_vec());
            self.outputs
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(anyhow::anyhow!("the script has run out")))
        }
    }

    impl InferLangModel for ScriptedModel {
        fn infer(
            &self,
            messages: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
            let output = self.next(messages);
            Box::pin(async move { output })
        }

        fn infer_stream(
            &self,
            messages: &[Message],
            _: &[ToolDesc],
            _: &LangModelOptions,
        ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
            let output = self.next(messages).map(MessageDeltaOutput::from);
            Box::pin(stream::once(async move { output }))
        }
    }

    fn answer(text: &str) -> anyhow::Result<MessageOutput> {
        Ok(MessageOutput {
            message: Message::new(Role::Assistant).with_contents([Part::text(text)]),
            finish_reason: FinishReason::Stop {},
            usage: None,
        })
    }

    fn calls(calls: &[(&str, &str)]) -> anyhow::Result<MessageOutput> {
        Ok(MessageOutput {
            message: Message::new(Role::Assistant).with_tool_calls(
                calls
                    .iter()
                    .map(|(id, name)| Part::function(*id, *name, to_value!({}))),
            ),
            finish_reason: FinishReason::ToolCall {},
            usage: None,
        })
    }

    /// Registers `script` as the model `name` and returns its spec, with the `echo`,
    /// `boom` (panics) and `console_only` tools available through the agent provider
    /// `name`.
    fn scripted(
        name: &str,
        script: impl IntoIterator<Item = anyhow::Result<MessageOutput>>,
    ) -> (Arc<ScriptedModel>, AgentSpec) {
        let model = Arc::new(ScriptedModel {
            outputs: StdMutex::new(script.into_iter().collect()),
            ..Default::default()
        });
        let shared = model.clone();
        set_lang_model_provider(
            name,
            LangModelProvider::custom(move |_, _| Ok(shared.clone())),
        );

        let mut tp = ToolProvider::new();
        tp.insert_func(
            "echo",
            tool_func!(|_args: Value| -> Value { Value::string("echoed") }),
        );
        tp.insert_func(
            "boom",
            tool_func!(|_args: Value| -> Value {
                if true {
                    panic!("boom");
                }
                Value::Null
            }),
        );
        tp.insert_func(
            "console_only",
            tool_func!(async |_args: Value, _console: &mut ConsoleClient| -> Value { Value::Null }),
        );
        get_tool_providers_mut().insert(name.to_owned(), tp);
        get_agent_providers_mut().insert(name.to_owned(), AgentProvider::new(name));

        let tool = |name: &str| {
            ToolDescBuilder::new(name)
                .parameters(to_value!({"type": "object", "properties": {}}))
                .build()
        };
        let spec = AgentSpec::new(format!("{name}/any"))
            .instruction("Be brief.")
            .tools([tool("echo"), tool("boom"), tool("console_only")]);
        (model, spec)
    }

    fn user(text: &str) -> Message {
        Message::new(Role::User).with_contents([Part::text(text)])
    }

    fn roles(history: &[Message]) -> Vec<Role> {
        history.iter().map(|m| m.role.clone()).collect()
    }

    fn text_of(message: &Message) -> String {
        message
            .contents
            .iter()
            .filter_map(|p| p.as_text())
            .collect()
    }

    #[tokio::test]
    async fn test_run_calls_tools_until_the_model_answers() {
        let name = "test-agent-run";
        let (model, spec) = scripted(name, [calls(&[("c1", "echo")]), answer("done")]);
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();

        let outputs: Vec<_> = agent.run(user("hi")).map(Result::unwrap).collect().await;
        assert_eq!(
            roles(
                &outputs
                    .iter()
                    .map(|o| o.message.clone())
                    .collect::<Vec<_>>()
            ),
            [Role::Assistant, Role::Tool, Role::Assistant]
        );
        assert_eq!(outputs[1].message.id.as_deref(), Some("c1"));
        assert_eq!(
            roles(agent.history()),
            [
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::Assistant
            ]
        );
        assert_eq!(text_of(&agent.history()[0]), "Be brief.");
        // The second call sees the tool result.
        assert_eq!(model.seen.lock().unwrap()[1].len(), 4);
    }

    #[tokio::test]
    async fn test_run_stream_matches_run() {
        let name = "test-agent-run-stream";
        let (_, spec) = scripted(name, [calls(&[("c1", "echo")]), answer("done")]);
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();

        let deltas: Vec<_> = agent
            .run_stream(user("hi"))
            .map(Result::unwrap)
            .collect()
            .await;
        assert_eq!(deltas.len(), 3);
        assert_eq!(
            roles(agent.history()),
            [
                Role::System,
                Role::User,
                Role::Assistant,
                Role::Tool,
                Role::Assistant
            ]
        );
        assert_eq!(text_of(agent.history().last().unwrap()), "done");
    }

    /// Registers `func` as the tool `tool` in the provider [`scripted`] made for `name`, and
    /// adds it to `spec`.
    fn with_tool(name: &str, spec: AgentSpec, tool: &str, func: ToolFunc) -> AgentSpec {
        get_tool_providers_mut()
            .get_mut(name)
            .unwrap()
            .insert_func(tool, func);
        spec.tool(
            ToolDescBuilder::new(tool)
                .parameters(to_value!({"type": "object", "properties": {}}))
                .build(),
        )
    }

    fn piece(id: &str, text: &str) -> MessageOutput {
        MessageOutput {
            message: Message::new(Role::Tool)
                .with_contents([Part::text(text)])
                .with_id(id),
            finish_reason: FinishReason::Stop {},
            usage: None,
        }
    }

    fn delta_text(delta: &MessageDeltaOutput) -> String {
        delta
            .delta
            .contents
            .iter()
            .map(|p| match p {
                crate::message::PartDelta::Text { text } => text.as_str(),
                _ => "",
            })
            .collect()
    }

    /// Tool messages, as JSON so they compare.
    fn tool_results<'a>(messages: impl IntoIterator<Item = &'a Message>) -> Vec<Value> {
        messages
            .into_iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| to_value!(serde_json::to_value(m).unwrap()))
            .collect()
    }

    #[tokio::test]
    async fn test_a_tool_piece_arrives_before_the_next_exists() {
        let name = "test-agent-tool-stream";
        let (_, spec) = scripted(name, [calls(&[("c1", "gated")]), answer("done")]);
        let gate = Arc::new(tokio::sync::Notify::new());
        let opener = gate.clone();
        let spec = with_tool(
            name,
            spec,
            "gated",
            ToolFunc::new(move |_, id| {
                let gate = gate.clone();
                Box::pin(async_stream::stream! {
                    yield piece(&id, "first ");
                    gate.notified().await;
                    yield piece(&id, "second");
                })
            }),
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();

        let mut deltas = agent.run_stream(user("hi"));
        deltas.next().await.unwrap().unwrap(); // the model's call
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), deltas.next())
            .await
            .expect("the first piece is held back until the second exists")
            .unwrap()
            .unwrap();
        assert_eq!(first.delta.role, Some(Role::Tool));
        assert_eq!(first.delta.id.as_deref(), Some("c1"));
        assert_eq!(first.finish_reason, None);
        assert_eq!(delta_text(&first), "first ");

        opener.notify_one();
        let rest: Vec<_> = deltas.map(Result::unwrap).collect().await;
        assert_eq!(delta_text(&rest[0]), "second");
        let result = agent
            .history()
            .iter()
            .find(|m| m.role == Role::Tool)
            .unwrap();
        assert_eq!(text_of(result), "first second");
    }

    #[tokio::test]
    async fn test_a_tool_that_panics_mid_stream_keeps_what_it_streamed() {
        suppress_panics!();
        let name = "test-agent-tool-stream-panic";
        let (_, spec) = scripted(
            name,
            [calls(&[("c1", "shaky"), ("c2", "silent")]), answer("done")],
        );
        let spec = with_tool(
            name,
            spec,
            "shaky",
            ToolFunc::new(|_, id| {
                Box::pin(async_stream::stream! {
                    yield piece(&id, "partial");
                    tokio::task::yield_now().await;
                    if true {
                        panic!("mid-stream");
                    }
                    yield piece(&id, "never");
                })
            }),
        );
        let spec = with_tool(
            name,
            spec,
            "silent",
            ToolFunc::new(|_, _| Box::pin(stream::empty())),
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();
        agent.run(user("hi")).for_each(|_| async {}).await;

        let results: HashMap<_, _> = agent
            .history()
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| (m.id.clone().unwrap(), text_of(m)))
            .collect();
        assert_eq!(results["c1"], "partial\n\ntool 'shaky' panicked");
        assert_eq!(results["c2"], "tool 'silent' produced no output");
        assert_eq!(text_of(agent.history().last().unwrap()), "done");
    }

    #[tokio::test]
    async fn test_concurrent_calls_stream_into_their_own_results() {
        let name = "test-agent-tool-stream-concurrent";
        let (_, spec) = scripted(
            name,
            [calls(&[("c1", "count"), ("c2", "count")]), answer("done")],
        );
        let spec = with_tool(
            name,
            spec,
            "count",
            ToolFunc::new(|_, id| {
                Box::pin(async_stream::stream! {
                    for n in ["1", "2", "3"] {
                        yield piece(&id, n);
                        tokio::task::yield_now().await;
                    }
                })
            }),
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();

        let deltas: Vec<_> = agent
            .run_stream(user("hi"))
            .map(Result::unwrap)
            .collect()
            .await;
        let assembled: Vec<_> =
            crate::message::into_messages(stream::iter(deltas.into_iter().map(anyhow::Ok)))
                .map(|o| o.unwrap().message)
                .collect()
                .await;

        let results = tool_results(&assembled);
        assert_eq!(results.len(), 2);
        assert_eq!(results, tool_results(agent.history()));
        for result in agent.history().iter().filter(|m| m.role == Role::Tool) {
            assert_eq!(text_of(result), "123");
        }
    }

    #[tokio::test]
    async fn test_a_streamed_result_is_clamped_piece_by_piece_and_whole() {
        let name = "test-agent-tool-stream-clamp";
        let script = || [calls(&[("c1", "flood")]), answer("done")];
        let (model, spec) = scripted(name, script().into_iter().chain(script()));
        let spec = with_tool(
            name,
            spec,
            "flood",
            ToolFunc::new(|_, id| {
                Box::pin(async_stream::stream! {
                    for _ in 0..2 {
                        yield piece(&id, &"x".repeat(40_000));
                        tokio::task::yield_now().await;
                    }
                })
            }),
        );

        let mut agent = Agent::try_with_provider(spec.clone(), name).await.unwrap();
        let deltas: Vec<_> = agent
            .run_stream(user("hi"))
            .map(Result::unwrap)
            .collect()
            .await;
        let pieces: Vec<_> = deltas
            .iter()
            .filter(|d| d.delta.role == Some(Role::Tool) && !d.delta.contents.is_empty())
            .collect();
        assert_eq!(pieces.len(), 2);
        let cut = middle_truncate(&"x".repeat(40_000), 30_000);
        for piece in &pieces {
            assert_eq!(delta_text(piece), cut);
        }
        let assembled: Vec<_> =
            crate::message::into_messages(stream::iter(deltas.into_iter().map(anyhow::Ok)))
                .map(|o| o.unwrap().message)
                .collect()
                .await;
        let history = tool_results(agent.history());
        assert_eq!(tool_results(&assembled), history);
        // The pieces as they streamed, cut once more as a whole.
        assert_eq!(
            text_of(&agent.history()[3]),
            middle_truncate(&cut.repeat(2), 30_000)
        );
        // The model read the clamped result.
        assert_eq!(
            tool_results(&model.seen.lock().unwrap()[1]),
            history,
            "the model reads what the history keeps"
        );

        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();
        let outputs: Vec<_> = agent.run(user("hi")).map(Result::unwrap).collect().await;
        let yielded: Vec<_> = outputs.iter().map(|o| &o.message).collect();
        assert_eq!(tool_results(yielded), history);
    }

    /// A result clamped as it streamed is not cut again when it finishes in one piece.
    #[tokio::test]
    async fn test_a_large_result_returned_once_is_cut_once() {
        let name = "test-agent-tool-large";
        let (_, spec) = scripted(name, [calls(&[("c1", "large")]), answer("done")]);
        let spec = with_tool(
            name,
            spec,
            "large",
            tool_func!(|_args: Value, id: String| -> Message {
                Message::new(Role::Tool)
                    .with_contents([Part::text("x".repeat(40_000))])
                    .with_id(id)
            }),
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();
        agent.run(user("hi")).for_each(|_| async {}).await;
        assert_eq!(
            text_of(&agent.history()[3]),
            middle_truncate(&"x".repeat(40_000), 30_000)
        );
    }

    #[tokio::test]
    async fn test_every_call_gets_one_result_whatever_goes_wrong() {
        suppress_panics!();
        let name = "test-agent-tool-errors";
        let (_, spec) = scripted(
            name,
            [
                calls(&[
                    ("c1", "echo"),
                    ("c2", "boom"),
                    ("c3", "missing"),
                    ("c4", "console_only"),
                ]),
                answer("done"),
            ],
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();
        agent.run(user("hi")).for_each(|_| async {}).await;

        let results: HashMap<_, _> = agent
            .history()
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| (m.id.clone().unwrap(), m.clone()))
            .collect();
        assert_eq!(results.len(), 4);
        assert!(text_of(&results["c2"]).contains("panicked"));
        assert!(text_of(&results["c3"]).contains("no tool named"));
        assert!(text_of(&results["c4"]).contains("needs a console"));
        assert_eq!(text_of(agent.history().last().unwrap()), "done");
    }

    #[tokio::test]
    async fn test_a_failed_turn_takes_its_query_back() {
        let name = "test-agent-rollback";
        let (_, spec) = scripted(name, [Err(anyhow::anyhow!("down")), answer("up")]);
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();

        assert!(agent.run(user("first")).next().await.unwrap().is_err());
        assert_eq!(roles(agent.history()), [Role::System]);

        let outputs: Vec<_> = agent.run_stream(user("second")).collect().await;
        assert!(outputs.iter().all(Result::is_ok));
        assert_eq!(
            roles(agent.history()),
            [Role::System, Role::User, Role::Assistant]
        );
    }

    #[tokio::test]
    async fn test_a_given_system_message_is_kept() {
        let name = "test-agent-system";
        let (_, spec) = scripted(name, []);
        let state = AgentState::new()
            .with_history([Message::new(Role::System).with_contents([Part::text("Mine.")])]);
        let agent = Agent::try_with_provider_and_state(spec, name, state)
            .await
            .unwrap();
        assert_eq!(agent.history().len(), 1);
        assert_eq!(text_of(&agent.history()[0]), "Mine.");
    }

    #[tokio::test]
    async fn test_a_missing_tool_fails_construction() {
        let name = "test-agent-missing-tool";
        let (_, spec) = scripted(name, []);
        let spec = spec.tool(ToolDescBuilder::new("nowhere").build());
        assert!(Agent::try_with_provider(spec, name).await.is_err());
    }

    /// `output` as having read `input_tokens`.
    fn with_input(
        output: anyhow::Result<MessageOutput>,
        input_tokens: u64,
    ) -> anyhow::Result<MessageOutput> {
        output.map(|output| MessageOutput {
            usage: Some(TokenUsage {
                input_tokens,
                ..TokenUsage::default()
            }),
            ..output
        })
    }

    #[tokio::test]
    async fn test_context_manager_truncates_once_input_grows_past_its_limit() {
        let name = "test-agent-context";
        let (model, spec) = scripted(
            name,
            [
                with_input(calls(&[("c1", "echo")]), 10),
                with_input(answer("one"), 200),
                with_input(answer("two"), 10),
            ],
        );
        let mut agent = Agent::try_with_provider(spec, name).await.unwrap();
        agent.set_context_manager(Some(ContextManager {
            max_input_tokens: 100,
            preserve_recent_turns: 1,
        }));

        agent.run(user("first")).for_each(|_| async {}).await;
        // The last call read 200 tokens, so the next turn's first call cuts the old result.
        agent.run(user("second")).for_each(|_| async {}).await;

        let seen = model.seen.lock().unwrap();
        let result = seen[2].iter().find(|m| m.role == Role::Tool).unwrap();
        assert_eq!(text_of(result), "[context truncated]");
        assert_eq!(result.id.as_deref(), Some("c1"));
        // Under the limit before that, so the second call saw the result whole.
        let result = seen[1].iter().find(|m| m.role == Role::Tool).unwrap();
        assert_ne!(text_of(result), "[context truncated]");
    }
}
