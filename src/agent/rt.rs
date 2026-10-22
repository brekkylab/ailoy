use std::{collections::HashMap, panic::AssertUnwindSafe, path::Path, sync::Arc};

use anyhow::Context as _;
use futures::{
    FutureExt as _, StreamExt as _,
    stream::{BoxStream, FuturesUnordered},
};
use tokio::sync::Mutex;
use virtx::console::ConsoleClient;

use super::{AgentSpec, AgentState, ContextManager, skill::render_skills, spec::split_desc};
use crate::{
    browser,
    datatype::Value,
    message::{Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, Role},
    model::{InferLangModel, LangModelOptions, create_lang_model},
    tool::{
        ToolDesc, ToolFunc, create_tool,
        r#impl::{
            get_mem_insert_tool_desc, get_mem_insert_tool_func, get_mem_search_tool_desc,
            get_mem_search_tool_func, get_web_search_tool_factory,
        },
    },
};

/// Per-part size limit for tool results, past which the middle is cut out.
const MAX_TOOL_RESULT_CHARS: usize = 30_000;

/// An agent that drives an [`InferLangModel`] through multi-turn, tool-augmented
/// conversations.
///
/// Pairs an [`AgentSpec`] (model + instruction + tools) with an [`AgentState`]
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
    /// Create an agent with a fresh [`AgentState`].
    pub async fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        Self::try_with_state(spec, AgentState::new()).await
    }

    /// Create an agent with `state`.
    ///
    /// The model is built from `spec.model` through [`create_lang_model`], and each of
    /// `spec.tools` through [`create_tool`]. Unless `state.history` already has a
    /// [`Role::System`] message, one built from `spec.instruction` and
    /// [`AgentSpec::skills`] is put first.
    ///
    /// Async because skills are read through the console.
    pub async fn try_with_state(spec: AgentSpec, mut state: AgentState) -> anyhow::Result<Self> {
        let model = create_lang_model(&spec.model).await?;

        // Errors if any spec tool is missing.
        let mut tools = HashMap::with_capacity(spec.tools.len());
        for desc in &spec.tools {
            let func = match (&spec.web_search_engines, desc.name.as_str()) {
                // The engines are this spec's, so its `web_search` is built here.
                (Some(engines), "web_search") => {
                    get_web_search_tool_factory(engines.clone())(desc)?
                }
                _ => create_tool(desc)?,
            };
            tools.insert(desc.name.clone(), func);
        }
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

        // Like memory, bound to this agent's own browser rather than resolved by name.
        if let Some(session) = &state.browser {
            let console = state.console.lock().await.is_some();
            for (desc, func) in browser::tools(session, console) {
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
                while let Some(message) = results.next().await {
                    let message = message?;
                    self.state.history.push(message.clone());
                    yield tool_output(message);
                }
            }
        })
    }

    /// Like [`run`](Self::run), but yields the model's output as deltas while it is
    /// generated, and each tool result as one complete delta. A `finish_reason` marks the
    /// end of a message.
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
                while let Some(message) = results.next().await {
                    let message = message?;
                    self.state.history.push(message.clone());
                    yield tool_output(message).into();
                }
            }
        })
    }

    /// Add `query` to the history, led by a browser reconnection the agent has not heard of,
    /// and tell the model where its console mounts are.
    async fn begin_turn(&mut self, mut query: Message) {
        close_dangling_calls(&mut self.state.history);
        if let Some(note) = self.state.browser.as_ref().and_then(|b| b.catch_up()) {
            query.contents.insert(0, Part::text(note));
        }
        self.state.history.push(query);
        self.seed_mounts().await;
    }

    /// Connects this agent's browser now rather than on its first use. This is when the browser
    /// asks the person, so call it while they are at their screen. A no-op once connected.
    pub async fn start_browser(&mut self) -> anyhow::Result<()> {
        self.state
            .browser
            .clone()
            .context("this agent has no browser")?
            .start()
            .await
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

    /// Run `calls` concurrently, yielding each result as a [`Role::Tool`] message as it
    /// finishes, one per call.
    ///
    /// The console is started for the batch only if a console tool is in it, and stopped
    /// after, so its backend (possibly a whole micro-VM) is not kept up while the model
    /// thinks. Stopping keeps the session, so the next batch resumes the same console.
    fn call_tools(&self, calls: Vec<Part>) -> BoxStream<'static, anyhow::Result<Message>> {
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

            let mut pending: FuturesUnordered<_> = calls
                .into_iter()
                .map(|(id, name, args, tool)| call_tool(id, name, args, tool, console.clone()))
                .collect();
            while let Some(message) = pending.next().await {
                yield message;
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

fn tool_output(message: Message) -> MessageOutput {
    MessageOutput {
        message,
        finish_reason: FinishReason::Stop {},
        usage: None,
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

/// Run one tool call to its result, the last message its stream yields.
///
/// Whatever goes wrong (an unknown tool, a missing console, a panic) becomes the result,
/// so the model gets exactly one per call and can recover.
async fn call_tool(
    id: String,
    name: String,
    args: Value,
    tool: Option<ToolFunc>,
    console: Arc<Mutex<Option<ConsoleClient>>>,
) -> Message {
    let Some(tool) = tool else {
        return tool_error(id, format!("there is no tool named '{name}'"));
    };

    async fn last<T>(_: Option<T>, item: T) -> Option<T> {
        Some(item)
    }
    let call_id = id.clone();
    let run = async move {
        match tool.call_pure(args.clone(), call_id.clone()) {
            Some(stream) => Ok(stream.fold(None, last).await),
            None => {
                let mut guard = console.lock().await;
                let Some(console) = guard.as_mut() else {
                    return Err("it needs a console, and the agent has none; \
                                give it one with `AgentState::with_console`");
                };
                Ok(tool.call(args, call_id, console).fold(None, last).await)
            }
        }
    };

    let mut message = match AssertUnwindSafe(run).catch_unwind().await {
        Ok(Ok(Some(output))) => output.message,
        Ok(Ok(None)) => return tool_error(id, format!("tool '{name}' produced no output")),
        Ok(Err(e)) => return tool_error(id, format!("tool '{name}' failed: {e}")),
        Err(_) => return tool_error(id, format!("tool '{name}' panicked")),
    };
    message.role = Role::Tool;
    message.id.get_or_insert(id);
    cap_tool_result(message)
}

/// Clamp every part of a tool result, so large payloads (e.g. web pages) don't pile up in
/// the history. An oversized [`Part::Value`] is measured as JSON and replaced by the cut
/// JSON as a string.
fn cap_tool_result(mut message: Message) -> Message {
    for part in &mut message.contents {
        match part {
            Part::Value { value } => {
                let json = serde_json::to_string(value).unwrap_or_default();
                if json.len() > MAX_TOOL_RESULT_CHARS {
                    *value = Value::string(middle_truncate(&json, MAX_TOOL_RESULT_CHARS));
                }
            }
            Part::Text { text } if text.len() > MAX_TOOL_RESULT_CHARS => {
                *text = middle_truncate(text, MAX_TOOL_RESULT_CHARS);
            }
            _ => {}
        }
    }
    message
}

/// Keep `max_chars` characters of `s`, half from each end, around a note of what was cut.
fn middle_truncate(s: &str, max_chars: usize) -> String {
    let len = s.chars().count();
    if len <= max_chars {
        return s.to_owned();
    }
    let head = max_chars / 2;
    let tail = max_chars - head;
    let head_str: String = s.chars().take(head).collect();
    let tail_str: String = s.chars().skip(len - tail).collect();
    format!(
        "{head_str}\n\n... [{} characters omitted] ...\n\n{tail_str}",
        len - head - tail
    )
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
        message::{Role, TokenUsage},
        model::{LangModelProvider, set_lang_model_provider},
        suppress_panics, to_value,
        tool::{ToolDescBuilder, ToolProvider, set_tool_provider},
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
    /// `boom` (panics) and `console_only` tools.
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

        set_tool_provider(
            "echo",
            tool_func!(|_args: Value| -> Value { Value::string("echoed") }),
        );
        set_tool_provider(
            "boom",
            tool_func!(|_args: Value| -> Value {
                if true {
                    panic!("boom");
                }
                Value::Null
            }),
        );
        set_tool_provider(
            "console_only",
            tool_func!(async |_args: Value, _console: &mut ConsoleClient| -> Value { Value::Null }),
        );

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
        let mut agent = Agent::try_new(spec).await.unwrap();

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
        let mut agent = Agent::try_new(spec).await.unwrap();

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
        let mut agent = Agent::try_new(spec).await.unwrap();
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
        let mut agent = Agent::try_new(spec).await.unwrap();

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
        let agent = Agent::try_with_state(spec, state).await.unwrap();
        assert_eq!(agent.history().len(), 1);
        assert_eq!(text_of(&agent.history()[0]), "Mine.");
    }

    #[tokio::test]
    async fn test_a_missing_tool_fails_construction() {
        let name = "test-agent-missing-tool";
        let (_, spec) = scripted(name, []);
        let spec = spec.tool(ToolDescBuilder::new("nowhere").build());
        assert!(Agent::try_new(spec).await.is_err());
    }

    #[tokio::test]
    async fn test_a_failing_tool_factory_fails_construction() {
        let name = "test-agent-failing-factory";
        let (_, spec) = scripted(name, []);
        set_tool_provider(
            "refusing",
            ToolProvider::function(|_| Err(anyhow::anyhow!("not today"))),
        );
        let spec = spec.tool(ToolDescBuilder::new("refusing").build());
        let err = Agent::try_new(spec).await.err().unwrap();
        assert!(err.to_string().contains("not today"), "{err}");
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
        let mut agent = Agent::try_new(spec).await.unwrap();
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

    #[test]
    fn test_middle_truncate_keeps_both_ends() {
        let cut = middle_truncate(&"a".repeat(10), 4);
        assert!(cut.starts_with("aa\n\n... [6 characters omitted]"), "{cut}");
        assert_eq!(middle_truncate("short", 10), "short");
    }
}
