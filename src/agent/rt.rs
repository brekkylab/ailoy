use std::{collections::HashMap, path::PathBuf, pin::Pin};

use futures::{FutureExt as _, Stream, StreamExt as _, stream::FuturesUnordered};

use crate::{
    agent::{AgentProvider, AgentSpec, ContextManager, default_provider},
    lang_model::{LangModel, LangModelOptions},
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::{Console, Local, SharedMachine},
    skill::{render_skills_table, scan_declared_skills},
    tool::{
        ToolDesc, ToolFunc,
        r#impl::{get_subagent_tool_desc, get_subagent_tool_func, get_web_search_tool_factory},
    },
};

/// Walk the spec tree and write every declared file (this agent's plus the
/// subtree's) to the machine with **write-once** semantics: if a file already
/// exists at the target path, the existing content is left untouched so that
/// runtime modifications survive subsequent invocations.
fn materialise_files_recursive<'a>(
    spec: &'a AgentSpec,
    console: &'a dyn Console,
) -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>> {
    Box::pin(async move {
        for f in &spec.files {
            // Write-once: skip if the file already exists.
            if console.read(&f.path).await.is_ok() {
                continue;
            }
            console.write(&f.path, f.content.as_ref()).await?;
        }
        for sub in &spec.subagents {
            materialise_files_recursive(sub, console).await?;
        }
        Ok(())
    })
}

pub struct AgentState {
    pub history: Vec<Message>,

    /// Shared machine handle. Defaults to [`Local`] wrapped in `Arc<Mutex<>>`.
    /// Sub-agents inherit this via `Arc::clone` so they share the same VM.
    pub machine: SharedMachine,

    /// Token count from the most recent model API call; used to decide when to truncate history.
    pub last_input_tokens: Option<u64>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            machine: SharedMachine::new(Local::new()),
            last_input_tokens: None,
        }
    }

    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    pub fn machine(mut self, machine: SharedMachine) -> Self {
        self.machine = machine;
        self
    }
}

/// An agent that drives a language model through multi-turn, tool-augmented conversations.
///
/// `Agent` pairs an [`AgentSpec`] (model + instruction + tools + sub-agents) with an
/// [`AgentProvider`] (credentials + tool sources) and an internal [`AgentState`]
/// (message history + shared machine).  Call [`Agent::run`] to stream a single turn;
/// tool calls are resolved automatically and the conversation is appended to history
/// after each turn.
///
/// Sub-agents declared in [`AgentSpec::subagents`] are materialised at construction time
/// and registered as callable tools, inheriting the parent's machine so they share
/// filesystem state.
///
/// For construction options, see [`Agent::try_new`], [`Agent::try_with_provider`],
/// [`Agent::try_with_machine`], and [`Agent::try_with_provider_and_machine`].
pub struct Agent {
    model: LangModel,

    model_options: LangModelOptions,

    tool_descs: Vec<ToolDesc>,

    tools: HashMap<String, ToolFunc>,

    pub state: AgentState,

    context_manager: Option<ContextManager>,

    /// The spec this agent was built from.  Carries the agent's identity:
    /// model, instruction, tools, sub-agents, card, declared files, and
    /// declared [`AgentSpec::skills`].
    spec: AgentSpec,

    /// Lazy gate: whether the declared [`FileEntry`](crate::runenv::FileEntry)
    /// list for this agent (and its declared sub-spec subtree) has been
    /// written to the machine.  Toggled by `ensure_files_materialised` on
    /// the first [`Self::run`].
    files_materialised: bool,
}

impl Agent {
    /// Create an agent using the process-wide [`default_provider`] and a [`Local`] machine.
    pub fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        let provider = default_provider();
        Self::try_with_provider(spec, &provider)
    }

    /// Create an agent using the process-wide [`default_provider`] and an explicit machine.
    pub fn try_with_machine(spec: AgentSpec, machine: SharedMachine) -> anyhow::Result<Self> {
        let provider = default_provider();
        Self::try_with_provider_and_machine(spec, &provider, machine)
    }

    /// Create an agent with an explicit [`AgentProvider`] and a local machine.
    ///
    /// Use this for scoped, explicit control over models and tool sources without
    /// touching global state.  The provider is not stored in the agent and can be
    /// reused across multiple agents.
    pub fn try_with_provider(spec: AgentSpec, provider: &AgentProvider) -> anyhow::Result<Self> {
        Self::try_with_provider_and_machine(spec, provider, SharedMachine::new(Local::new()))
    }

    /// Create an agent with an explicit [`AgentProvider`] and shared machine.
    ///
    /// The full constructor.  The supplied `machine` is stored in [`AgentState::machine`]
    /// and is also cloned into every sub-agent declared in [`AgentSpec::subagents`], so
    /// the parent and its sub-agents observe the same filesystem and process state.
    ///
    /// Files declared in [`AgentSpec::files`] are materialised lazily on the
    /// first [`Self::run`].  Each path in [`AgentSpec::skills`] is an absolute
    /// directory containing a `SKILL.md`; the spec is taken as-is — sub-spec
    /// skill paths are **not** rewritten, so sub-agent skills are portable
    /// across parents.
    pub fn try_with_provider_and_machine(
        spec: AgentSpec,
        provider: &AgentProvider,
        machine: SharedMachine,
    ) -> anyhow::Result<Self> {
        // Resolve LangModel from the registry (handles glob lookup + prefix stripping)
        let model = provider.models.provide(&spec.model)?;

        let model_options = spec.model_options.clone().unwrap_or_default();

        // Collect tools required by the spec; error if any tool is missing.
        // When the spec requests specific web_search engines, override the default factory.
        let mut tools = if let Some(engines) = spec.web_search_engines.as_ref() {
            let mut tp = provider.tools.clone();
            tp.insert_func_factory("web_search", get_web_search_tool_factory(engines.clone()));
            tp.provide(&spec.tools)?
        } else {
            provider.tools.provide(&spec.tools)?
        };
        let mut tool_descs = spec.tools.clone();

        // Sub-agents become regular tool entries: each is a one-shot ToolFunc
        // that materialises a fresh Agent on call and shares the parent's
        // machine so filesystem state is shared.  Sub-specs are taken as-is —
        // no path rewriting — so sub-agent skills are portable across
        // parents.  Files are materialised lazily on the first run via
        // [`Self::ensure_files_materialised`].
        for sub_spec in &spec.subagents {
            let card = sub_spec
                .card
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("subagent must declare an AgentCard"))?;
            let desc = get_subagent_tool_desc(card);
            let tool_name = desc.name.clone();
            let func = get_subagent_tool_func(sub_spec.clone(), provider.clone(), machine.clone());
            tool_descs.push(desc);
            tools.insert(tool_name, func);
        }

        // Build the system message: instruction + (optionally) the skills table.
        // The table is rendered from declared `spec.skills` by matching
        // each entry against an in-memory `SKILL.md` FileEntry.
        let declared_skills = scan_declared_skills(&spec.files, &spec.skills)?;
        let skills_block = render_skills_table(&declared_skills);
        let system_text = match (spec.instruction.as_deref(), skills_block) {
            (Some(inst), Some(block)) => Some(format!("{inst}\n\n{block}")),
            (Some(inst), None) => Some(inst.to_string()),
            (None, Some(block)) => Some(block),
            (None, None) => None,
        };
        let history = system_text
            .map(|t| vec![Message::new(Role::System).with_contents([Part::text(t)])])
            .unwrap_or_default();

        let state = AgentState::new().history(history).machine(machine);

        Ok(Self {
            model,
            model_options,
            tools,
            tool_descs,
            state,
            context_manager: None,
            spec,
            files_materialised: false,
        })
    }

    /// Maximum number of characters kept in a single tool-result message before
    /// middle-truncation is applied.  Mirrors the limit already enforced by the
    /// built-in shell tool so that *all* tool results stay within a consistent bound.
    const MAX_TOOL_RESULT_CHARS: usize = 30_000;

    /// Clamp every [`Part`] in a [`Role::Tool`] message so that large payloads
    /// (e.g. web-search results) do not accumulate unbounded in history and
    /// trigger 429 rate-limit errors.
    ///
    /// * `Part::Value` – serialised to JSON to measure size; if over the limit the
    ///   truncated string is stored back as a `Part::Value` wrapping a JSON string.
    /// * `Part::Text`  – measured directly; truncated in-place if needed.
    fn cap_tool_result(mut msg: Message) -> Message {
        for part in &mut msg.contents {
            match part {
                Part::Value { value } => {
                    let serialised = serde_json::to_string(value).unwrap_or_default();
                    if serialised.len() > Self::MAX_TOOL_RESULT_CHARS {
                        let truncated = crate::util::truncate::middle_truncate(
                            serialised,
                            Self::MAX_TOOL_RESULT_CHARS,
                        );
                        *value = crate::datatype::Value::string(truncated);
                    }
                }
                Part::Text { text } => {
                    if text.len() > Self::MAX_TOOL_RESULT_CHARS {
                        *text = crate::util::truncate::middle_truncate(
                            std::mem::take(text),
                            Self::MAX_TOOL_RESULT_CHARS,
                        );
                    }
                }
                _ => {}
            }
        }
        msg
    }

    pub(crate) fn set_context_manager(&mut self, cm: Option<ContextManager>) {
        self.context_manager = cm;
    }

    /// Lazy gate: materialise this agent's declared files (and the entire
    /// declared sub-spec subtree) into the machine on first call.  Uses
    /// write-once semantics — files that already exist are left untouched —
    /// so any runtime modifications survive subsequent invocations.
    async fn ensure_files_materialised(&mut self) -> anyhow::Result<()> {
        if self.files_materialised {
            return Ok(());
        }
        let mut guard = self.state.machine.get().await;
        let console = guard.start().await?;
        materialise_files_recursive(&self.spec, console).await?;
        self.files_materialised = true;
        Ok(())
    }

    /// Read-only view of the spec this agent was built from.
    pub fn spec(&self) -> &AgentSpec {
        &self.spec
    }

    /// Read-only view of the agent's declared skill directories.
    pub fn skills(&self) -> &[PathBuf] {
        &self.spec.skills
    }

    /// Execute tool calls concurrently within the current task and return a
    /// stream of all outputs.
    ///
    /// Each tool's future independently borrows the shared machine via the
    /// Mutex — pure tools skip the lock entirely. Driven by
    /// [`FuturesUnordered`] so completions interleave naturally; sub-agent
    /// invocations (pure ToolFunc) do their own machine locking inside their
    /// nested `run()` without re-entering the parent's tool future.
    ///
    /// Panics inside a tool's stream are caught via `catch_unwind` and
    /// converted to synthetic error tool results so the LM always receives
    /// exactly one result per call.
    ///
    /// Returns `Err` immediately (before launching) if any tool name is not found.
    fn execute_tool_calls(
        &self,
        tool_calls: Vec<Part>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<MessageOutput>>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();

        let mut futs: FuturesUnordered<Pin<Box<dyn std::future::Future<Output = ()> + Send>>> =
            FuturesUnordered::new();

        for tool_call in tool_calls {
            let Some((call_id, tool_name, call_args)) = tool_call.as_function() else {
                continue;
            };
            let (tool_name, call_id, call_args) = (
                tool_name.to_string(),
                call_id.to_string(),
                call_args.to_owned(),
            );

            let tool = self
                .tools
                .get(&tool_name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No tool found for '{}'", tool_name))?;

            let machine = self.state.machine.clone();
            let tx = tx.clone();

            futs.push(Box::pin(async move {
                let tx_inner = tx.clone();
                let tool_name_inner = tool_name.clone();
                let call_id_for_call = call_id.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        if tool.needs_console() {
                            // Lock the machine for the duration of the tool's stream:
                            // the returned BoxStream borrows the started console.
                            let mut guard = machine.get().await;
                            let console = guard.start().await?;
                            let mut stream = tool.call(call_args, call_id_for_call, console);
                            let mut last: Option<MessageOutput> = None;
                            while let Some(item) = stream.next().await {
                                if let Some(mut prev) = last.replace(item) {
                                    prev.depth = Some(prev.depth.map_or(0, |d| d) + 1);
                                    if tx_inner.send(Ok(prev)).is_err() {
                                        return anyhow::Ok(false);
                                    }
                                }
                            }
                            match last {
                                Some(mut item) => {
                                    item.depth = Some(0);
                                    item.message.role = Role::Tool;
                                    let _ = tx_inner.send(Ok(item));
                                    anyhow::Ok(true)
                                }
                                None => anyhow::Ok(false),
                            }
                        } else {
                            // Pure tool: the returned stream is `'static`, so it does
                            // not need the machine lock. Critically, this lets a
                            // sub-agent ToolFunc invoke its own nested `run()` (which
                            // re-locks the same Arc<Mutex<>>) without deadlocking
                            // against the parent's tool batch.
                            let dummy = Local::default();
                            let mut stream =
                                tool.call(call_args, call_id_for_call, dummy.into_dummy_console());
                            let mut last: Option<MessageOutput> = None;
                            while let Some(item) = stream.next().await {
                                if let Some(mut prev) = last.replace(item) {
                                    prev.depth = Some(prev.depth.map_or(0, |d| d) + 1);
                                    if tx_inner.send(Ok(prev)).is_err() {
                                        return anyhow::Ok(false);
                                    }
                                }
                            }
                            match last {
                                Some(mut item) => {
                                    item.depth = Some(0);
                                    item.message.role = Role::Tool;
                                    let _ = tx_inner.send(Ok(item));
                                    anyhow::Ok(true)
                                }
                                None => anyhow::Ok(false),
                            }
                        }
                    })
                    .catch_unwind()
                    .await;

                let needs_fallback = !matches!(outcome, Ok(Ok(true)));
                if needs_fallback {
                    if let Ok(Err(e)) = outcome {
                        let _ = tx.send(Err(e));
                    } else {
                        let reason = if outcome.is_err() {
                            "panicked during execution"
                        } else {
                            "produced no output"
                        };
                        let err_msg = Message::new(Role::Tool)
                            .with_contents([Part::value(crate::datatype::Value::string(format!(
                                "tool '{}' {}",
                                tool_name_inner, reason
                            )))])
                            .with_id(call_id);
                        let _ = tx.send(Ok(MessageOutput {
                            message: err_msg,
                            finish_reason: FinishReason::Stop {},
                            usage: None,
                            depth: Some(0),
                            source_agent: None,
                        }));
                    }
                }
            }));
        }

        // Drop the outer sender so the channel closes once all futures finish.
        drop(tx);

        Ok(Box::pin(async_stream::stream! {
            // Drive futures concurrently within the current task.
            let drive = async move {
                while futs.next().await.is_some() {}
            };
            let mut rx = rx;
            let drive = std::pin::pin!(drive);
            let mut drive = drive.fuse();
            loop {
                tokio::select! {
                    _ = &mut drive => {
                        // All tool futures done; drain remaining receives.
                        while let Some(event) = rx.recv().await {
                            yield event;
                        }
                        break;
                    }
                    maybe_event = rx.recv() => {
                        match maybe_event {
                            Some(event) => yield event,
                            None => break,
                        }
                    }
                }
            }
        }))
    }

    /// Stamp `out.source_agent` with this agent's card name if not already set.
    ///
    /// Called on every `MessageOutput` just before it is yielded from
    /// [`Agent::run`].  Because it only writes when the field is `None`,
    /// messages that already carry a name from a deeper subagent are forwarded
    /// unchanged — the innermost producer always wins in nested chains.
    fn stamp_source_agent(&self, out: &mut MessageOutput) {
        if out.source_agent.is_none()
            && let Some(card) = self.spec.card.as_ref()
        {
            out.source_agent = Some(card.name.clone());
        }
    }

    /// Return the full message history accumulated so far.
    pub fn get_history(&self) -> &[Message] {
        &self.state.history
    }

    pub fn get_context_manager(&self) -> Option<&ContextManager> {
        self.context_manager.as_ref()
    }

    /// Stream all events for a single agent turn.
    pub fn run(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageOutput>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            // Lazy gate: write declared files (this agent + sub-spec subtree)
            // to the machine on first call.  Write-once semantics preserve any
            // runtime modifications across subsequent runs.
            self.ensure_files_materialised().await?;

            self.state.history.push(query);

            loop {
                // Truncation check based on previous call's token usage.
                if let Some(cm) = &self.context_manager
                    && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens {
                        cm.truncate_history(&mut self.state.history);
                    }

                let mut output = self
                    .model
                    .run(&self.state.history, &self.tool_descs, &self.model_options)
                    .await?;

                // Capture token usage for next iteration's truncation check.
                if let Some(u) = &output.usage {
                    self.state.last_input_tokens = Some(u.input_tokens);
                }

                output.depth = Some(0);
                self.state.history.push(output.message.clone());
                self.stamp_source_agent(&mut output);

                let tool_calls = match &output.finish_reason {
                    FinishReason::ToolCall {} => {
                        let tc = output.message.tool_calls.clone().unwrap_or_default();
                        yield output;
                        tc
                    },
                    _ => {
                        yield output;
                        break;
                    }
                };

                let mut tool_stream = self.execute_tool_calls(tool_calls)?;
                while let Some(event) = tool_stream.next().await {
                    match event {
                        Err(e) => Err(e)?,
                        Ok(mut output) => {
                            if output.message.role == Role::Tool && output.depth == Some(0) {
                                output.message = Self::cap_tool_result(output.message);
                                self.state.history.push(output.message.clone());
                            }
                            self.stamp_source_agent(&mut output);
                            yield output;
                        }
                    }
                }
            }
        })
    }
}

/// Test-only extension: produce a `&dyn Console` cheaply from a default `Local`
/// without going through `Machine::start`. This dummy is only passed to Pure
/// `ToolFunc`s — they never actually use it.
trait IntoDummyConsole {
    fn into_dummy_console(&self) -> &dyn Console;
}

impl IntoDummyConsole for Local {
    fn into_dummy_console(&self) -> &dyn Console {
        // `Local` always holds a `LocalConsole`; expose it as a trait object.
        // SAFETY: This relies on `Local`'s public `start()` returning the same
        // console after `&mut self` is released — we just read through `&self`.
        // We avoid `&mut self` here because we're inside a `&self` context.
        // `LocalConsole` has no state and `Console`'s methods take `&self`.
        // We achieve this by going through a static stand-in.
        static C: crate::runenv::LocalConsole = crate::runenv::LocalConsole {};
        &C
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{
        agent::{AgentCard, AgentProvider, AgentSpec, ContextManager},
        datatype::Value,
        lang_model::LangModelProvider,
        message::{Message, Part, Role},
        suppress_panics, to_value,
        tool::{ToolDescBuilder, ToolProvider},
        tool_func,
    };

    // ── helpers ───────────────────────────────────────────────────────────────
    fn get_provider() -> AgentProvider {
        dotenvy::dotenv().ok();
        let mut provider = AgentProvider::new();
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            provider
                .models
                .insert("openai/*".into(), LangModelProvider::openai(key.clone()));
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            provider.models.insert(
                "anthropic/*".into(),
                LangModelProvider::anthropic(key.clone()),
            );
        }
        provider
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_simple_tool_call() {
        let temperature_desc = ToolDescBuilder::new("temperature")
            .description("Get the current temperature for a given city")
            .parameters(to_value!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            }))
            .build();
        let temperature_fn = tool_func!(|_args: Value| -> Value { Value::unsigned(25) });
        let mut provider = get_provider();
        provider.tools = ToolProvider::new();
        provider.tools.insert_func("temperature", temperature_fn);

        let spec = AgentSpec::new("openai/gpt-4o-mini").tool(temperature_desc);
        let mut agent = Agent::try_with_provider(spec, &provider).unwrap();

        let query = Message::new(Role::User)
            .with_contents([Part::text("What is the temperature in Seoul?")]);

        let mut strm = agent.run(query);
        let mut events = vec![];
        while let Some(event) = strm.next().await {
            events.push(event.unwrap());
        }

        let has_tool_call = events
            .iter()
            .any(|e| e.message.role == Role::Assistant && e.message.tool_calls.is_some());
        assert!(
            has_tool_call,
            "Expected an assistant message with tool calls"
        );

        let has_tool_result = events.iter().any(|e| e.message.role == Role::Tool);
        assert!(has_tool_result, "Expected a tool result message");

        let last = events.last().expect("Expected at least one event");
        assert_eq!(last.message.role, Role::Assistant);
        assert!(
            last.message.contents.iter().any(|p| p.is_text()),
            "Last event should contain text"
        );
    }

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    /// Verifies that the main agent actually delegates to the in-memory subagent.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_delegate_to_subagent() {
        let provider = get_provider();

        let sub_spec = AgentSpec::new("openai/gpt-4o-mini")
            .instruction(
                "You are a calculator. Answer math questions with the numeric result only."
                    .to_string(),
            )
            .card(AgentCard {
                name: "math-agent".to_string(),
                description:
                    "Handles arithmetic and math computations. Use this for any math question."
                        .to_string(),
                skills: vec![],
            });

        let main_spec = AgentSpec::new("openai/gpt-4o-mini")
            .instruction(
                "You are a coordinator. For any arithmetic or math question, \
                 always delegate to the math-agent tool."
                    .to_string(),
            )
            .subagent(sub_spec);

        let mut main_agent = Agent::try_with_provider(main_spec, &provider).unwrap();

        let query =
            Message::new(Role::User).with_contents([Part::text("What is 123 multiplied by 7?")]);

        {
            let mut strm = main_agent.run(query);
            while let Some(event) = strm.next().await {
                let _ = event.unwrap();
            }
        }

        let history = main_agent.get_history();
        assert!(
            history.iter().any(|m| m.role == Role::Tool),
            "Expected main agent history to contain a Tool message (subagent was called)"
        );

        let last_assistant = history
            .iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .expect("Expected at least one assistant message");
        assert!(
            last_assistant.contents.iter().any(|p| p.is_text()),
            "Expected final assistant message to contain text"
        );
    }

    /// Verifies that run() emits intermediate sub-agent outputs (depth > 0)
    /// followed by a final Role::Tool result (depth == 0) when using a streaming
    /// subagent tool.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_streaming_subagent_emits_tool_deltas() {
        let provider = get_provider();

        let sub_spec = AgentSpec::new("openai/gpt-4o-mini")
            .instruction(
                "You are a calculator. Answer math questions with the numeric result only."
                    .to_string(),
            )
            .card(AgentCard {
                name: "math-agent".to_string(),
                description: "Handles arithmetic and math computations.".to_string(),
                skills: vec![],
            });

        let main_spec = AgentSpec::new("openai/gpt-4o-mini").subagent(sub_spec);

        let mut main_agent = Agent::try_with_provider(main_spec, &provider).unwrap();

        let query = Message::new(Role::User).with_contents([Part::text("What is 99 plus 1?")]);

        let mut strm = main_agent.run(query);
        let mut tool_deltas = 0usize;
        let mut tool_results = 0usize;

        while let Some(event) = strm.next().await {
            let output = event.unwrap();
            if output.depth.is_some_and(|d| d > 0) {
                tool_deltas += 1;
                assert_eq!(
                    output.source_agent.as_deref(),
                    Some("math-agent"),
                    "Intermediate subagent events must carry the subagent's card name"
                );
            }
            if output.message.role == Role::Tool && output.depth == Some(0) {
                tool_results += 1;
                assert_eq!(
                    output.source_agent.as_deref(),
                    Some("math-agent"),
                    "Final tool-result event must carry the subagent's card name"
                );
            }
        }

        assert!(tool_results > 0, "Expected at least one tool result");
        assert!(
            tool_deltas > 0,
            "Expected at least one intermediate sub-agent output (tool delta)"
        );
    }

    /// Verifies history consistency when one of two parallel tool calls panics.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_tool_panic_causes_inconsistent_history() {
        suppress_panics!();

        let good_desc = ToolDescBuilder::new("get_weather")
            .description("Get the current weather for a city")
            .parameters(to_value!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }))
            .build();
        let good_fn = tool_func!(async |_args: Value| -> Value {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            to_value!("sunny, 25 degrees")
        });

        let bad_desc = ToolDescBuilder::new("get_traffic")
            .description("Get the current traffic conditions for a city")
            .parameters(to_value!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }))
            .build();
        let bad_fn = tool_func!(async |_args: Value| -> Value {
            panic!("simulated tool crash");
            #[allow(unreachable_code)]
            Value::null()
        });

        let mut provider = get_provider();
        provider.tools = ToolProvider::new();
        provider.tools.insert_func("get_weather", good_fn);
        provider.tools.insert_func("get_traffic", bad_fn);

        let spec = AgentSpec::new("openai/gpt-4o-mini")
            .tools([good_desc, bad_desc])
            .instruction(
                "When asked about a city, ALWAYS call both get_weather AND \
                 get_traffic tools in a single response. Never call just one."
                    .to_string(),
            );

        let mut agent = Agent::try_with_provider(spec, &provider).unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Tell me about Seoul. Use get_weather for weather and get_traffic for traffic.",
        )]);

        {
            let mut strm = agent.run(query);
            while let Some(result) = strm.next().await {
                let _ = result;
            }
        }

        let history = agent.get_history();

        let tool_use_count: usize = history
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .filter_map(|m| m.tool_calls.as_ref())
            .map(|tc| tc.len())
            .sum();

        let tool_result_count = history.iter().filter(|m| m.role == Role::Tool).count();

        if tool_use_count < 2 {
            eprintln!(
                "LLM only produced {} tool call(s); skipping consistency check",
                tool_use_count
            );
            return;
        }

        assert_eq!(
            tool_use_count, tool_result_count,
            "History is inconsistent: {} tool_use call(s) but {} tool result(s).",
            tool_use_count, tool_result_count,
        );

        for tool_msg in history.iter().filter(|m| m.role == Role::Tool) {
            for part in &tool_msg.contents {
                assert!(
                    part.as_value().is_some(),
                    "Tool result content must be Part::Value for correct API marshalling"
                );
            }
        }

        let last_msg = history.last().expect("History should not be empty");
        assert_eq!(
            last_msg.role,
            Role::Assistant,
            "History must end with an Assistant message, not {:?}",
            last_msg.role
        );
        assert!(
            last_msg.contents.iter().any(|p| p.is_text()),
            "Final Assistant message must contain text"
        );
    }

    /// Verifies that ContextManager replaces old tool results with "[context truncated]"
    /// when last_input_tokens exceeds max_input_tokens at the start of a run.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_context_manager_truncates_tool_results_when_threshold_exceeded() {
        let dummy_desc = ToolDescBuilder::new("dummy_tool")
            .description("A no-op testing tool")
            .parameters(to_value!({ "type": "object", "properties": {} }))
            .build();
        let dummy_fn = tool_func!(|_args: Value| -> Value { Value::string("result".to_string()) });
        let mut provider = get_provider();
        provider.tools = ToolProvider::new();
        provider.tools.insert_func("dummy_tool", dummy_fn);

        let spec = AgentSpec::new("openai/gpt-5.4-mini")
            .instruction("Reply with exactly 'OK'. Do not call any tools.")
            .tool(dummy_desc);

        let mut agent = Agent::try_with_provider(spec, &provider).unwrap();

        let old_id = "call_old";
        let recent_id = "call_recent";
        for (user_text, call_id) in [("q1", old_id), ("q2", recent_id)] {
            agent
                .state
                .history
                .push(Message::new(Role::User).with_contents([Part::text(user_text)]));
            agent.state.history.push(
                Message::new(Role::Assistant).with_tool_calls([Part::function(
                    call_id,
                    "dummy_tool",
                    to_value!({}),
                )]),
            );
            agent.state.history.push(
                Message::new(Role::Tool)
                    .with_id(call_id)
                    .with_contents([Part::value(Value::string(format!("{call_id}_value")))]),
            );
        }

        agent.set_context_manager(Some(ContextManager {
            max_input_tokens: 1,
            preserve_recent_turns: 2,
        }));
        agent.state.last_input_tokens = Some(9999);

        {
            let mut strm = agent.run(Message::new(Role::User).with_contents([Part::text("q3")]));
            while let Some(ev) = strm.next().await {
                ev.unwrap();
            }
        }

        let history = agent.get_history();

        let old_tool = history
            .iter()
            .find(|m| m.role == Role::Tool && m.id.as_deref() == Some(old_id))
            .expect("old Tool message must still exist in history");
        assert_eq!(
            old_tool.contents.first().and_then(|p| p.as_text()),
            Some("[context truncated]"),
            "tool result outside preserve window must become '[context truncated]'"
        );

        let recent_tool = history
            .iter()
            .find(|m| m.role == Role::Tool && m.id.as_deref() == Some(recent_id))
            .expect("recent Tool message must still exist in history");
        let recent_val = recent_tool
            .contents
            .first()
            .and_then(|p| p.as_value())
            .expect("recent tool result must still be a Value part");
        assert_eq!(
            recent_val.as_str(),
            Some("call_recent_value"),
            "tool result inside preserve window must retain its original content"
        );
    }

    /// Verifies that ContextManager does NOT truncate tool results when
    /// last_input_tokens is below max_input_tokens.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_context_manager_no_truncation_when_below_threshold() {
        let dummy_desc = ToolDescBuilder::new("dummy_tool")
            .description("A no-op testing tool")
            .parameters(to_value!({ "type": "object", "properties": {} }))
            .build();
        let dummy_fn = tool_func!(|_args: Value| -> Value { Value::string("result".to_string()) });
        let mut provider = get_provider();
        provider.tools = ToolProvider::new();
        provider.tools.insert_func("dummy_tool", dummy_fn);

        let spec = AgentSpec::new("openai/gpt-5.4-mini")
            .instruction("Reply with exactly 'OK'. Do not call any tools.")
            .tool(dummy_desc);

        let mut agent = Agent::try_with_provider(spec, &provider).unwrap();

        let old_id = "call_old_b";
        let recent_id = "call_recent_b";
        for (user_text, call_id) in [("q1", old_id), ("q2", recent_id)] {
            agent
                .state
                .history
                .push(Message::new(Role::User).with_contents([Part::text(user_text)]));
            agent.state.history.push(
                Message::new(Role::Assistant).with_tool_calls([Part::function(
                    call_id,
                    "dummy_tool",
                    to_value!({}),
                )]),
            );
            agent.state.history.push(
                Message::new(Role::Tool)
                    .with_id(call_id)
                    .with_contents([Part::value(Value::string(format!("{call_id}_value")))]),
            );
        }

        agent.set_context_manager(Some(ContextManager {
            max_input_tokens: 1_000_000,
            preserve_recent_turns: 1,
        }));
        agent.state.last_input_tokens = Some(100);

        {
            let mut strm = agent.run(Message::new(Role::User).with_contents([Part::text("q3")]));
            while let Some(ev) = strm.next().await {
                ev.unwrap();
            }
        }

        let history = agent.get_history();

        let old_tool = history
            .iter()
            .find(|m| m.role == Role::Tool && m.id.as_deref() == Some(old_id))
            .expect("old Tool message must still exist in history");
        let old_val = old_tool
            .contents
            .first()
            .and_then(|p| p.as_value())
            .expect("tool result must still be a Value part when threshold is not exceeded");
        assert_eq!(
            old_val.as_str(),
            Some("call_old_b_value"),
            "tool result must not be replaced when last_input_tokens is below max_input_tokens"
        );
    }
}
