use std::{collections::HashMap, path::Path, pin::Pin};

use futures::{FutureExt as _, Stream, StreamExt as _, stream::FuturesUnordered};

use crate::{
    agent::{
        AgentProvider, AgentSpec, AgentState, ContextManager, get_agent_providers,
        subagent::{get_subagent_tool_desc, get_subagent_tool_func},
    },
    lang_model::{LangModel, LangModelOptions},
    message::{Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, Role},
    tool::{ToolDesc, ToolFunc, get_tool_providers, r#impl::get_web_search_tool_factory},
};

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
/// Constructors:
/// * [`Agent::try_new`] — `"default"` provider + fresh [`AgentState`].
/// * [`Agent::try_with_provider`] — named provider + fresh state.
/// * [`Agent::try_with_state`] — `"default"` provider + explicit state.
/// * [`Agent::try_with_provider_and_state`] — named provider + explicit state.
pub struct Agent {
    model: LangModel,

    model_options: LangModelOptions,

    tool_descs: Vec<ToolDesc>,

    tools: HashMap<String, ToolFunc>,

    pub state: AgentState,

    context_manager: Option<ContextManager>,

    /// Card name lifted from the originating spec.  Used by
    /// [`Self::stamp_source_agent`] to tag streamed events.
    card_name: Option<String>,
}

impl Agent {
    /// Create an agent using the `"default"` entry of the process-wide
    /// [`get_agent_providers`] registry and a fresh [`AgentState`].
    pub fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", AgentState::new())
    }

    /// Create an agent using the [`AgentProvider`] registered under `provider`
    /// in [`get_agent_providers`] and a fresh [`AgentState`].
    pub fn try_with_provider(spec: AgentSpec, provider: impl AsRef<str>) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, provider, AgentState::new())
    }

    /// Create an agent using the `"default"` entry of the process-wide
    /// [`get_agent_providers`] registry and an explicit [`AgentState`].
    pub fn try_with_state(spec: AgentSpec, state: AgentState) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", state)
    }

    /// Create an agent using the [`AgentProvider`] registered under `provider`
    /// in [`get_agent_providers`] and an explicit [`AgentState`].
    ///
    /// The canonical constructor.  `state.machine` is cloned into every
    /// sub-agent declared in [`AgentSpec::subagents`], so the parent and its
    /// sub-agents observe the same filesystem and process state.  Sub-agents
    /// inherit the same `provider` name and re-resolve it from the registry
    /// on every invocation — make sure the name stays registered for the
    /// lifetime of the agent.
    ///
    /// Unless `state.history` already leads with a [`Role::System`] message, one
    /// built from `spec.instruction` is inserted at the front; a history that leads
    /// with one is taken as-is, so the caller's own system message wins.
    pub fn try_with_provider_and_state(
        spec: AgentSpec,
        provider: impl AsRef<str>,
        mut state: AgentState,
    ) -> anyhow::Result<Self> {
        let provider_name = provider.as_ref();
        let provider_value: AgentProvider = get_agent_providers()
            .get(provider_name)
            .ok_or_else(|| anyhow::anyhow!("agent_provider '{}' not registered", provider_name))?
            .clone();

        let model =
            LangModel::try_from_provider(spec.model.clone(), &provider_value.lang_model_provider)?;
        let model_options = spec.model_options.clone().unwrap_or_default();

        // Collect tools required by the spec; error if any tool is missing.
        // When the spec requests specific web_search engines, override the default factory.
        let mut tools = {
            let registry = get_tool_providers();
            let tp = registry.get(&provider_value.tool_provider).ok_or_else(|| {
                anyhow::anyhow!(
                    "tool_provider '{}' not registered",
                    provider_value.tool_provider
                )
            })?;
            if let Some(engines) = spec.web_search_engines.as_ref() {
                let mut tp = tp.clone();
                tp.insert_func_factory("web_search", get_web_search_tool_factory(engines.clone()));
                tp.provide(&spec.tools)?
            } else {
                tp.provide(&spec.tools)?
            }
        };
        let mut tool_descs = spec.tools.clone();

        // Sub-agents become regular tool entries: each is a one-shot ToolFunc
        // that materialises a fresh Agent on call (re-resolving the provider
        // name from the registry) and shares the parent's machine so
        // filesystem state is shared.  Sub-specs are taken as-is — no path
        // rewriting — so sub-agent skills are portable across parents.
        for sub_spec in &spec.subagents {
            let card = sub_spec
                .card
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("subagent must declare an AgentCard"))?;
            let desc = get_subagent_tool_desc(card);
            let tool_name = desc.name.clone();
            let func = get_subagent_tool_func(
                sub_spec.clone(),
                provider_name.to_string(),
                state.console.clone(),
            );
            tool_descs.push(desc);
            tools.insert(tool_name, func);
        }

        // Build the system message from the instruction.
        // A system message is expected only at index 0; `any` covers a stray one too,
        // since seeding a second would either shadow theirs or ship both.
        if !state.history.iter().any(|m| m.role == Role::System) {
            if let Some(text) = spec.instruction.as_deref() {
                // Front: that is where every schema expects a system message, whether
                // it extracts the first one or sends them in place.
                state.history.insert(
                    0,
                    Message::new(Role::System).with_contents([Part::text(text)]),
                );
            }
        }

        Ok(Self {
            model,
            model_options,
            tool_descs,
            tools,
            state,
            context_manager: None,
            card_name: spec.card.map(|c| c.name),
        })
    }

    /// Tell the model which directories it has and what each one is for.
    ///
    /// **Asked of the console, not kept beside it.** Cortex already answers
    /// [`context_path`](cortex::console::Console::context_path) and its two siblings, so a copy on this
    /// side would be a second answer to a settled question. The only thing that ever
    /// made one tempting is that [`try_with_provider_and_state`](Self::try_with_provider_and_state)
    /// is a `fn` and the console sits behind a lock — so the question is asked here
    /// instead, at the top of a turn, which is `async` and takes that lock to
    /// [`start`](Self::start_console) anyway.
    ///
    /// Only the trees the caller actually mounted are described, each under the path
    /// the server answered — a section about a directory that is not there would be a
    /// place for the model to try to write and fail. For the same reason each section's
    /// advice to put something *elsewhere* is written only when that elsewhere exists:
    /// told to move its drafts to a scratch directory it has not got, a model has
    /// nowhere to put them.
    ///
    /// Appended to the system message rather than merged into
    /// [`AgentSpec::instruction`], and appended even to a system message the caller
    /// wrote themselves: these paths are not advice anyone can author ahead of time,
    /// and an instruction that says "leave the result in the artifacts directory" still
    /// needs something to say which directory that is.
    async fn seed_console_trees(&mut self) {
        // Held only long enough to read three paths — nothing below it awaits.
        let (context, artifacts, scratch) = {
            let guard = self.state.console.lock().await;
            let Some(console) = guard.as_ref() else {
                return;
            };
            (
                console.context_path().map(Path::to_path_buf),
                console.artifacts_path().map(Path::to_path_buf),
                console.scratch_path().map(Path::to_path_buf),
            )
        };

        // Every continuation fragment carries its own *leading* space. A `\` line
        // continuation would read as well, but it eats the leading whitespace of the
        // next line, so the space that keeps two words apart has to sit at the end of
        // the line before — where a trailing-whitespace trimmer silently removes it and
        // joins the words. This way the spacing is visible and nothing can take it.
        let mut sections = String::new();

        if let Some(path) = &context {
            sections.push_str(&format!(
                concat!(
                    "\n\n# Context\n\n",
                    "Path: `{}`\n\n",
                    "What you were given to work from: the user's own information, the",
                    " documents and data this task is about. Read what is here before",
                    " assuming anything it would have told you, and prefer it over a",
                    " guess or a search when both could answer.",
                ),
                path.display()
            ));
            if scratch.is_some() {
                sections.push_str(concat!(
                    " A file you need to change is copied into the scratch directory",
                    " and changed there.",
                ));
            }
        }

        if let Some(path) = &artifacts {
            sections.push_str(&format!(
                concat!(
                    "\n\n# Artifacts\n\n",
                    "Path: `{}`\n\n",
                    "Where what the user asked for goes. Every result — the report, the",
                    " figure, the file they came for — is written here, because the",
                    " whole contents of this directory are what gets collected and",
                    " handed back; a result left anywhere else is not delivered.",
                ),
                path.display()
            ));
            if scratch.is_some() {
                sections.push_str(concat!(
                    "\n\nSo put finished work here and only finished work. Intermediate",
                    " files, working copies and anything you write in order to read it",
                    " back belong in the scratch directory.",
                ));
            }
        }

        if let Some(path) = &scratch {
            sections.push_str(&format!(
                concat!(
                    "\n\n# Scratch\n\n",
                    "Path: `{}`\n\n",
                    "Where you work, and where you start: a relative path lands here",
                    " unless you move. Downloads, unpacked archives, drafts,",
                    " intermediate output, anything you write in order to read it back",
                    " — all of it goes here, and you do not have to keep it tidy.\n\n",
                    "Nothing here is delivered: it is thrown away when the session ends",
                ),
                path.display()
            ));
            sections.push_str(if artifacts.is_some() {
                ", so copy what is meant to be seen into the artifacts directory."
            } else {
                "."
            });
        }

        // A console that mounted nothing says nothing about where the agent stands.
        if sections.is_empty() {
            return;
        }

        // Into the *first* text part, not a part of its own. Three of the four provider
        // schemas take `contents.first()` off the system message and drop whatever
        // follows — see the `instructions`/`system`/`system_instruction` extraction in
        // `openai`, `anthropic` and `gemini` — so a second part would reach
        // chat-completions and nowhere else, and silently.
        let Some(at) = self
            .state
            .history
            .iter()
            .position(|m| m.role == Role::System)
        else {
            // No instruction was given and the caller wrote no system message, but the
            // trees are still worth saying on their own.
            self.state.history.insert(
                0,
                Message::new(Role::System).with_contents([Part::text(sections.trim_start())]),
            );
            return;
        };

        let system = &mut self.state.history[at];
        match system.contents.iter_mut().find_map(|p| match p {
            Part::Text { text } => Some(text),
            _ => None,
        }) {
            // Every turn seeds, and a second `Agent` may be built over the history a
            // first one produced. The text is built from the same three paths each
            // time, so what was already written is what would be written again — which
            // makes the section its own marker, with nothing to keep in step with it.
            Some(text) => {
                if text.contains(sections.as_str()) {
                    return;
                }
                text.push_str(&sections);
            }
            // A system message carrying no text at all: the trees lead it.
            None => system.contents.insert(0, Part::text(sections.trim_start())),
        }
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
                Part::Text { text } if text.len() > Self::MAX_TOOL_RESULT_CHARS => {
                    *text = crate::util::truncate::middle_truncate(
                        std::mem::take(text),
                        Self::MAX_TOOL_RESULT_CHARS,
                    );
                }
                _ => {}
            }
        }
        msg
    }

    /// Boot the console's backend for a batch of tool calls.
    ///
    /// Paid per batch rather than once per agent, because the other half of a turn is
    /// spent waiting on the model and a booted console is a backend sitting idle —
    /// on a micro-VM one, a whole VM. `start`/`stop` is the pair cortex offers for
    /// exactly that: `stop` releases what booting took and leaves the session open,
    /// so the next batch starts again on the same console.
    ///
    /// An agent with no console has no backend to boot, and says nothing about it —
    /// its pure tools run either way, and a console tool reports the absence itself.
    async fn start_console(&self) -> anyhow::Result<()> {
        if let Some(console) = self.state.console.lock().await.as_mut() {
            console.start().await?;
        }
        Ok(())
    }

    /// Release what [`start_console`](Self::start_console) booted.
    ///
    /// Not the end of the session — another `start` is allowed and is what the next
    /// batch of tool calls does.
    async fn stop_console(&self) -> anyhow::Result<()> {
        if let Some(console) = self.state.console.lock().await.as_mut() {
            console.stop().await?;
        }
        Ok(())
    }

    pub(crate) fn set_context_manager(&mut self, cm: Option<ContextManager>) {
        self.context_manager = cm;
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

            let console_slot = self.state.console.clone();
            let tx = tx.clone();

            futs.push(Box::pin(async move {
                let tx_inner = tx.clone();
                let tool_name_inner = tool_name.clone();
                // A second clone: the error path below reports the name after the
                // inner block has moved its own.
                let tool_name_for_call = tool_name.clone();
                let call_id_for_call = call_id.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        if let Some(mut stream) =
                            tool.call_pure(call_args.clone(), call_id_for_call.clone())
                        {
                            // Pure tool: the stream is `'static`, so it does not take
                            // the console lock. Critically, this lets a sub-agent
                            // ToolFunc drive its own nested `run()` without deadlocking
                            // against the parent's tool batch.
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
                            // Console tool: hold the lock for the whole stream, which
                            // borrows the console exclusively. That serialises tool
                            // calls against one console — which is the protocol, not a
                            // choice: one outstanding request at a time.
                            let mut guard = console_slot.lock().await;
                            let console = guard.as_mut().ok_or_else(|| {
                                anyhow::anyhow!(
                                    "{tool_name_for_call} needs a console, and this agent \
                                     has none — build one and pass it to \
                                     `AgentBuilder::console`"
                                )
                            })?;
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

    /// Stamp a `source_agent` field with this agent's card name if not already
    /// set. Takes the field directly so it works for both `MessageOutput` and
    /// `MessageDeltaOutput`. Because it only writes when the field is `None`,
    /// items already carrying a name from a deeper subagent are forwarded
    /// unchanged — the innermost producer always wins in nested chains.
    fn stamp_source_agent(&self, source_agent: &mut Option<String>) {
        if source_agent.is_none()
            && let Some(name) = self.card_name.as_ref()
        {
            *source_agent = Some(name.clone());
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

            self.state.history.push(query);
            // See run_stream: pop the dangling query if the turn fails before its
            // assistant message commits, so the reused agent doesn't later push
            // two consecutive User messages.
            let mut committed = false;

            self.seed_console_trees().await;

            loop {
                // Truncation check based on previous call's token usage.
                if let Some(cm) = &self.context_manager
                    && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens {
                        cm.truncate_history(&mut self.state.history);
                    }

                let mut output = match self
                    .model
                    .run(&self.state.history, &self.tool_descs, &self.model_options)
                    .await
                {
                    Ok(o) => o,
                    Err(e) => {
                        if !committed {
                            self.state.history.pop();
                        }
                        Err(e)?
                    }
                };

                // Capture token usage for next iteration's truncation check.
                if let Some(u) = &output.usage {
                    self.state.last_input_tokens = Some(u.input_tokens);
                }

                output.depth = Some(0);
                self.state.history.push(output.message.clone());
                committed = true;
                self.stamp_source_agent(&mut output.source_agent);

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

                self.start_console().await?;

                // Drained to the end even on failure, so the console is released
                // before the error leaves this scope — a `?` here would step over the
                // `stop` and leave the backend booted with nobody driving it.
                let mut tool_stream = self.execute_tool_calls(tool_calls)?;
                let mut failure = None;
                while let Some(event) = tool_stream.next().await {
                    match event {
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                        Ok(mut output) => {
                            if output.message.role == Role::Tool && output.depth == Some(0) {
                                output.message = Self::cap_tool_result(output.message);
                                self.state.history.push(output.message.clone());
                            }
                            self.stamp_source_agent(&mut output.source_agent);
                            yield output;
                        }
                    }
                }
                drop(tool_stream);

                let stopped = self.stop_console().await;
                // The tools' failure first: it is what the caller asked about, and a
                // console that would not stop is the less useful of the two.
                if let Some(e) = failure {
                    Err(e)?;
                }
                stopped?;
            }
        })
    }

    /// Token-streaming counterpart to [`run`](Self::run). Drives the same
    /// agentic loop but calls [`LangModel::run_stream`] per turn, yielding a
    /// uniform stream of [`MessageDeltaOutput`]: the model's incremental deltas
    /// for live rendering, then each tool result as a complete one-shot delta.
    /// A `finish_reason` (or a role change) marks a message boundary; the
    /// blocking [`run`](Self::run) is the accumulate-and-finish counterpart.
    pub fn run_stream(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageDeltaOutput>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {

            self.state.history.push(query);
            // If a turn fails before its assistant message commits, pop the
            // dangling user query so the reused agent's next run doesn't push a
            // second consecutive User message (which most providers reject).
            let mut committed = false;

            self.seed_console_trees().await;

            loop {
                // Truncation check based on previous call's token usage.
                if let Some(cm) = &self.context_manager
                    && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens {
                        cm.truncate_history(&mut self.state.history);
                    }

                // Stream the model's deltas, forwarding each while accumulating
                // the full turn for loop control (history / tool dispatch).
                let mut acc = MessageDeltaOutput::new();
                {
                    let mut delta_stream = self.model.run_stream(
                        &self.state.history,
                        &self.tool_descs,
                        &self.model_options,
                    );
                    while let Some(item) = delta_stream.next().await {
                        let mut delta = match item {
                            Ok(d) => d,
                            Err(e) => {
                                if !committed {
                                    self.state.history.pop();
                                }
                                Err(e)?
                            }
                        };
                        acc = match acc.accumulate(delta.clone()) {
                            Ok(a) => a,
                            Err(e) => {
                                if !committed {
                                    self.state.history.pop();
                                }
                                Err(e)?
                            }
                        };
                        // Tag with the top-level metadata so accumulating these
                        // deltas reconstructs the same MessageOutput `run` yields.
                        delta.depth = Some(0);
                        self.stamp_source_agent(&mut delta.source_agent);
                        yield delta;
                    }
                }
                // LangModel::run_stream closes the contract — every message ends
                // with a finish_reason delta (a synthesized Stop if the provider
                // sent none) — so acc always carries one here. finish() promotes
                // Stop to ToolCall if tool calls were produced.
                let mut output = match acc.finish() {
                    Ok(o) => o,
                    Err(e) => {
                        if !committed {
                            self.state.history.pop();
                        }
                        Err(e)?
                    }
                };

                // Capture token usage for next iteration's truncation check.
                if let Some(u) = &output.usage {
                    self.state.last_input_tokens = Some(u.input_tokens);
                }

                output.depth = Some(0);
                self.state.history.push(output.message.clone());
                committed = true;

                // The assistant turn was already streamed as deltas above; drive
                // the loop off its finish_reason without re-emitting it.
                let tool_calls = match &output.finish_reason {
                    FinishReason::ToolCall {} => {
                        output.message.tool_calls.clone().unwrap_or_default()
                    }
                    _ => break,
                };

                self.start_console().await?;

                // See `run`: drained to the end even on failure, so `stop` is not
                // stepped over by an early `?`.
                let mut tool_stream = self.execute_tool_calls(tool_calls)?;
                let mut failure = None;
                while let Some(event) = tool_stream.next().await {
                    match event {
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                        Ok(mut output) => {
                            // Tool results are complete MessageOutputs; commit to
                            // history, stamp, then re-emit on the delta stream.
                            if output.message.role == Role::Tool && output.depth == Some(0) {
                                output.message = Self::cap_tool_result(output.message);
                                self.state.history.push(output.message.clone());
                            }
                            self.stamp_source_agent(&mut output.source_agent);
                            yield output.into();
                        }
                    }
                }
                drop(tool_stream);

                let stopped = self.stop_console().await;
                if let Some(e) = failure {
                    Err(e)?;
                }
                stopped?;
            }
        })
    }
}

// The `DummyConsoleExt` stand-in that used to live here is gone: a pure tool is now
// run through `ToolFunc::call_pure`, which asks for no console at all. There is no
// fabricating a `cortex` one anyway — it is a live session with a server behind it.

#[cfg(test)]
mod tests {
    use cortex::console::TreeRole;
    use futures::StreamExt as _;

    use super::*;
    use crate::{
        agent::{AgentCard, AgentProvider, AgentSpec, ContextManager, get_agent_providers_mut},
        datatype::Value,
        lang_model::{LangModelProvider, get_lm_providers_mut},
        message::{Message, Part, PartDelta, Role},
        suppress_panics, to_value,
        tool::{ToolDescBuilder, ToolProvider, get_tool_providers_mut},
        tool_func,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Load `.env`, then re-register the `"default"` lang-model provider so
    /// that any `*_API_KEY` introduced by `dotenv` is picked up — the
    /// `LangModelProvider::default()` that backs the global LazyLock may have
    /// been initialised before `dotenv` ran.
    fn refresh_default_lang_models() {
        dotenvy::dotenv().ok();
        get_lm_providers_mut().insert("default".to_string(), LangModelProvider::default());
    }

    /// Register an `AgentProvider` under `unique_name` whose `tool_provider`
    /// points at a freshly-registered `ToolProvider` configured by `build`.
    /// Reuses the `"default"` lang-model provider.  Returns the registry name
    /// to pass into `Agent::try_with_provider*`.
    fn provider_with_tools(unique_name: &str, build: impl FnOnce(&mut ToolProvider)) -> String {
        refresh_default_lang_models();
        let mut tp = ToolProvider::new();
        build(&mut tp);
        get_tool_providers_mut().insert(unique_name.to_string(), tp);
        get_agent_providers_mut().insert(
            unique_name.to_string(),
            AgentProvider::new("default", unique_name),
        );
        unique_name.to_string()
    }

    /// Refresh env-derived lang-model providers and return the `"default"`
    /// agent-provider name (auto-registered at startup).
    fn default_test_provider() -> &'static str {
        refresh_default_lang_models();
        "default"
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// A console whose trees are the directories named, each an empty temp dir.
    ///
    /// `None` for a role leaves that tree unmounted, which is how the sections are
    /// checked against a session that has only some of them.
    async fn console_with_trees(
        root: &std::path::Path,
        roles: &[TreeRole],
    ) -> cortex::console::Console {
        dotenvy::dotenv().ok();
        let program = std::env::var("AILOY_CORTEX_CONSOLE")
            .unwrap_or_else(|_| "cortex-local-console".to_string());

        let mut builder = cortex::console::Console::builder().stdio_client(&[&program]);
        for role in roles {
            let dir = root.join(role.as_str());
            std::fs::create_dir_all(&dir).unwrap();
            builder = match role {
                TreeRole::Context => builder.context(dir),
                TreeRole::Artifacts => builder.artifacts(dir),
                TreeRole::Scratch => builder.scratch(dir),
            };
        }

        let mut console = builder
            .build()
            .await
            .unwrap_or_else(|e| panic!("starting `{program}`: {e:#}"));
        console.start().await.expect("starting a test console");
        console
    }

    /// The whole system message, parts joined — what a provider ends up sending.
    fn system_text(agent: &Agent) -> String {
        agent
            .get_history()
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.contents.iter())
            .filter_map(|p| p.as_text())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Seed a turn's worth without calling a model: `seed_console_trees` is what the
    /// top of `run` and `run_stream` both do before anything else.
    async fn seeded(agent: &mut Agent) -> String {
        agent.seed_console_trees().await;
        system_text(agent)
    }

    /// A console that lent trees gets each one described, under the path the server
    /// answered, alongside the instruction rather than in place of it.
    #[tokio::test]
    async fn test_console_trees_are_described_in_the_system_message() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(
            dir.path(),
            &[TreeRole::Context, TreeRole::Artifacts, TreeRole::Scratch],
        )
        .await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        let text = seeded(&mut agent).await;
        assert!(text.contains("Be brief."), "the instruction survives");
        assert!(text.contains("# Context"), "{text}");
        assert!(text.contains("# Artifacts"), "{text}");
        assert!(text.contains("# Scratch"), "{text}");
        for role in ["context", "artifacts", "scratch"] {
            assert!(text.contains(role), "{role} is named by path: {text}");
        }

        // Exactly one system message: the trees join the instruction's, they do not
        // ship a second one for a provider to pick between.
        assert_eq!(
            agent
                .get_history()
                .iter()
                .filter(|m| m.role == Role::System)
                .count(),
            1
        );
    }

    /// The trees land in the *first* text part of the system message.
    ///
    /// Not a detail of layout: `openai`, `anthropic` and `gemini` each extract
    /// `contents.first()` and drop the rest, so a tree section written into a part of
    /// its own would reach chat-completions and nothing else — and would do it without
    /// an error anywhere.
    #[tokio::test]
    async fn test_trees_land_in_the_first_text_part() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(dir.path(), &[TreeRole::Artifacts]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();
        agent.seed_console_trees().await;

        let system = agent
            .get_history()
            .iter()
            .find(|m| m.role == Role::System)
            .expect("a system message");
        let first = system.contents.first().and_then(|p| p.as_text());
        let first = first.expect("its first part is text");

        assert!(
            first.starts_with("Be brief."),
            "the instruction still leads"
        );
        assert!(
            first.contains("# Artifacts"),
            "and the trees are in the same part: {first:?}"
        );
    }

    /// The section is prose, not source: no line carries the indentation the literals
    /// are written at, and no two words have been run together by a lost space.
    #[tokio::test]
    async fn test_tree_sections_carry_no_source_indentation() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(
            dir.path(),
            &[TreeRole::Context, TreeRole::Artifacts, TreeRole::Scratch],
        )
        .await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        let text = seeded(&mut agent).await;
        for line in text.lines() {
            assert!(
                !line.starts_with(' '),
                "line is indented like the literal it came from: {line:?}"
            );
            assert!(!line.contains("  "), "doubled space: {line:?}");
        }
        // The joins the fragments make, spelled out: a lost leading space shows up here
        // and nowhere else, since the result is still a perfectly plausible sentence.
        assert!(text.contains("information, the documents and data"));
        assert!(text.contains("the report, the figure, the file"));
        assert!(text.contains("lands here unless you move"));
    }

    /// A tree the caller never mounted is not described, and nothing sends the model
    /// to one: advice to move drafts into a scratch directory it has not got leaves it
    /// nowhere to put them.
    #[tokio::test]
    async fn test_unmounted_trees_are_not_described() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(dir.path(), &[TreeRole::Artifacts]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        let text = seeded(&mut agent).await;
        assert!(text.contains("# Artifacts"));
        assert!(!text.contains("# Context"));
        assert!(!text.contains("# Scratch"));
        assert!(
            !text.contains("scratch directory"),
            "no advice pointing at an unmounted tree: {text}"
        );
    }

    /// An agent with no console says nothing about where it works.
    #[tokio::test]
    async fn test_no_console_leaves_the_system_message_alone() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let mut agent = Agent::try_with_provider(spec, provider).unwrap();

        assert_eq!(seeded(&mut agent).await, "Be brief.");
    }

    /// A console that mounted nothing is the same case: nothing to say about where it
    /// stands, so nothing is said.
    #[tokio::test]
    async fn test_console_without_trees_leaves_the_system_message_alone() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(crate::test_console().await);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        assert_eq!(seeded(&mut agent).await, "Be brief.");
    }

    /// The caller's own system message wins on instruction and still learns the paths:
    /// which directory "the artifacts directory" is cannot be authored ahead of time.
    #[tokio::test]
    async fn test_trees_are_appended_to_a_caller_supplied_system_message() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(dir.path(), &[TreeRole::Artifacts]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("ignored");
        let state = AgentState::new()
            .with_history([Message::new(Role::System).with_contents([Part::text("Mine.")])])
            .with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        let text = seeded(&mut agent).await;
        assert!(text.contains("Mine."));
        assert!(!text.contains("ignored"), "the caller's message wins");
        assert!(text.contains("# Artifacts"), "and still learns the paths");
    }

    /// Every turn seeds, so seeding twice must be seeding once — otherwise a
    /// many-turn conversation stacks a copy of the section per turn.
    #[tokio::test]
    async fn test_trees_are_seeded_once_across_turns() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(dir.path(), &[TreeRole::Artifacts]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state).unwrap();

        agent.seed_console_trees().await;
        let text = seeded(&mut agent).await;
        assert_eq!(
            text.matches("# Artifacts").count(),
            1,
            "one copy of the section: {text}"
        );

        // And a second Agent built over the history the first produced does not add
        // another either.
        let rebuilt = AgentState::new().with_history(agent.get_history().to_vec());
        let mut rebuilt = Agent::try_with_provider_and_state(
            AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief."),
            provider,
            rebuilt.with_console_slot(agent.state.console.clone()),
        )
        .unwrap();
        assert_eq!(seeded(&mut rebuilt).await.matches("# Artifacts").count(), 1);
    }

    /// A sub-agent shares the parent's console slot, and so is told the same trees
    /// without anything having to be passed alongside it.
    #[tokio::test]
    async fn test_subagents_are_told_the_same_trees() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_trees(dir.path(), &[TreeRole::Artifacts]).await;

        let sub = AgentSpec::new("openai/gpt-4o-mini").instruction("Sub.");
        let parent = Agent::try_with_provider_and_state(
            AgentSpec::new("openai/gpt-4o-mini").instruction("Parent."),
            provider,
            AgentState::new().with_console(console),
        )
        .unwrap();

        // Materialised the way the sub-agent ToolFunc does it: the parent's slot, and
        // nothing else.
        let mut child = Agent::try_with_provider_and_state(
            sub,
            provider,
            AgentState::new().with_console_slot(parent.state.console.clone()),
        )
        .unwrap();

        assert!(seeded(&mut child).await.contains("# Artifacts"));
    }

    /// Verifies `run_stream` emits multiple incremental `Delta` events whose
    /// accumulated text matches the final assistant `Message`.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_emits_text_deltas() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        let mut agent = Agent::try_with_provider(spec, provider).unwrap();

        let query =
            Message::new(Role::User).with_contents([Part::text("Reply with a short greeting.")]);

        let mut strm = agent.run_stream(query);
        let mut delta_text = String::new();
        let mut delta_count = 0usize;
        let mut acc = MessageDeltaOutput::new();
        while let Some(event) = strm.next().await {
            let d = event.unwrap();
            delta_count += 1;
            for p in &d.delta.contents {
                if let PartDelta::Text { text } = p {
                    delta_text.push_str(text);
                }
            }
            acc = acc.accumulate(d).unwrap();
        }

        assert!(
            delta_count > 1,
            "expected multiple deltas, got {delta_count}"
        );
        // Accumulating the same deltas reconstructs the assistant message.
        let msg = acc.finish().unwrap().message;
        let final_text: String = msg.contents.iter().filter_map(|p| p.as_text()).collect();
        assert!(!final_text.is_empty());
        assert_eq!(delta_text, final_text);
    }

    /// A model whose endpoint refuses connection, so the first model call fails
    /// deterministically and offline (no API key needed).
    fn unreachable_agent() -> Agent {
        use crate::lang_model::{LangModelAPISchema, LangModelProvider, get_lm_providers_mut};

        // Register a lang-model provider whose endpoint refuses connection, so
        // the first model call fails deterministically; pair it with the
        // auto-registered `"default"` tool provider.
        let mut lmp = LangModelProvider::new();
        lmp.insert_api(
            "test/*".into(),
            LangModelAPISchema::ChatCompletion,
            url::Url::parse("http://127.0.0.1:1/").unwrap(),
            None,
        );
        get_lm_providers_mut().insert("unreachable_lm".into(), lmp);
        get_agent_providers_mut().insert(
            "unreachable_agent".into(),
            AgentProvider::new("unreachable_lm", "default"),
        );
        Agent::try_with_provider(AgentSpec::new("test/model"), "unreachable_agent").unwrap()
    }

    /// A first model call that fails must roll the just-pushed user query back
    /// out of history. Otherwise the reused agent would start its next run with
    /// the query still dangling at the tail and push a second consecutive User
    /// message, which most providers reject.
    #[tokio::test]
    async fn test_run_stream_rolls_back_query_on_failure() {
        let mut agent = unreachable_agent();
        let query = Message::new(Role::User).with_contents([Part::text("hi")]);
        let mut errored = false;
        {
            let mut strm = agent.run_stream(query);
            while let Some(ev) = strm.next().await {
                errored |= ev.is_err();
            }
        }
        assert!(
            errored,
            "expected the failed model call to surface an error"
        );
        assert!(
            agent.get_history().is_empty(),
            "dangling query not rolled back: {:?}",
            agent.get_history()
        );
    }

    /// Same rollback guarantee for the non-streaming `run`.
    #[tokio::test]
    async fn test_run_rolls_back_query_on_failure() {
        let mut agent = unreachable_agent();
        let query = Message::new(Role::User).with_contents([Part::text("hi")]);
        let mut errored = false;
        {
            let mut strm = agent.run(query);
            while let Some(ev) = strm.next().await {
                errored |= ev.is_err();
            }
        }
        assert!(
            errored,
            "expected the failed model call to surface an error"
        );
        assert!(
            agent.get_history().is_empty(),
            "dangling query not rolled back: {:?}",
            agent.get_history()
        );
    }

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
        let provider = provider_with_tools("test_simple_tool_call", |tp| {
            tp.insert_func("temperature", temperature_fn);
        });

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
        let provider = default_test_provider();

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

        let mut main_agent = Agent::try_with_provider(main_spec, provider).unwrap();

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
        let provider = default_test_provider();

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

        let mut main_agent = Agent::try_with_provider(main_spec, provider).unwrap();

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

        let provider = provider_with_tools("test_tool_panic_causes_inconsistent_history", |tp| {
            tp.insert_func("get_weather", good_fn);
            tp.insert_func("get_traffic", bad_fn);
        });

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
        let provider = provider_with_tools(
            "test_context_manager_truncates_tool_results_when_threshold_exceeded",
            |tp| {
                tp.insert_func("dummy_tool", dummy_fn);
            },
        );

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
        let provider = provider_with_tools(
            "test_context_manager_no_truncation_when_below_threshold",
            |tp| {
                tp.insert_func("dummy_tool", dummy_fn);
            },
        );

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

    /// The cycle `run` performs once per batch of tool calls: boot, run, release,
    /// and boot again for the next batch.
    ///
    /// What makes it safe is that `stop` releases the backend without ending the
    /// session — so what a tool wrote in one batch is still there in the next. The
    /// whole start/stop-per-batch design rests on that, which is why it is pinned
    /// here rather than assumed.
    #[tokio::test]
    async fn a_console_survives_the_stop_start_cycle_between_tool_batches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("across.txt").display().to_string();

        let console = crate::test_console().await;
        let agent_state = AgentState::new().with_console(console);

        // First batch.
        {
            let mut guard = agent_state.console.lock().await;
            let console = guard.as_mut().unwrap();
            console.start().await.expect("first start");
            console
                .exec(["sh", "-c", &format!("echo batch-1 > {path}")], None)
                .await
                .expect("first batch");
            console.stop().await.expect("release between batches");
        }

        // Second batch, on the same console: `stop` released the backend but not the
        // session, so this is a fresh boot rather than a new conversation.
        {
            let mut guard = agent_state.console.lock().await;
            let console = guard.as_mut().unwrap();
            console.start().await.expect("second start");
            let seen = console
                .exec(["cat", &path], None)
                .await
                .expect("second batch");
            assert_eq!(
                String::from_utf8_lossy(&seen.stdout).trim(),
                "batch-1",
                "what the first batch wrote must outlive the stop between them"
            );
            console.stop().await.expect("release after the last batch");
        }
    }

    /// An agent with no console runs its pure tools and says so plainly when a
    /// console tool is called — nothing builds one on its behalf.
    #[tokio::test]
    async fn an_agent_without_a_console_says_so_rather_than_building_one() {
        let state = AgentState::new();
        assert!(
            state.console.lock().await.is_none(),
            "a fresh state has no console, and nothing fills it in"
        );
    }
}
