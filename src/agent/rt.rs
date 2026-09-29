use std::{collections::HashMap, path::Path, pin::Pin};

use futures::{FutureExt as _, Stream, StreamExt as _, stream::FuturesUnordered};

use crate::{
    agent::{
        AgentProvider, AgentSpec, AgentState, ContextManager, get_agent_providers,
        skill::render_skills,
        subagent::{get_subagent_tool_desc, get_subagent_tool_func},
    },
    lang_model::{LangModel, LangModelOptions},
    message::{Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, Role},
    tool::{
        ToolDesc, ToolFunc, get_tool_providers,
        r#impl::{
            get_mem_insert_tool_desc, get_mem_insert_tool_func, get_mem_search_tool_desc,
            get_mem_search_tool_func, get_web_search_tool_factory,
        },
    },
};

/// An agent that drives a language model through multi-turn, tool-augmented conversations.
///
/// `Agent` pairs an [`AgentSpec`] (model + instruction + tools + sub-agents) with an
/// [`AgentProvider`] (credentials + tool sources) and an [`AgentState`] (history +
/// console). [`Agent::run`] streams one turn, resolving tool calls and appending to history.
///
/// Sub-agents in [`AgentSpec::subagents`] are registered as callable tools that share the
/// parent's console.
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

    /// Spec's card name, used by [`Self::stamp_source_agent`] to tag streamed events.
    card_name: Option<String>,
}

impl Agent {
    /// Create an agent using the `"default"` entry of the process-wide
    /// [`get_agent_providers`] registry and a fresh [`AgentState`].
    pub async fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", AgentState::new()).await
    }

    /// Create an agent using the [`AgentProvider`] registered under `provider`
    /// in [`get_agent_providers`] and a fresh [`AgentState`].
    pub async fn try_with_provider(
        spec: AgentSpec,
        provider: impl AsRef<str>,
    ) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, provider, AgentState::new()).await
    }

    /// Create an agent using the `"default"` entry of the process-wide
    /// [`get_agent_providers`] registry and an explicit [`AgentState`].
    pub async fn try_with_state(spec: AgentSpec, state: AgentState) -> anyhow::Result<Self> {
        Self::try_with_provider_and_state(spec, "default", state).await
    }

    /// Create an agent using the [`AgentProvider`] registered under `provider`
    /// in [`get_agent_providers`] and an explicit [`AgentState`].
    ///
    /// The canonical constructor. Sub-agents share `state.console` and re-resolve
    /// `provider` from the registry on every invocation, so the name must stay
    /// registered for the agent's lifetime.
    ///
    /// Unless `state.history` already has a [`Role::System`] message, one built from
    /// `spec.instruction` and [`AgentSpec::skills`] is inserted at the front.
    ///
    /// Async because skills are read through the console (a read, not a boot: the
    /// session boots when the console is built).
    pub async fn try_with_provider_and_state(
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

        // Errors if any spec tool is missing.
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

        // Sub-specs are taken as-is (no path rewriting) so sub-agent skills are portable
        // across parents.
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

        // Built here rather than resolved from the ToolProvider: the store is this agent's
        // `Memory` value, not a registry name. The instruction is left untouched.
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

        // `any` rather than index 0: seeding a second system message would shadow or
        // duplicate a stray one.
        if !state.history.iter().any(|m| m.role == Role::System) {
            let skills = if spec.skills.is_empty() {
                None
            } else {
                Some(render_skills(&spec.skills, &state.console, &spec.model).await?)
            };
            let text = match (spec.instruction.as_deref(), skills) {
                (Some(instruction), Some(skills)) => Some(format!("{instruction}\n\n{skills}")),
                (instruction, skills) => skills.or(instruction.map(str::to_string)),
            };
            if let Some(text) = text {
                // Every provider schema expects the system message first.
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

    /// Tell the model which directories it has been given.
    ///
    /// Runs at the top of each turn and reads
    /// [`mounts`](cortex::console::ConsoleClient::mounts) from the console instead of
    /// caching a copy.
    ///
    /// Only paths are listed; what a mount is for belongs in the instruction. The section
    /// is appended to the system message, including a caller-written one, because mount
    /// locations cannot be known ahead of time.
    async fn seed_console_mounts(&mut self) {
        // Lock held only while reading the paths.
        let mounts: Vec<std::path::PathBuf> = {
            let guard = self.state.console.lock().await;
            let Some(console) = guard.as_ref() else {
                return;
            };
            console.mounts().map(Path::to_path_buf).collect()
        };

        if mounts.is_empty() {
            return;
        }

        let mut section = String::from(concat!(
            "\n\n# Mounts\n\n",
            "The directories you have been given, in the order they were mounted:\n",
        ));
        for path in &mounts {
            section.push_str(&format!("\n- `{}`", path.display()));
        }

        // Into the *first* text part: the openai, anthropic and gemini schemas keep only
        // `contents.first()` of the system message, silently dropping any later part.
        let Some(at) = self
            .state
            .history
            .iter()
            .position(|m| m.role == Role::System)
        else {
            // No system message at all: the mounts become one.
            self.state.history.insert(
                0,
                Message::new(Role::System).with_contents([Part::text(section.trim_start())]),
            );
            return;
        };

        let system = &mut self.state.history[at];
        match system.contents.iter_mut().find_map(|p| match p {
            Part::Text { text } => Some(text),
            _ => None,
        }) {
            // Seeding runs every turn and may run over a history another `Agent` seeded;
            // the section is deterministic, so it serves as its own idempotence marker.
            Some(text) => {
                if text.contains(section.as_str()) {
                    return;
                }
                text.push_str(&section);
            }
            // System message without text: the mounts lead it.
            None => system.contents.insert(0, Part::text(section.trim_start())),
        }
    }

    /// Per-part character limit for tool results before middle-truncation, keeping all
    /// tool results within the same bound as the built-in shell tool.
    const MAX_TOOL_RESULT_CHARS: usize = 30_000;

    /// Clamp every [`Part`] of a [`Role::Tool`] message so large payloads (e.g. web-search
    /// results) don't accumulate in history and trigger 429 rate-limit errors.
    ///
    /// An oversized `Part::Value` is measured as serialized JSON and replaced by a string
    /// value holding the truncated JSON.
    fn cap_tool_result(mut msg: Message) -> Message {
        for part in &mut msg.contents {
            match part {
                Part::Value { value } => {
                    let serialised = serde_json::to_string(value).unwrap_or_default();
                    if serialised.len() > Self::MAX_TOOL_RESULT_CHARS {
                        let truncated =
                            Self::middle_truncate(serialised, Self::MAX_TOOL_RESULT_CHARS);
                        *value = crate::datatype::Value::string(truncated);
                    }
                }
                Part::Text { text } if text.len() > Self::MAX_TOOL_RESULT_CHARS => {
                    *text =
                        Self::middle_truncate(std::mem::take(text), Self::MAX_TOOL_RESULT_CHARS);
                }
                _ => {}
            }
        }
        msg
    }

    /// Keep `max_chars` characters of `s` as head and tail around an omission notice.
    fn middle_truncate(s: String, max_chars: usize) -> String {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() <= max_chars {
            return s;
        }
        let head = max_chars / 2;
        let tail = max_chars - head;
        let omitted = chars.len() - head - tail;
        let head_str: String = chars[..head].iter().collect();
        let tail_str: String = chars[chars.len() - tail..].iter().collect();
        format!("{head_str}\n\n... [{omitted} characters omitted] ...\n\n{tail_str}")
    }

    /// Boot the console's backend for a batch of tool calls.
    ///
    /// Per batch rather than per agent, so the backend (possibly a whole micro-VM) is not
    /// idle while waiting on the model. `stop` keeps the session, so the next batch
    /// resumes the same console. A no-op without a console.
    async fn start_console(&self) -> anyhow::Result<()> {
        if let Some(console) = self.state.console.lock().await.as_mut() {
            console.start().await?;
        }
        Ok(())
    }

    /// Release what [`start_console`](Self::start_console) booted.
    ///
    /// Keeps the session open for the next batch's `start`.
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
    /// Console tools lock the shared console; pure tools (including sub-agents, which
    /// lock inside their nested `run()`) skip it. Panics are caught and turned into
    /// error tool results so the LM gets exactly one result per call.
    ///
    /// Returns `Err` before launching anything if a tool name is unknown.
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
                // The inner block moves its clone; the error path below needs this one.
                let tool_name_for_call = tool_name.clone();
                let call_id_for_call = call_id.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        if let Some(mut stream) =
                            tool.call_pure(call_args.clone(), call_id_for_call.clone())
                        {
                            // No console lock, so a sub-agent's nested `run()` cannot
                            // deadlock against the parent's tool batch.
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
                            // Lock held for the whole stream: the console protocol
                            // allows one outstanding request at a time.
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

        // So the channel closes once all futures finish.
        drop(tx);

        Ok(Box::pin(async_stream::stream! {
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

    /// Set `source_agent` to this agent's card name if unset, so the innermost sub-agent
    /// wins in nested chains. Takes the field to serve both output types.
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
            // Pop the query if the turn fails before its assistant message commits, so a
            // reused agent doesn't push two consecutive User messages (most providers
            // reject that).
            let mut committed = false;

            self.seed_console_mounts().await;

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

                // Collect failure instead of `?` so `stop` always runs and the backend is
                // not left booted.
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
                // A tool failure takes precedence over a stop failure.
                if let Some(e) = failure {
                    Err(e)?;
                }
                stopped?;
            }
        })
    }

    /// Token-streaming variant of [`run`](Self::run): yields the model's deltas, then each
    /// tool result as one complete delta. A `finish_reason` or role change marks a message
    /// boundary.
    pub fn run_stream(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageDeltaOutput>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {

            self.state.history.push(query);
            // Pop the query if the turn fails before its assistant message commits, so a
            // reused agent doesn't push two consecutive User messages (most providers
            // reject that).
            let mut committed = false;

            self.seed_console_mounts().await;

            loop {
                // Truncation check based on previous call's token usage.
                if let Some(cm) = &self.context_manager
                    && self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens {
                        cm.truncate_history(&mut self.state.history);
                    }

                // Accumulated for history and tool dispatch while forwarding each delta.
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
                        // So accumulated deltas equal the MessageOutput `run` yields.
                        delta.depth = Some(0);
                        self.stamp_source_agent(&mut delta.source_agent);
                        yield delta;
                    }
                }
                // `run_stream` always ends with a finish_reason (synthesized Stop if
                // needed); finish() promotes Stop to ToolCall when tool calls exist.
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

                // Already streamed above; not re-emitted.
                let tool_calls = match &output.finish_reason {
                    FinishReason::ToolCall {} => {
                        output.message.tool_calls.clone().unwrap_or_default()
                    }
                    _ => break,
                };

                self.start_console().await?;

                // Collect failure instead of `?` so `stop` always runs.
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

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{
        agent::{AgentCard, AgentProvider, AgentSpec, ContextManager, get_agent_providers_mut},
        datatype::Value,
        lang_model::{LangModelProvider, get_lm_providers_mut},
        memory::Memory,
        message::{Message, Part, PartDelta, Role},
        suppress_panics, to_value,
        tool::{ToolDescBuilder, ToolProvider, get_tool_providers_mut},
        tool_func,
    };

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Load `.env` and re-register the `"default"` lang-model provider, whose LazyLock
    /// may have initialised before `dotenv` set the `*_API_KEY`s.
    fn refresh_default_lang_models() {
        dotenvy::dotenv().ok();
        get_lm_providers_mut().insert("default".to_string(), LangModelProvider::default());
    }

    /// Register an `AgentProvider` named `unique_name` over the `"default"` lang models and
    /// a new `ToolProvider` configured by `build`; returns the name.
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

    /// Never called; for tests that only inspect a constructed agent.
    const DUMMY_MODEL: &str = "openai/gpt-4o-mini";

    fn dummy_provider(name: &'static str) -> &'static str {
        let mut lmps = get_lm_providers_mut();
        if !lmps.contains_key(name) {
            let mut lmp = LangModelProvider::new();
            lmp.insert(
                DUMMY_MODEL.into(),
                LangModelProvider::openai("dummy".into()),
            );
            lmps.insert(name.to_string(), lmp);
        }
        drop(lmps);
        get_agent_providers_mut()
            .entry(name.to_string())
            .or_insert_with(|| AgentProvider::new(name, "default"));
        name
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// A console with one writable mount per name, each an empty temp dir under `root`
    /// and mounted at itself.
    async fn console_with_mounts(
        root: &std::path::Path,
        names: &[&str],
    ) -> cortex::console::ConsoleClient {
        dotenvy::dotenv().ok();
        let mut builder = cortex::console::ConsoleClient::builder();
        for name in names {
            let dir = root.join(name);
            std::fs::create_dir_all(&dir).unwrap();
            builder = builder.mount(dir.clone(), dir);
        }

        let mut console = builder
            .build()
            .await
            .unwrap_or_else(|e| panic!("starting the console server: {e:#}"));
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

    /// Seed as the top of a turn does, without calling a model.
    async fn seeded(agent: &mut Agent) -> String {
        agent.seed_console_mounts().await;
        system_text(agent)
    }

    /// A console with mounts gets each one listed, under the path the server
    /// answered, alongside the instruction rather than in place of it.
    #[tokio::test]
    async fn test_console_mounts_are_listed_in_the_system_message() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_mounts(dir.path(), &["first", "second"]).await;
        let paths: Vec<String> = console.mounts().map(|p| p.display().to_string()).collect();
        assert_eq!(paths.len(), 2);

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state)
            .await
            .unwrap();

        let text = seeded(&mut agent).await;
        assert!(
            text.starts_with("Be brief."),
            "the instruction survives and leads"
        );
        assert!(text.contains("# Mounts"), "{text}");
        let at: Vec<usize> = paths
            .iter()
            .map(|p| {
                text.find(&format!("`{p}`"))
                    .unwrap_or_else(|| panic!("{p} in {text}"))
            })
            .collect();
        assert!(at[0] < at[1], "listed in the order mounted: {text}");

        // The mounts join the existing system message rather than adding one.
        assert_eq!(
            agent
                .get_history()
                .iter()
                .filter(|m| m.role == Role::System)
                .count(),
            1
        );
    }

    /// The mounts land in the *first* text part of the system message.
    ///
    /// `openai`, `anthropic` and `gemini` keep only `contents.first()`, so a separate part
    /// would be silently dropped.
    #[tokio::test]
    async fn test_mounts_land_in_the_first_text_part() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_mounts(dir.path(), &["work"]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state)
            .await
            .unwrap();
        agent.seed_console_mounts().await;

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
            first.contains("# Mounts"),
            "and the mounts are in the same part: {first:?}"
        );
    }

    /// An agent with no console says nothing about where it works.
    #[tokio::test]
    async fn test_no_console_leaves_the_system_message_alone() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let mut agent = Agent::try_with_provider(spec, provider).await.unwrap();

        assert_eq!(seeded(&mut agent).await, "Be brief.");
    }

    /// Nor does a console with no mounts.
    #[tokio::test]
    async fn test_console_without_mounts_leaves_the_system_message_alone() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(crate::test_console().await);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state)
            .await
            .unwrap();

        assert_eq!(seeded(&mut agent).await, "Be brief.");
    }

    /// A caller's system message replaces the instruction but still gets the mount paths.
    #[tokio::test]
    async fn test_mounts_are_appended_to_a_caller_supplied_system_message() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_mounts(dir.path(), &["work"]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("ignored");
        let state = AgentState::new()
            .with_history([Message::new(Role::System).with_contents([Part::text("Mine.")])])
            .with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state)
            .await
            .unwrap();

        let text = seeded(&mut agent).await;
        assert!(text.contains("Mine."));
        assert!(!text.contains("ignored"), "the caller's message wins");
        assert!(text.contains("# Mounts"), "and still learns the paths");
    }

    /// Seeding is idempotent, so multi-turn conversations don't stack copies.
    #[tokio::test]
    async fn test_mounts_are_seeded_once_across_turns() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_mounts(dir.path(), &["work"]).await;

        let spec = AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief.");
        let state = AgentState::new().with_console(console);
        let mut agent = Agent::try_with_provider_and_state(spec, provider, state)
            .await
            .unwrap();

        agent.seed_console_mounts().await;
        let text = seeded(&mut agent).await;
        assert_eq!(
            text.matches("# Mounts").count(),
            1,
            "one copy of the section: {text}"
        );

        // Also across an Agent rebuilt over the same history.
        let rebuilt = AgentState::new().with_history(agent.get_history().to_vec());
        let mut rebuilt = Agent::try_with_provider_and_state(
            AgentSpec::new("openai/gpt-4o-mini").instruction("Be brief."),
            provider,
            rebuilt.with_console_slot(agent.state.console.clone()),
        )
        .await
        .unwrap();
        assert_eq!(seeded(&mut rebuilt).await.matches("# Mounts").count(), 1);
    }

    /// A sub-agent shares the parent's console slot and so sees the same mounts.
    #[tokio::test]
    async fn test_subagents_are_told_the_same_mounts() {
        let provider = default_test_provider();
        let dir = tempfile::tempdir().unwrap();
        let console = console_with_mounts(dir.path(), &["work"]).await;

        let sub = AgentSpec::new("openai/gpt-4o-mini").instruction("Sub.");
        let parent = Agent::try_with_provider_and_state(
            AgentSpec::new("openai/gpt-4o-mini").instruction("Parent."),
            provider,
            AgentState::new().with_console(console),
        )
        .await
        .unwrap();

        // As the sub-agent ToolFunc builds it: the parent's slot only.
        let mut child = Agent::try_with_provider_and_state(
            sub,
            provider,
            AgentState::new().with_console_slot(parent.state.console.clone()),
        )
        .await
        .unwrap();

        assert!(seeded(&mut child).await.contains("# Mounts"));
    }

    /// Verifies `run_stream` emits multiple incremental `Delta` events whose
    /// accumulated text matches the final assistant `Message`.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_run_stream_emits_text_deltas() {
        let provider = default_test_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini");
        let mut agent = Agent::try_with_provider(spec, provider).await.unwrap();

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
    async fn unreachable_agent() -> Agent {
        use crate::lang_model::{LangModelAPISchema, LangModelProvider, get_lm_providers_mut};

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
        Agent::try_with_provider(AgentSpec::new("test/model"), "unreachable_agent")
            .await
            .unwrap()
    }

    /// A failed first model call rolls the user query back out of history, so a reused
    /// agent doesn't push two consecutive User messages.
    #[tokio::test]
    async fn test_run_stream_rolls_back_query_on_failure() {
        let mut agent = unreachable_agent().await;
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

    /// Query rollback on failure for the non-streaming `run`.
    #[tokio::test]
    async fn test_run_rolls_back_query_on_failure() {
        let mut agent = unreachable_agent().await;
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
        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

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

        let mut main_agent = Agent::try_with_provider(main_spec, provider).await.unwrap();

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

        let mut main_agent = Agent::try_with_provider(main_spec, provider).await.unwrap();

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

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

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

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

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

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

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

    /// The per-batch start/stop cycle relies on `stop` keeping the session, so one
    /// batch's writes are visible to the next.
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

        // Second batch: a fresh boot of the same session.
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

    /// A fresh state has no console, and nothing builds one implicitly.
    #[tokio::test]
    async fn an_agent_without_a_console_says_so_rather_than_building_one() {
        let state = AgentState::new();
        assert!(
            state.console.lock().await.is_none(),
            "a fresh state has no console, and nothing fills it in"
        );
    }

    /// A memory on the state yields the two memory tools without spec or ToolProvider entries.
    #[tokio::test]
    async fn test_memory_brings_its_two_tools() {
        let provider = dummy_provider("agent_rt_memory_tests");
        let state = AgentState::new().with_memory(Memory::new("/work/notes.sqlite"));
        let agent =
            Agent::try_with_provider_and_state(AgentSpec::new(DUMMY_MODEL), provider, state)
                .await
                .unwrap();

        let names: Vec<&str> = agent.tool_descs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, ["mem_search", "mem_insert"]);
        assert!(agent.tools.contains_key("mem_search"));
        assert!(agent.tools.contains_key("mem_insert"));
    }

    /// No memory, no memory tools.
    #[tokio::test]
    async fn test_no_memory_means_no_memory_tools() {
        let provider = dummy_provider("agent_rt_memory_tests");
        let agent = Agent::try_with_provider_and_state(
            AgentSpec::new(DUMMY_MODEL),
            provider,
            AgentState::new(),
        )
        .await
        .unwrap();

        assert!(agent.tool_descs.is_empty(), "{:?}", agent.tool_descs);
        assert!(agent.tools.is_empty());
    }
}
