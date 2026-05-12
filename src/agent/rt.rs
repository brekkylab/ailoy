use std::{collections::HashMap, path::PathBuf, pin::Pin, sync::Arc};

use futures::{FutureExt as _, Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec, ContextManager, default_provider},
    lang_model::LangModel,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    runenv::{Dirent, FileEntry, Local, RunEnv},
    skill::{SkillMeta, discover_skills, render_skills_table, scan_declared_skills},
    tool::{
        ToolDesc, ToolFunc,
        r#impl::{get_subagent_tool_desc, get_subagent_tool_func},
    },
};

/// Root directory where agent skill files are materialised inside the runenv.
const SKILLS_ROOT: &str = "/workspace/skills";

/// Walk the spec tree and write every declared file (this agent's plus the
/// subtree's) to the runenv with **write-once** semantics: if a file already
/// exists at the target path, the existing content is left untouched so that
/// runtime modifications survive subsequent invocations.
fn materialise_files_recursive<'a>(
    spec: &'a AgentSpec,
    runenv: &'a dyn RunEnv,
) -> Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>> {
    Box::pin(async move {
        for f in &spec.files {
            // Write-once: skip if the file already exists.
            if runenv.read(&f.path).await.is_ok() {
                continue;
            }
            f.write_to(runenv).await?;
        }
        for sub in &spec.subagents {
            materialise_files_recursive(sub, runenv).await?;
        }
        Ok(())
    })
}

/// Recursively walk `dir` inside `runenv` and collect every regular file
/// as a [`FileEntry`].  Returns an empty vec when `dir` does not exist.
async fn scan_files_recursive(
    runenv: &dyn RunEnv,
    dir: &std::path::Path,
) -> anyhow::Result<Vec<FileEntry>> {
    let mut out = Vec::new();
    let mut stack: Vec<PathBuf> = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let entries = match runenv.ls(&cur).await {
            Ok(es) => es,
            Err(_) => continue, // missing dir is fine
        };
        for entry in entries {
            match entry {
                Dirent::Dir { name, .. } => {
                    stack.push(cur.join(name));
                }
                Dirent::File { name, .. } => {
                    let path = cur.join(name);
                    if let Ok(fe) = FileEntry::read_from(runenv, path).await {
                        out.push(fe);
                    }
                }
            }
        }
    }
    Ok(out)
}

pub struct AgentState {
    pub history: Vec<Message>,

    pub runenv: Arc<dyn RunEnv>,

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
            runenv: Arc::new(Local {}),
            last_input_tokens: None,
        }
    }

    pub fn history(mut self, history: Vec<Message>) -> Self {
        self.history = history;
        self
    }

    pub fn runenv(mut self, runenv: Arc<dyn RunEnv>) -> Self {
        self.runenv = runenv;
        self
    }
}

/// An agent that drives a language model through multi-turn, tool-augmented conversations.
///
/// `Agent` pairs an [`AgentSpec`] (model + instruction + tools + sub-agents) with an
/// [`AgentProvider`] (credentials + tool sources) and an internal [`AgentState`]
/// (message history + [`RunEnv`]).  Call [`Agent::run`] to stream a single turn; tool
/// calls are resolved automatically and the conversation is appended to history after
/// each turn.
///
/// Sub-agents declared in [`AgentSpec::subagents`] are materialised at construction time
/// and registered as callable tools, inheriting the parent's [`RunEnv`] so they share
/// filesystem state.
///
/// For construction options, see [`Agent::try_new`], [`Agent::try_with_provider`],
/// [`Agent::try_with_runenv`], and [`Agent::try_with_provider_and_runenv`].
pub struct Agent {
    model: LangModel,

    tool_descs: Vec<ToolDesc>,

    tools: HashMap<String, ToolFunc>,

    pub state: AgentState,

    context_manager: Option<ContextManager>,

    /// The spec this agent was built from.  Carries the agent's identity,
    /// declared files, and its [`AgentSpec::skill_dir`] — the root for skill
    /// materialisation and discovery.  Used by [`Agent::snapshot`] as the
    /// template for non-skill fields (model, instruction, tools, card,
    /// subagents).
    spec: AgentSpec,

    /// Lazy gate: whether the declared [`FileEntry`](crate::runenv::FileEntry)
    /// list for this agent (and its declared sub-spec subtree) has been
    /// written to the runenv.  Toggled by [`Self::ensure_files_materialised`]
    /// on first [`Self::run`] or [`Self::snapshot`].
    files_materialised: bool,
}

impl Agent {
    /// Create an agent using the process-wide [`default_provider`] and a [`Local`] runenv.
    pub fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        let provider = default_provider();
        Self::try_with_provider(spec, &provider)
    }

    /// Create an agent using the process-wide [`default_provider`] and an explicit [`RunEnv`].
    pub fn try_with_runenv(spec: AgentSpec, runenv: Arc<dyn RunEnv>) -> anyhow::Result<Self> {
        let provider = default_provider();
        Self::try_with_provider_and_runenv(spec, &provider, runenv)
    }

    /// Create an agent with an explicit [`AgentProvider`] and a [`Local`] runenv.
    ///
    /// Use this for scoped, explicit control over models and tool sources without
    /// touching global state.  The provider is not stored in the agent and can be
    /// reused across multiple agents.
    pub fn try_with_provider(spec: AgentSpec, provider: &AgentProvider) -> anyhow::Result<Self> {
        let runenv: Arc<dyn RunEnv> = Arc::new(Local {});
        Self::try_with_provider_and_runenv(spec, provider, runenv)
    }

    /// Create an agent with an explicit [`AgentProvider`] and [`RunEnv`].
    ///
    /// The full constructor.  The supplied `runenv` is stored in [`AgentState::runenv`]
    /// and is also cloned into every sub-agent declared in [`AgentSpec::subagents`], so
    /// the parent and its sub-agents observe the same filesystem and process state.
    ///
    /// Each agent's skill files (declared in [`AgentSpec::files`]) are materialised
    /// lazily on first [`Self::run`] / [`Self::snapshot`] under
    /// [`AgentSpec::skill_dir`] (defaults to `/workspace/skills`).  Sub-agents'
    /// skill directories are re-rooted at `<skill_dir>/<card.name>` here, so any
    /// `skill_dir` value declared on a sub-spec is overwritten with the nested
    /// layout.
    pub fn try_with_provider_and_runenv(
        spec: AgentSpec,
        provider: &AgentProvider,
        runenv: Arc<dyn RunEnv>,
    ) -> anyhow::Result<Self> {
        Self::try_with_provider_and_runenv_and_skill_dir(
            spec,
            provider,
            runenv,
            PathBuf::from(SKILLS_ROOT),
        )
    }

    /// Internal constructor that takes the per-agent `skill_dir`.  Sub-agents
    /// register a [`ToolFunc`] that builds a fresh `Agent` per call with their
    /// own nested skill directory; there is no cascade.  Exposed to
    /// `crate::tool_impl` so sub-agent ToolFuncs can construct with the right
    /// skill directory.
    pub(crate) fn try_with_provider_and_runenv_and_skill_dir(
        spec: AgentSpec,
        provider: &AgentProvider,
        runenv: Arc<dyn RunEnv>,
        skill_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        // Resolve LangModel from the registry (handles glob lookup + prefix stripping)
        let model = provider.models.provide(&spec.model)?;

        // Collect tools required by the spec; error if any tool is missing.
        // `spec.tools` is cloned (rather than moved) so the full `spec` can be
        // retained on `self.spec` for snapshot + `.spec()` access.
        let mut tools = provider.tools.provide(&spec.tools)?;
        let mut tool_descs = spec.tools.clone();

        // Sub-agents become regular tool entries: each is a one-shot ToolFunc
        // that materialises a fresh Agent on call and shares the parent's
        // runenv so filesystem state is shared.  Each sub-agent's own files
        // live under a nested directory keyed by its card name; the cloned
        // sub-spec's `skill_dir` is overwritten with that nested path so the
        // ToolFunc's Agent picks it up via the canonical constructor.  Files
        // themselves are *not* written to disk here — they're materialised
        // lazily on first run/snapshot via [`Self::ensure_files_materialised`].
        for sub_spec in &spec.subagents {
            let card = sub_spec
                .card
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("subagent must declare an AgentCard"))?;
            let desc = get_subagent_tool_desc(card);
            let name = card.name.clone();
            let mut nested_spec = sub_spec.clone();
            nested_spec.skill_dir = spec.skill_dir.join(&card.name);
            let func = get_subagent_tool_func(nested_spec, provider.clone(), runenv.clone());
            tool_descs.push(desc);
            tools.insert(name, func);
        }

        // Build the system message: instruction + (optionally) the skills table.
        // The table is rendered from declared `spec.files` filtered to the
        // `<skill_dir>/<name>/SKILL.md` pattern.
        let declared_skills = scan_declared_skills(&spec.files, &spec.skill_dir);
        let skills_block = render_skills_table(&declared_skills, &spec.skill_dir);
        let system_text = match (spec.instruction.as_deref(), skills_block) {
            (Some(inst), Some(block)) => Some(format!("{inst}\n\n{block}")),
            (Some(inst), None) => Some(inst.to_string()),
            (None, Some(block)) => Some(block),
            (None, None) => None,
        };
        let history = system_text
            .map(|t| vec![Message::new(Role::System).with_contents([Part::text(t)])])
            .unwrap_or_default();

        let state = AgentState::new().history(history).runenv(runenv);

        Ok(Self {
            model,
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
    /// built-in bash tool so that *all* tool results stay within a consistent bound.
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
    /// declared sub-spec subtree) into the runenv on first call.  Uses
    /// write-once semantics — files that already exist are left untouched —
    /// so any runtime modifications survive subsequent invocations.
    async fn ensure_files_materialised(&mut self) -> anyhow::Result<()> {
        if self.files_materialised {
            return Ok(());
        }
        materialise_files_recursive(&self.spec, &*self.state.runenv).await?;
        self.files_materialised = true;
        Ok(())
    }

    /// Scan the runenv for skills owned by this agent.  Always re-scans
    /// (no in-process cache).  Triggers the materialise gate first, so a
    /// freshly built agent reports its declared skill set.
    pub async fn skills(&mut self) -> anyhow::Result<Vec<SkillMeta>> {
        self.ensure_files_materialised().await?;
        discover_skills(&*self.state.runenv, &self.spec.skill_dir).await
    }

    /// Capture the agent's *current* runtime state back into a serialisable
    /// [`AgentSpec`].
    ///
    /// Scope (combined):
    /// * The whole `skill_dir` subtree is walked and every file is captured
    ///   as a [`FileEntry`].  This picks up runtime modifications,
    ///   additions, and deletions inside the skill area.
    /// * Any *declared* [`AgentSpec::files`] whose path lives *outside* the
    ///   `skill_dir` subtree is re-read from disk (if still present) and
    ///   included as well.  Paths already covered by the subtree scan are
    ///   not duplicated.
    ///
    /// Sub-agents are kept as their **declared** specs (no recursion):
    /// upstream sub-agents are stateless ToolFunc closures with no
    /// persistent handle, so their runtime state isn't separately
    /// observable from the parent.
    pub async fn snapshot(&mut self) -> anyhow::Result<AgentSpec> {
        self.ensure_files_materialised().await?;

        let mut files = scan_files_recursive(&*self.state.runenv, &self.spec.skill_dir).await?;
        let covered: std::collections::HashSet<PathBuf> =
            files.iter().map(|f| f.path.clone()).collect();

        for declared in &self.spec.files {
            if covered.contains(&declared.path) {
                continue; // already captured by the subtree walk
            }
            if declared.path.starts_with(&self.spec.skill_dir) {
                continue; // would have been in the subtree if it still existed
            }
            if let Ok(fe) = FileEntry::read_from(&*self.state.runenv, declared.path.clone()).await {
                files.push(fe);
            }
            // Silently drop declared files that have been deleted on disk.
        }

        let mut out = self.spec.clone();
        out.files = files;
        Ok(out)
    }

    /// Read-only view of the spec this agent was built from.
    pub fn spec(&self) -> &AgentSpec {
        &self.spec
    }

    /// Read-only path to the agent's skill directory inside the runenv.
    /// Sourced from [`AgentSpec::skill_dir`] on the spec this agent was built
    /// from.
    pub fn skill_dir(&self) -> &std::path::Path {
        &self.spec.skill_dir
    }

    /// Execute tool calls in parallel and return a stream of all outputs.
    ///
    /// Synchronously spawns one tokio task per tool call, then returns a stream
    /// backed by an mpsc channel. Each task runs the tool, forwards intermediate
    /// sub-agent outputs (with bumped depth), and emits a final `Role::Tool`
    /// message at `depth = Some(0)`. Panics are caught via `catch_unwind` and
    /// converted to synthetic error tool results so the LM always receives
    /// exactly one result per call.
    ///
    /// Returns `Err` immediately (before spawning) if any tool name is not found.
    fn execute_tool_calls(
        &self,
        tool_calls: Vec<Part>,
    ) -> anyhow::Result<futures::stream::BoxStream<'static, anyhow::Result<MessageOutput>>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<anyhow::Result<MessageOutput>>();

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

            let runenv = self.state.runenv.clone();
            let tx = tx.clone();

            tokio::spawn(async move {
                // tx_inner is moved into AssertUnwindSafe and may be dropped
                // during a panic unwind. tx stays alive for the fallback send.
                let tx_inner = tx.clone();
                let tool_name_inner = tool_name.clone();
                let call_id_for_call = call_id.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        let mut stream = tool.call(call_args, call_id_for_call, &*runenv);
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
                            depth: Some(0),
                            message: err_msg,
                            finish_reason: FinishReason::Stop {},
                            usage: None,
                        }));
                    }
                }
            });
        }

        // Drop the original sender so the channel closes once all tasks finish.
        drop(tx);

        Ok(Box::pin(async_stream::stream! {
            let mut rx = rx;
            while let Some(event) = rx.recv().await {
                yield event;
            }
        }))
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
            // to the runenv on first call.  Write-once semantics preserve any
            // runtime modifications across subsequent runs.
            self.ensure_files_materialised().await?;

            self.state.history.push(query);

            loop {
                // Truncation check based on previous call's token usage.
                if let Some(cm) = &self.context_manager {
                    if self.state.last_input_tokens.unwrap_or(0) > cm.max_input_tokens {
                        cm.truncate_history(&mut self.state.history);
                    }
                }

                let mut output = self.model.run(&self.state.history, &self.tool_descs).await?;

                // Capture token usage for next iteration's truncation check.
                if let Some(u) = &output.usage {
                    self.state.last_input_tokens = Some(u.input_tokens);
                }

                output.depth = Some(0);
                self.state.history.push(output.message.clone());

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
                            yield output;
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "sandbox")]
    use std::sync::Arc;

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

        // Must contain at least one ToolCall assistant message.
        let has_tool_call = events
            .iter()
            .any(|e| e.message.role == Role::Assistant && e.message.tool_calls.is_some());
        assert!(
            has_tool_call,
            "Expected an assistant message with tool calls"
        );

        // Must contain at least one tool result.
        let has_tool_result = events.iter().any(|e| e.message.role == Role::Tool);
        assert!(has_tool_result, "Expected a tool result message");

        // Last event must be a final assistant text response.
        let last = events.last().expect("Expected at least one event");
        assert_eq!(last.message.role, Role::Assistant);
        assert!(
            last.message.contents.iter().any(|p| p.is_text()),
            "Last event should contain text"
        );
    }

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    /// Verifies that the main agent actually delegates to the in-memory subagent.
    ///
    /// Sets up a math subagent and registers it as a subagent tool on a coordinator agent.
    /// Asks a math question and confirms the main agent's history contains a [`Role::Tool`]
    /// message (proof that the subagent tool was called), and that the final reply is text.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_delegate_to_subagent() {
        let provider = get_provider();

        // Sub-agent: a minimal calculator that replies with just the numeric result.
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

        // Main agent: coordinator that should always delegate math to math-agent.
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

        // The history must contain a Tool message, confirming the subagent was called.
        let history = main_agent.get_history();
        assert!(
            history.iter().any(|m| m.role == Role::Tool),
            "Expected main agent history to contain a Tool message (subagent was called)"
        );

        // The final assistant message must contain text.
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

        // Sub-agent: gives a multi-step response so we see intermediate messages.
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
            // Intermediate sub-agent outputs: depth > 0.
            if output.depth.is_some_and(|d| d > 0) {
                tool_deltas += 1;
            }
            // Final tool result yielded back to the outer agent: Role::Tool at depth 0.
            if output.message.role == Role::Tool && output.depth == Some(0) {
                tool_results += 1;
            }
        }

        assert!(tool_results > 0, "Expected at least one tool result");
        assert!(
            tool_deltas > 0,
            "Expected at least one intermediate sub-agent output (tool delta)"
        );
    }

    /// Verifies that multiple tool calls in a single LLM response are executed
    /// and both results appear in history.
    ///
    /// Two async tools are registered:
    /// - `temperature_fast`: 50 ms delay, returns 15
    /// - `temperature_slow`: 300 ms delay, returns 25
    ///
    /// Note: tool calls are currently executed sequentially, so this test
    /// verifies correct sequential ordering rather than concurrency.
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_parallel_tool_calls() {
        use std::sync::{Arc as StdArc, Mutex as StdMutex};

        // Record the order in which tools complete.
        let completion_order: StdArc<StdMutex<Vec<&'static str>>> =
            StdArc::new(StdMutex::new(Vec::new()));

        let order_fast = completion_order.clone();
        let fast_desc = ToolDescBuilder::new("temperature_fast")
            .description("Get the current temperature in Tokyo (returns quickly)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let fast_fn = tool_func!(async |_args: Value| -> Value
            with [order = order_fast.clone()]
            {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                order.lock().unwrap().push("fast");
                to_value!(15u64)
            }
        );

        let order_slow = completion_order.clone();
        let slow_desc = ToolDescBuilder::new("temperature_slow")
            .description("Get the current temperature in Seoul (takes longer)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let slow_fn = tool_func!(async |_args: Value| -> Value
            with [order = order_slow.clone()]
            {
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                order.lock().unwrap().push("slow");
                to_value!(25u64)
            }
        );

        let mut provider = get_provider();
        provider.tools = ToolProvider::new();
        provider.tools.insert_func("temperature_fast", fast_fn);
        provider.tools.insert_func("temperature_slow", slow_fn);

        let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tools([fast_desc, slow_desc])
            .instruction(
                "When asked for temperatures in multiple cities, always call \
                 temperature_fast and temperature_slow in a single response."
                    .to_string(),
            );

        let mut agent = Agent::try_with_provider(spec, &provider).unwrap();

        let query = Message::new(Role::User).with_contents([Part::text(
            "Get the temperature in Tokyo using temperature_fast \
             and in Seoul using temperature_slow.",
        )]);

        {
            let mut strm = agent.run(query);
            while let Some(event) = strm.next().await {
                event.unwrap();
            }
        }

        // Verify both tools were actually called.
        let order = completion_order.lock().unwrap();
        assert!(
            order.contains(&"fast") && order.contains(&"slow"),
            "Both tools must have been called, got: {:?}",
            *order
        );

        // With sequential execution fast is called first, so it finishes first.
        assert_eq!(
            order.as_slice(),
            &["fast", "slow"],
            "Expected fast to complete before slow (sequential order): {:?}",
            order.as_slice()
        );

        // History must contain results for both tools.
        let history = agent.get_history();
        let tool_msgs: Vec<_> = history.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(
            tool_msgs.len(),
            2,
            "Expected exactly two Tool messages in history"
        );

        // First result is 15 (fast), second is 25 (slow).
        let first_value = tool_msgs[0]
            .contents
            .iter()
            .find_map(|p| p.as_value().cloned())
            .expect("Expected Value part in first tool result");
        let second_value = tool_msgs[1]
            .contents
            .iter()
            .find_map(|p| p.as_value().cloned())
            .expect("Expected Value part in second tool result");

        assert_eq!(
            first_value,
            Value::unsigned(15),
            "Fast tool result should be 15"
        );
        assert_eq!(
            second_value,
            Value::unsigned(25),
            "Slow tool result should be 25"
        );
    }

    /// Verifies history consistency when one of two parallel tool calls panics.
    ///
    /// Two tools are registered — one good, one that panics — and the model is
    /// instructed to call both in the same response.  The assertion checks that
    /// the number of tool-call entries in history equals the number of tool-result
    /// entries, so that history is never left in a half-written state.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_tool_panic_causes_inconsistent_history() {
        // This test intentionally panics inside a tool function.
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
                let _ = result; // ignore errors produced by the panic
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

        // 1. Every tool call must have a corresponding tool result — no silent drops.
        assert_eq!(
            tool_use_count, tool_result_count,
            "History is inconsistent: {} tool_use call(s) but {} tool result(s). \
             The panicked tool's result was silently lost.",
            tool_use_count, tool_result_count,
        );

        // 2. All tool result messages must use Part::Value, not Part::Text.
        //    Part::Text is marshalled as a JSON object by provider codecs, which
        //    breaks the `function_call_output.output` field (must be a string).
        for tool_msg in history.iter().filter(|m| m.role == Role::Tool) {
            for part in &tool_msg.contents {
                assert!(
                    part.as_value().is_some(),
                    "Tool result content must be Part::Value for correct API marshalling, \
                     but found a non-Value part. This causes provider API errors when the \
                     result is sent back to the model."
                );
            }
        }

        // 3. History must end with a final Assistant text response, not a tool call.
        //    If the second LM call fails (e.g. due to a malformed tool result), the
        //    stream terminates early and the agent never produces a closing answer.
        let last_msg = history.last().expect("History should not be empty");
        assert_eq!(
            last_msg.role,
            Role::Assistant,
            "History must end with an Assistant message, not {:?}",
            last_msg.role
        );
        assert!(
            last_msg.contents.iter().any(|p| p.is_text()),
            "Final Assistant message must contain text — the agent never produced a closing answer."
        );
    }

    /// Verifies that ContextManager replaces old tool results with "[context truncated]"
    /// when last_input_tokens exceeds max_input_tokens at the start of a run.
    ///
    /// History layout when truncation fires (after run() pushes the new query):
    ///   [0] sys  [1] u1  [2] a1_tc(old)  [3] tr_old  [4] u2  [5] a2_tc(recent)  [6] tr_recent  [7] u3
    /// With preserve_recent_turns=1 the boundary lands at index 4 (u2), so tr_old at
    /// index 3 is outside the preserve window and must become a placeholder.
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

        // Build two complete tool-call turns in history.
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
            max_input_tokens: 1, // always exceeded
            preserve_recent_turns: 1,
        }));
        agent.state.last_input_tokens = Some(9999);

        {
            let mut strm = agent.run(Message::new(Role::User).with_contents([Part::text("q3")]));
            while let Some(ev) = strm.next().await {
                ev.unwrap();
            }
        }

        let history = agent.get_history();

        // tr_old is outside the preserve boundary → must be "[context truncated]".
        let old_tool = history
            .iter()
            .find(|m| m.role == Role::Tool && m.id.as_deref() == Some(old_id))
            .expect("old Tool message must still exist in history");
        assert_eq!(
            old_tool.contents.first().and_then(|p| p.as_text()),
            Some("[context truncated]"),
            "tool result outside preserve window must become '[context truncated]'"
        );

        // tr_recent is inside the preserve boundary → original value must be intact.
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

        // Same two-turn history shape as the threshold test.
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

        // High threshold — will never be exceeded by the preset last_input_tokens.
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

        // tr_old must still hold its original value — no truncation should have fired.
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

    /// Verifies that parallel bash tool calls from the agent loop do not interleave
    /// inside the sandbox. The LLM is instructed to issue both bash calls in a single
    /// response. Each command writes "start_N", sleeps, then writes "end_N" to a shared
    /// log file. The sandbox Mutex guarantees serial execution, so no interleaving occurs.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_agent_parallel_bash_calls_are_serialized_in_sandbox() {
        use crate::runenv::{Sandbox, SandboxConfig};

        let mut provider = get_provider();
        provider.tools = ToolProvider::new();

        let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tool(crate::tool::r#impl::get_bash_tool_desc())
            .instruction(
                "You have a bash tool. When asked to run two commands, always call bash \
                 TWICE in a SINGLE response (parallel tool calls). Never run them sequentially \
                 across multiple turns.",
            );

        let runenv: Arc<dyn RunEnv> = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );
        let mut agent = Agent::try_with_provider_and_runenv(spec, &provider, runenv).unwrap();

        let log = "/tmp/agent_serial_test.txt";
        let query = Message::new(Role::User).with_contents([Part::text(format!(
            "Run these two shell commands in a single response using two parallel bash tool calls:\n\
             1. echo start_1 >> {log} && sleep 0.3 && echo end_1 >> {log}\n\
             2. echo start_2 >> {log} && sleep 0.3 && echo end_2 >> {log}"
        ))]);

        {
            let mut stream = agent.run(query);
            while let Some(event) = stream.next().await {
                event.expect("agent stream error");
            }
        }

        // Read the log via RunEnv — exec handles start/stop internally.
        let log_bytes = agent
            .state
            .runenv
            .read(std::path::Path::new(log))
            .await
            .expect("failed to read log");
        let log_content = String::from_utf8_lossy(&log_bytes).into_owned();

        let lines: Vec<&str> = log_content.lines().collect();
        assert_eq!(lines.len(), 4, "expected 4 log lines, got: {lines:?}");

        // Serial: start_N must be immediately followed by end_N (same N).
        let id0 = lines[0]
            .strip_prefix("start_")
            .expect("line 0 should be start_N");
        let id1 = lines[1]
            .strip_prefix("end_")
            .expect("line 1 should be end_N");
        assert_eq!(
            id0, id1,
            "first command's start and end must be adjacent — interleaving detected: {lines:?}"
        );

        let id2 = lines[2]
            .strip_prefix("start_")
            .expect("line 2 should be start_N");
        let id3 = lines[3]
            .strip_prefix("end_")
            .expect("line 3 should be end_N");
        assert_eq!(
            id2, id3,
            "second command's start and end must be adjacent — interleaving detected: {lines:?}"
        );
    }

    /// Verifies the convert_pdf_to_md skill end-to-end:
    ///   1. Agent receives an instruction listing available skills (name, description, path only).
    ///   2. Agent reads the SKILL.md via `bash cat` to activate the skill.
    ///   3. Agent installs Docling and converts the PDF to Markdown.
    ///
    /// Requires ANTHROPIC_API_KEY and the sandbox feature.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    #[ignore = "slow: installs docling (~minutes) and requires model artifacts"]
    async fn test_convert_pdf_to_md_skill() {
        use crate::runenv::{Sandbox, SandboxConfig};

        // Skill content hardcoded — not loaded from disk at test time.
        // This is what gets written to /workspace/skills/convert_pdf_to_md.md inside the sandbox.
        let skill_md = r#"# Skill: Convert PDF to Markdown

Convert a local PDF file to Markdown using [Docling](https://github.com/DS4SD/docling).

## When to use

When asked to convert a PDF file to Markdown (or extract text/structure from a PDF).

## Steps

### 1. Install dependencies

Install Docling (this takes a few minutes the first time):

```
pip install 'docling>=2,<3'
```

If the conversion later fails with an error related to `libxcb` or OpenCV display libraries,
replace the OpenCV build with the headless variant:

```
pip install --force-reinstall --no-deps opencv-python-headless
```

### 2. Run the conversion

Run the following script, substituting the actual paths:

```python
import logging
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("docling").setLevel(logging.CRITICAL)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

pipeline_options = PdfPipelineOptions(
    do_ocr=False,
    do_table_structure=True,
    table_structure_options=TableStructureOptions(do_cell_matching=True, mode="accurate"),
    accelerator_options={"num_threads": 4, "device": "auto"},
    do_picture_classification=False,
    do_picture_description=False,
    do_chart_extraction=False,
    do_code_enrichment=False,
    do_formula_enrichment=False,
    generate_page_images=False,
    generate_picture_images=False,
)

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

pdf_path = Path("/workspace/input.pdf")   # ← replace with actual path
output_path = pdf_path.with_suffix(".md")

markdown = converter.convert(pdf_path).document.export_to_markdown()
output_path.write_text(markdown, encoding="utf-8")
print(f"Saved: {output_path}")
```

### 3. Confirm

After the script exits with code 0, the Markdown file is written next to the PDF (same path,
`.md` extension) unless you specified a different `output_path`.
"#;

        // Instruction lists available skills by name, description, and SKILL.md path only.
        // The agent must `cat` the SKILL.md to obtain the full instructions before proceeding.
        let instruction = "\
You are a helpful assistant with access to a set of skills. \
Skills provide step-by-step instructions for specific tasks. \
To activate a skill, read its SKILL.md using the bash tool \
(`cat <path>`), then follow the instructions inside.

## Available Skills

| Name | Description | Path |
|------|-------------|------|
| convert_pdf_to_md | Convert a local PDF file to Markdown using Docling. | /workspace/skills/convert_pdf_to_md.md |
";

        let mut provider = get_provider();
        provider.tools = ToolProvider::new();

        let spec = AgentSpec::new("anthropic/claude-sonnet-4-6")
            .tools([
                crate::tool::r#impl::get_bash_tool_desc(),
                crate::tool::r#impl::get_python_repl_tool_desc(),
            ])
            .instruction(instruction);

        let runenv: Arc<dyn RunEnv> = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );
        let mut agent = Agent::try_with_provider_and_runenv(spec, &provider, runenv).unwrap();

        // Seed the sandbox with the skill file and the test PDF before running the agent.
        let pdf_bytes = minimal_pdf_bytes();
        // mkdir via exec — start/stop is handled internally by RunEnv.
        let _ = agent
            .state
            .runenv
            .exec(
                "sh".to_string(),
                vec!["-c".to_string(), "mkdir -p /workspace/skills".to_string()],
                None,
            )
            .await;
        agent
            .state
            .runenv
            .write(
                std::path::Path::new("/workspace/skills/convert_pdf_to_md.md"),
                skill_md.as_bytes(),
            )
            .await
            .expect("failed to write skill file into sandbox");
        agent
            .state
            .runenv
            .write(std::path::Path::new("/workspace/test.pdf"), &pdf_bytes)
            .await
            .expect("failed to write PDF into sandbox");

        // Ask the agent to convert the PDF — it must cat the SKILL.md first.
        let query = Message::new(Role::User).with_contents([Part::text(
            "Convert /workspace/test.pdf to Markdown. \
             The output file should be at /workspace/test.md.",
        )]);

        {
            let mut stream = agent.run(query);
            while let Some(event) = stream.next().await {
                println!("{:?}", event);
                event.expect("agent stream error");
            }
        }

        // Verify the markdown file was written to the sandbox.
        let markdown_bytes = agent
            .state
            .runenv
            .read(std::path::Path::new("/workspace/test.md"))
            .await
            .expect("agent should have written /workspace/test.md");
        let markdown = String::from_utf8_lossy(&markdown_bytes).into_owned();

        assert!(
            !markdown.trim().is_empty(),
            "converted markdown should not be empty"
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    #[cfg(feature = "sandbox")]
    fn minimal_pdf_bytes() -> Vec<u8> {
        let stream = "BT\n/F1 24 Tf\n72 100 Td\n(Hello Docling) Tj\nET";
        let objects = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>".to_string(),
            format!("<< /Length {} >>\nstream\n{stream}\nendstream", stream.len()),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut pdf = Vec::from("%PDF-1.4\n".as_bytes());
        let mut offsets = Vec::with_capacity(objects.len());
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, obj).as_bytes());
        }
        let xref = pdf.len();
        pdf.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for o in offsets {
            pdf.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        pdf.extend_from_slice(
            format!(
                "trailer\n<< /Root 1 0 R /Size {} >>\nstartxref\n{}\n%%EOF\n",
                objects.len() + 1,
                xref
            )
            .as_bytes(),
        );
        pdf
    }

    /// End-to-end: parent agent delegates to subagent via tool call, subagent writes
    /// a sentinel file in the shared sandbox, parent reads it back with its own bash tool
    /// and returns the content.  Proves the runenv passed to
    /// [`Agent::try_with_provider_and_runenv`] is propagated to spec subagents so they
    /// share the same VM.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_subagent_write_visible_to_parent_in_shared_sandbox() {
        use crate::runenv::{Sandbox, SandboxConfig};

        let mut provider = get_provider();
        provider.tools = ToolProvider::new();

        let sandbox = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );

        // Subagent: writes a file when asked, has bash tool + shared sandbox.
        let sub_spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tool(crate::tool::r#impl::get_bash_tool_desc())
            .instruction(
                "You are a file-writer agent. When asked to write content to a path, \
                 use the bash tool to do so (e.g. `echo CONTENT > PATH`). \
                 Confirm once the write succeeded.",
            )
            .card(AgentCard {
                name: "file_writer".into(),
                description: "Writes files to the sandbox filesystem.".into(),
                skills: vec![],
            });

        // Parent: delegates writing to the subagent, then reads back with bash.
        let main_spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tool(crate::tool::r#impl::get_bash_tool_desc())
            .instruction(
                "You are an orchestrator. You have a 'file_writer' subagent and a bash tool. \
                 When asked to verify shared sandbox state: \
                 1. Call the file_writer subagent to write the text 'sandbox_shared_ok' to \
                    /workspace/sentinel.txt. \
                 2. After it confirms, use your bash tool to run `cat /workspace/sentinel.txt`. \
                 3. Return the exact output of cat.",
            )
            .subagent(sub_spec);

        let runenv: Arc<dyn RunEnv> = sandbox.clone();
        let mut parent = Agent::try_with_provider_and_runenv(main_spec, &provider, runenv)
            .expect("parent build failed");

        let query =
            Message::new(Role::User).with_contents([Part::text("Verify shared sandbox state.")]);

        let mut stream = parent.run(query);
        while let Some(event) = stream.next().await {
            event.expect("agent stream error");
        }

        // The sentinel must be readable directly through the shared vm Arc,
        // confirming the subagent's write landed in the same VM.
        let result = sandbox
            .exec(
                "sh".to_string(),
                vec!["-c".to_string(), "cat /workspace/sentinel.txt".to_string()],
                None,
            )
            .await
            .expect("cat failed");
        assert!(
            result.stdout.contains("sandbox_shared_ok"),
            "sentinel file written by subagent not visible in parent's sandbox; stdout: {:?}",
            result.stdout,
        );
    }
}
