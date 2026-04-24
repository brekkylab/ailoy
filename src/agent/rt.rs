use std::{pin::Pin, sync::Arc};

use futures::{FutureExt as _, Stream, StreamExt as _};

use crate::{
    agent::{AgentProvider, AgentSpec, default_provider},
    lang_model::LangModel,
    message::{FinishReason, Message, MessageOutput, Part, Role},
    sandbox::Sandbox,
    tool::{Tool, ToolContext, ToolSet},
};

pub struct AgentState {
    pub history: Vec<Message>,
    pub sandbox: Option<Arc<Sandbox>>,
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
            sandbox: None,
        }
    }

    pub fn with_history(history: Vec<Message>) -> Self {
        Self {
            history,
            sandbox: None,
        }
    }
}

/// An agent that drives a language model through multi-turn, tool-augmented conversations.
///
/// `Agent` pairs an [`AgentSpec`] (model + instruction + tools + sub-agents) with an
/// [`AgentProvider`] (credentials + tool sources) and an internal [`AgentState`]
/// (message history).  Call [`Agent::run`] to stream a single turn; tool calls are
/// resolved automatically and the conversation is appended to history after each turn.
///
/// For construction examples, see [`Agent::try_new`], [`Agent::try_with_provider`], and
/// [`Agent::try_with_tools`].
pub struct Agent {
    model: LangModel,
    tools: Vec<Tool>,
    pub state: AgentState,
}

impl Agent {
    /// Create an agent.
    ///
    /// Uses the process-wide [`default_provider`] for configuration. Configure it once
    /// at startup; all agents built with this method share those credentials and tool
    /// sources without passing a provider around.
    ///
    /// ```rust
    /// # use ailoy::{agent::{Agent, AgentSpec, default_provider_mut}, message::{Message, Part, Role}};
    /// # use futures::StreamExt as _;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// // One-time setup — configure the global provider before creating any agents.
    /// default_provider_mut().await.model_claude("ANTHROPIC_API_KEY");
    ///
    /// let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001");
    /// let agent = Agent::try_new(spec).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn try_new(spec: AgentSpec) -> anyhow::Result<Self> {
        let provider = default_provider().await;
        Self::try_with_provider(spec, &provider).await
    }

    /// Create an agent with an explicit [`AgentProvider`].
    ///
    /// Use this for scoped, explicit control over models and tool sources without
    /// touching global state.  The provider is not stored in the agent and can be
    /// reused across multiple agents.
    ///
    /// ```rust
    /// # use ailoy::agent::{Agent, AgentProvider, AgentSpec};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let mut provider = AgentProvider::new();
    /// provider.model_openai("OPENAI_API_KEY").tool_web_search();
    ///
    /// let spec = AgentSpec::new("openai/gpt-4o").tool("web_search");
    /// let agent = Agent::try_with_provider(spec, &provider).await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub async fn try_with_provider(
        spec: AgentSpec,
        provider: &AgentProvider,
    ) -> anyhow::Result<Self> {
        let tools = ToolSet::from_providers(&spec, provider).await?;
        Self::try_with_tools(spec, provider, &tools).await
    }

    /// Create an agent with a pre-built toolset, bypassing automatic tool-source initialisation.
    ///
    /// Use this when you need deterministic, in-process tools (e.g. unit tests, mock tools,
    /// or tools assembled at runtime from a [`ToolSet`]).
    ///
    /// ```rust
    /// # use ailoy::{agent::{Agent, AgentProvider, AgentSpec}, datatype::Value, message::ToolDescBuilder, to_value, tool::{ToolFactory, ToolFunc, ToolSet}};
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    ///     let mut tool_set = ToolSet::new();
    ///     tool_set.insert(
    ///         "temperature",
    ///         ToolFactory::simple(
    ///             ToolDescBuilder::new("temperature")
    ///                 .description("Return the temperature for a city")
    ///                 .parameters(to_value!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}))
    ///                 .build(),
    ///             ToolFunc::new(|_args: Value| Value::unsigned(25)),
    ///         ),
    ///     );
    ///
    ///     let mut provider = AgentProvider::new();
    ///     provider.model_openai("OPENAI_API_KEY");
    ///     let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
    ///     let agent = Agent::try_with_tools(spec, &provider, &tool_set).await?;
    /// #   Ok(())
    /// # }
    /// ```
    pub async fn try_with_tools(
        spec: AgentSpec,
        provider: &AgentProvider,
        tools: &ToolSet,
    ) -> anyhow::Result<Self> {
        // Parse model id
        let model_id = spec
            .model
            .split_once('/')
            .map(|(_, id)| id.to_string())
            .unwrap_or_else(|| spec.model.clone());
        // Resolve LangModel provider
        let model_provider = provider
            .get_model(&spec.model)
            .ok_or_else(|| anyhow::anyhow!("No provider found for model '{}'", spec.model))?
            .clone();

        // Collect tools required by the spec; error if any tool is missing
        let tools: Vec<Tool> = spec
            .tools
            .iter()
            .map(|n| {
                tools
                    .make_runtime(n, &spec)
                    .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", n))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Initialize history with system instruction if present
        let history = spec
            .instruction
            .as_ref()
            .map(|inst| vec![Message::new(Role::System).with_contents([Part::text(inst)])])
            .unwrap_or_default();

        #[allow(unused_mut)]
        let mut state = AgentState::with_history(history);
        #[cfg(feature = "sandbox")]
        {
            use std::sync::Arc;

            use crate::sandbox::Sandbox;
            if let Some(key) = spec.sandbox {
                if let Some(config) = provider.sandboxes.get(&key).cloned() {
                    let sandbox = Arc::new(
                        Sandbox::new(config)
                            .await
                            .expect("Failed to initialize sandbox"),
                    );
                    state.sandbox = Some(sandbox);
                } else {
                    return Err(anyhow::anyhow!("Sandbox '{}' not registered", key));
                }
            }
        }
        Ok(Self {
            model: LangModel::new(model_id, model_provider),
            tools,
            state,
        })
    }

    /// Assemble an `Agent` directly from runtime objects without going through
    /// `AgentSpec` or `AgentProvider`.  Used by [`crate::agent::AgentBuilder`].
    pub(crate) fn from_parts(model: LangModel, tools: Vec<Tool>, state: AgentState) -> Self {
        Self {
            model,
            tools,
            state,
        }
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
                .iter()
                .find(|t| t.get_desc().name == tool_name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No tool found for '{}'", tool_name))?;

            let mut ctx = ToolContext::new(call_id.clone());
            if let Some(sandbox) = &self.state.sandbox {
                ctx = ctx.sandbox(sandbox.clone());
            }

            let tx = tx.clone();

            tokio::spawn(async move {
                // tx_inner is moved into AssertUnwindSafe and may be dropped
                // during a panic unwind. tx stays alive for the fallback send.
                let tx_inner = tx.clone();
                let tool_name_inner = tool_name.clone();

                let outcome: Result<anyhow::Result<bool>, _> =
                    std::panic::AssertUnwindSafe(async move {
                        let mut stream = tool.call(call_args, ctx);
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

    /// Stream all events for a single agent turn.
    pub fn run(
        &mut self,
        query: Message,
    ) -> Pin<Box<impl Stream<Item = anyhow::Result<MessageOutput>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            self.state.history.push(query);
            let tool_descs: Vec<_> = self.tools.iter().map(|t| t.get_desc().clone()).collect();

            loop {
                let mut output = self.model.run(&self.state.history, &tool_descs).await?;
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
                        Ok(output) => {
                            if output.message.role == Role::Tool && output.depth == Some(0) {
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
    use std::sync::Arc;

    use futures::StreamExt as _;
    use tokio::sync::Mutex;

    use super::*;
    use crate::{
        agent::{AgentCard, AgentProvider, AgentSpec},
        datatype::Value,
        message::{Message, Part, Role, ToolDescBuilder},
        suppress_panics, to_value,
        tool::{ToolContext, ToolFactory, ToolFunc, ToolSet},
        tool_impl::make_subagent_tool_factory,
    };

    // ── helpers ───────────────────────────────────────────────────────────────
    fn get_provider() -> AgentProvider {
        dotenvy::dotenv().ok();
        let mut provider = AgentProvider::new();
        provider.model_openai(std::env::var("OPENAI_API_KEY").unwrap_or_default());
        provider.model_claude(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default());
        #[cfg(feature = "sandbox")]
        {
            use crate::sandbox::SandboxConfig;

            provider.sandbox("default", SandboxConfig::default());
        }
        provider
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    /// Verifies that the agent calls the temperature tool and returns a final answer.
    #[test_with::env(OPENAI_API_KEY)]
    #[tokio::test]
    async fn test_simple_tool_call() {
        // let tool_set = get_tool_set();
        let mut tool_set = ToolSet::new();
        tool_set.insert(
            "temperature",
            ToolFactory::simple(
                ToolDescBuilder::new("temperature")
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
                    .build(),
                ToolFunc::new(|_args: Value, _ctx: ToolContext| Value::unsigned(25)),
            ),
        );
        let provider = get_provider();
        let spec = AgentSpec::new("openai/gpt-4o-mini").tool("temperature");
        let mut agent = Agent::try_with_tools(spec, &provider, &tool_set)
            .await
            .unwrap();

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
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent = Agent::try_with_tools(sub_spec, &provider, &ToolSet::new())
            .await
            .unwrap();
        let sub_agent = Arc::new(Mutex::new(sub_agent));

        let card = AgentCard {
            name: "math-agent".to_string(),
            description:
                "Handles arithmetic and math computations. Use this for any math question."
                    .to_string(),
            skills: vec![],
        };
        let sub_tool = make_subagent_tool_factory(card, sub_agent);

        // Main agent: coordinator that should always delegate math to math-agent.
        let mut tool_set = ToolSet::new();
        tool_set.insert("math-agent", sub_tool);

        let mut main_agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini")
                .tool("math-agent")
                .instruction(
                    "You are a coordinator. For any arithmetic or math question, \
                     always delegate to the math-agent tool."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
        .unwrap();

        let query =
            Message::new(Role::User).with_contents([Part::text("What is 123 multiplied by 7?")]);

        {
            let mut strm = main_agent.run(query);
            while let Some(event) = strm.next().await {
                let event = event.unwrap();
                println!("{}", event);
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
        let sub_spec = AgentSpec::new("openai/gpt-4o-mini").instruction(
            "You are a calculator. Answer math questions with the numeric result only.".to_string(),
        );
        let sub_agent = Agent::try_with_tools(sub_spec, &provider, &ToolSet::new())
            .await
            .unwrap();
        let sub_agent = Arc::new(Mutex::new(sub_agent));

        let card = AgentCard {
            name: "math-agent".to_string(),
            description: "Handles arithmetic and math computations.".to_string(),
            skills: vec![],
        };
        let sub_tool = make_subagent_tool_factory(card, sub_agent);

        let mut tool_set = ToolSet::new();
        tool_set.insert("math-agent", sub_tool);

        let mut main_agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini").tool("math-agent"),
            &provider,
            &tool_set,
        )
        .await
        .unwrap();

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
        let fast_fn = ToolFunc::new(move |_args: Value, _ctx: ToolContext| {
            let order = order_fast.clone();
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                order.lock().unwrap().push("fast");
                to_value!(15u64)
            }
        });

        let order_slow = completion_order.clone();
        let slow_desc = ToolDescBuilder::new("temperature_slow")
            .description("Get the current temperature in Seoul (takes longer)")
            .parameters(to_value!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }))
            .build();
        let slow_fn = ToolFunc::new(move |_args: Value, _ctx: ToolContext| {
            let order = order_slow.clone();
            async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;
                order.lock().unwrap().push("slow");
                to_value!(25u64)
            }
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert("temperature_fast", ToolFactory::simple(fast_desc, fast_fn));
        tool_set.insert("temperature_slow", ToolFactory::simple(slow_desc, slow_fn));

        let provider = get_provider();
        let mut agent = Agent::try_with_tools(
            AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
                .tools(["temperature_fast", "temperature_slow"])
                .instruction(
                    "When asked for temperatures in multiple cities, always call \
                     temperature_fast and temperature_slow in a single response."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
        .unwrap();

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
        let good_fn = ToolFunc::new(|_args: Value, _ctx: ToolContext| async move {
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
        let bad_fn = ToolFunc::new(|_args: Value, _ctx: ToolContext| async move {
            panic!("simulated tool crash");
            #[allow(unreachable_code)]
            Value::null()
        });

        let mut tool_set = ToolSet::new();
        tool_set.insert("get_weather", ToolFactory::simple(good_desc, good_fn));
        tool_set.insert("get_traffic", ToolFactory::simple(bad_desc, bad_fn));

        let provider = get_provider();
        let mut agent = Agent::try_with_tools(
            AgentSpec::new("openai/gpt-4o-mini")
                .tools(["get_weather", "get_traffic"])
                .instruction(
                    "When asked about a city, ALWAYS call both get_weather AND \
                     get_traffic tools in a single response. Never call just one."
                        .to_string(),
                ),
            &provider,
            &tool_set,
        )
        .await
        .unwrap();

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

    /// Verifies the sandbox VM lifecycle using the `python_repl` builtin tool:
    ///
    /// 1. Stopped right after agent creation.
    /// 2. python_repl executes successfully (proves VM was running during the call).
    /// 3. Stopped again once the tool-call batch stream is exhausted.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_sandbox_lifecycle_with_python_repl() {
        use crate::{
            agent::AgentSpec,
            tool::{BuiltinToolProvider, ToolProvider},
        };

        let mut provider = get_provider();
        provider.tool(ToolProvider::Builtin(BuiltinToolProvider::PythonRepl {}));

        let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tool("python_repl")
            .instruction(
                "When asked to run Python code, always use the python_repl tool. \
                 Never skip the tool call.",
            )
            .sandbox("default");

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

        // ── 1. After creation: sandbox exists but is stopped ───────────────
        let sb = agent
            .state
            .sandbox
            .as_ref()
            .expect("sandbox should exist after construction");
        assert!(
            !sb.is_running().await,
            "sandbox should be stopped after construction"
        );

        // ── 2. Run a turn that triggers python_repl ─────────────────────────
        let query = Message::new(Role::User).with_contents([Part::text(
            "Run this Python code and tell me the output: print('ailoy_sandbox_ok')",
        )]);
        {
            let mut stream = agent.run(query);
            while let Some(event) = stream.next().await {
                event.unwrap();
            }
        }

        // ── 3. After run(): sandbox stopped ────────────────────────────────
        let sb = agent.state.sandbox.as_ref().unwrap();
        assert!(
            !sb.is_running().await,
            "sandbox should be stopped after run() completes"
        );

        // ── 2b. Verify python_repl actually ran inside the VM ───────────────
        let tool_result = agent
            .get_history()
            .iter()
            .find(|m| m.role == Role::Tool)
            .expect("history should contain a tool result");

        let stdout = tool_result
            .contents
            .iter()
            .find_map(|p| p.as_value())
            .and_then(|v| v.pointer("/stdout"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        assert!(
            stdout.contains("ailoy_sandbox_ok"),
            "python_repl should have produced expected output, got stdout: {stdout:?}"
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
        use futures::StreamExt as _;

        use crate::{
            agent::AgentSpec,
            message::{Message, Part, Role},
            tool::{BuiltinToolProvider, ToolProvider},
        };

        let mut provider = get_provider();
        provider.tool(ToolProvider::Builtin(BuiltinToolProvider::Bash {}));

        let spec = AgentSpec::new("anthropic/claude-haiku-4-5-20251001")
            .tool("bash")
            .instruction(
                "You have a bash tool. When asked to run two commands, always call bash \
                 TWICE in a SINGLE response (parallel tool calls). Never run them sequentially \
                 across multiple turns.",
            )
            .sandbox("default");

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

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

        // Read the log — shell() lazy-starts the VM.
        let sb = agent.state.sandbox.as_ref().unwrap();
        let log_content = sb
            .shell(&format!("cat {log}"))
            .await
            .expect("failed to read log");

        let lines: Vec<&str> = log_content.stdout.lines().collect();
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
        use crate::{
            agent::AgentSpec,
            tool::{BuiltinToolProvider, ToolProvider},
        };

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
        provider
            .tool(ToolProvider::Builtin(BuiltinToolProvider::Bash {}))
            .tool(ToolProvider::Builtin(BuiltinToolProvider::PythonRepl {}));

        let spec = AgentSpec::new("anthropic/claude-sonnet-4-6")
            .tools(["bash", "python_repl"])
            .instruction(instruction)
            .sandbox("default");

        let mut agent = Agent::try_with_provider(spec, &provider).await.unwrap();

        // Seed the sandbox with the skill file and the test PDF before running the agent.
        let pdf_bytes = minimal_pdf_bytes();
        let sb = agent.state.sandbox.as_ref().expect("sandbox must exist");
        sb.start().await.expect("failed to start sandbox for setup");
        sb.shell("mkdir -p /workspace/skills")
            .await
            .expect("failed to create skills directory");
        sb.write_file(
            "/workspace/skills/convert_pdf_to_md.md",
            skill_md.as_bytes(),
        )
        .await
        .expect("failed to write skill file into sandbox");
        sb.write_file("/workspace/test.pdf", &pdf_bytes)
            .await
            .expect("failed to write PDF into sandbox");
        sb.stop().await.expect("failed to stop sandbox after setup");

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
        agent
            .state
            .sandbox
            .as_ref()
            .unwrap()
            .start()
            .await
            .expect("failed to start sandbox for verification");

        let markdown = agent
            .state
            .sandbox
            .as_ref()
            .unwrap()
            .read_file("/workspace/test.md")
            .await
            .expect("agent should have written /workspace/test.md");

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
    /// and returns the content.  Proves sandbox state is shared across the agent boundary
    /// during a real multi-turn run.
    #[cfg(feature = "sandbox")]
    #[test_with::env(ANTHROPIC_API_KEY)]
    #[tokio::test]
    async fn test_subagent_write_visible_to_parent_in_shared_sandbox() {
        use futures::StreamExt as _;

        use crate::{
            agent::AgentBuilder,
            lang_model::{LangModel, LangModelProvider},
            message::{Message, Part, Role},
            sandbox::{Sandbox, SandboxConfig},
            tool::{BuiltinToolProvider, make_builtin_tool},
        };

        dotenvy::dotenv().ok();
        let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        let make_model = || {
            LangModel::new(
                "claude-haiku-4-5-20251001".to_string(),
                LangModelProvider::anthropic(api_key.clone()),
            )
        };

        let vm = Arc::new(
            Sandbox::new(SandboxConfig::default())
                .await
                .expect("sandbox creation failed"),
        );

        // Subagent: writes a file when asked, has bash tool + shared sandbox.
        let sub = AgentBuilder::new(make_model())
            .instruction(
                "You are a file-writer agent. When asked to write content to a path, \
                 use the bash tool to do so (e.g. `echo CONTENT > PATH`). \
                 Confirm once the write succeeded.",
            )
            .tool(make_builtin_tool(&BuiltinToolProvider::Bash {}))
            .sandbox(vm.clone())
            .build()
            .await
            .expect("subagent build failed");

        // Parent: delegates writing to the subagent, then reads back with bash.
        let mut parent = AgentBuilder::new(make_model())
            .instruction(
                "You are an orchestrator. You have a 'file_writer' subagent and a bash tool. \
                 When asked to verify shared sandbox state: \
                 1. Call the file_writer subagent to write the text 'sandbox_shared_ok' to \
                    /workspace/sentinel.txt. \
                 2. After it confirms, use your bash tool to run `cat /workspace/sentinel.txt`. \
                 3. Return the exact output of cat.",
            )
            .tool(make_builtin_tool(&BuiltinToolProvider::Bash {}))
            .sandbox(vm.clone())
            .subagent(
                AgentCard {
                    name: "file_writer".into(),
                    description: "Writes files to the sandbox filesystem.".into(),
                    skills: vec![],
                },
                sub,
            )
            .build()
            .await
            .expect("parent build failed");

        let query =
            Message::new(Role::User).with_contents([Part::text("Verify shared sandbox state.")]);

        let mut stream = parent.run(query);
        while let Some(event) = stream.next().await {
            event.expect("agent stream error");
        }

        // The sentinel must be readable directly through the shared vm Arc,
        // confirming the subagent's write landed in the same VM.
        let result = vm
            .shell("cat /workspace/sentinel.txt")
            .await
            .expect("cat failed");
        assert!(
            result.stdout.contains("sandbox_shared_ok"),
            "sentinel file written by subagent not visible in parent's sandbox; stdout: {:?}",
            result.stdout,
        );
    }
}
