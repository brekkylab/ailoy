//! A language model backed by the Codex CLI's app server (`codex app-server`).
//!
//! Each call starts one `codex app-server` process and talks JSON-RPC to it
//! over stdio, one message per line:
//!
//! 0. Once per [`CodexModel`], the model catalog is read (`codex debug models`)
//!    and rewritten so the models call tools directly (see [`direct_catalog`]);
//!    each process is started on it.
//! 1. `initialize`, opting into the experimental API, then `initialized`.
//! 2. `thread/start` with the system prompt as the base instructions, the
//!    tools as dynamic tools, so they are declared to the model like its own,
//!    and raw events on.
//! 3. `thread/inject_items` with the earlier turns as Responses API items,
//!    tool calls and results included.
//! 4. `turn/start` with the latest user message, or no input when the
//!    conversation ends with tool results.
//!
//! The reply streams in as `item/agentMessage/delta` notifications. Calls to
//! the tools are taken from the raw Responses API items
//! (`rawResponseItem/completed`), and the reply ends with them once the
//! response is complete (`rawResponse/completed`, which carries its usage);
//! the process is then stopped. Codex's own `item/tool/call` requests are left
//! unanswered: they come one at a time, since Codex runs dynamic tools in
//! turn, and its usage report waits for them.

use std::{path::Path, process::Stdio, sync::Arc};

use anyhow::Context as _;
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    process::{ChildStdin, Command},
    sync::OnceCell,
};

use crate::{
    experimental::model::{InferLangModel, LangModelOptions, ThinkingEffort},
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, Role, TokenUsage,
    },
    tool::ToolDesc,
};

/// Codex features that give the model tools of its own. Keys unknown to the
/// installed version are ignored by Codex with a warning.
const DISABLED_FEATURES: &[&str] = &[
    "shell_tool",
    "unified_exec",
    "js_repl",
    "code_mode",
    "apply_patch_freeform",
    "view_image",
    "sleep_tool",
    "search_tool",
    "request_permissions_tool",
    "multi_agent",
    "apps",
    "memories",
    "hooks",
    "goals",
    "image_generation",
];

/// Config overrides that keep Codex's own context out of the conversation.
const CONFIG: &[&str] = &[
    "web_search=\"disabled\"",
    "mcp_servers={}",
    "project_doc_max_bytes=0",
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    "tools.experimental_request_user_input.enabled=false",
];

/// Item types that mean the model used one of Codex's own tools.
const OWN_TOOL_ITEMS: &[&str] = &[
    "commandExecution",
    "fileChange",
    "mcpToolCall",
    "webSearch",
    "imageView",
    "imageGeneration",
    "collabAgentToolCall",
];

/// Runs GPT through the `codex` CLI's app server instead of calling the API directly.
///
/// The CLI's login (ChatGPT account or API key) is used as is. System messages
/// replace Codex's own instructions; without any, the instructions are empty.
/// Tools are declared to the model, so it calls them as it would any tool.
///
/// Limits of going through the app server:
/// - It is experimental, and the raw events it relies on are meant for
///   internal use, so it may change between Codex versions.
/// - Text only: image parts are rejected.
#[derive(Clone, Debug)]
pub struct CodexModel {
    program: String,
    model: Option<String>,
    /// The rewritten model catalog, read on the first call.
    catalog: Arc<OnceCell<String>>,
}

impl Default for CodexModel {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexModel {
    pub fn new() -> Self {
        Self {
            program: "codex".to_owned(),
            model: None,
            catalog: Arc::default(),
        }
    }

    /// Model name (e.g. `"gpt-5.5"`); unset keeps the CLI default.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Name or path of the `codex` executable.
    pub fn with_program(mut self, program: impl Into<String>) -> Self {
        self.program = program.into();
        self.catalog = Arc::default();
        self
    }

    /// The model catalog for direct tool calls, read from the CLI on first use.
    async fn catalog(&self) -> anyhow::Result<&str> {
        let catalog = self
            .catalog
            .get_or_try_init(|| async {
                let output = Command::new(&self.program)
                    .args(["debug", "models"])
                    .stdin(Stdio::null())
                    .output()
                    .await
                    .context("failed to start the codex CLI")?;
                if !output.status.success() {
                    anyhow::bail!(
                        "codex debug models failed with {}: {}",
                        output.status,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
                let catalog = serde_json::from_slice(&output.stdout)
                    .context("codex debug models printed no JSON catalog")?;
                Ok(serde_json::to_string(&direct_catalog(catalog))?)
            })
            .await?;
        Ok(catalog)
    }

    /// Builds the command for one call, without starting it.
    fn command(&self, workdir: &Path, catalog: &Path) -> anyhow::Result<Command> {
        let mut command = Command::new(&self.program);
        command.arg("app-server");
        for config in CONFIG {
            command.args(["-c", config]);
        }
        command.args([
            "-c".to_owned(),
            format!("model_catalog_json={}", serde_json::to_string(catalog)?),
        ]);
        for feature in DISABLED_FEATURES {
            command.args(["--disable", feature]);
        }
        command
            // An empty directory, so no AGENTS.md or project config is picked up.
            .current_dir(workdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Dropping a stream mid-reply ends the process with it.
            .kill_on_drop(true);
        Ok(command)
    }
}

/// The catalog with what makes Codex's models act as agents taken out of
/// each model:
/// - its tool mode (`code_mode_only` for the listed models, under which the
///   model writes code that calls the tools, instead of calling them);
/// - its multi-agent setup, which adds instructions about sub-agents;
/// - its experimental tools, such as a clock;
/// - Responses Lite, under which a reply holds one tool call at most.
fn direct_catalog(mut catalog: serde_json::Value) -> serde_json::Value {
    for model in catalog["models"].as_array_mut().into_iter().flatten() {
        let Some(model) = model.as_object_mut() else {
            continue;
        };
        model.remove("tool_mode");
        model.insert(
            "experimental_supported_tools".to_owned(),
            serde_json::json!([]),
        );
        model.insert("use_responses_lite".to_owned(), false.into());
        model.remove("multi_agent_version");
        model.remove("multi_agent_reasoning_effort");
        if let Some(messages) = model
            .get_mut("model_messages")
            .and_then(|m| m.as_object_mut())
        {
            messages.remove("multi_agent");
        }
    }
    catalog
}

impl InferLangModel for CodexModel {
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let mut deltas = self.infer_stream(messages, tools, options);
        Box::pin(async move {
            let mut acc = MessageDeltaOutput::new();
            while let Some(delta) = deltas.next().await {
                acc = acc.accumulate(delta?)?;
            }
            acc.finish()
        })
    }

    fn infer_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let plan = match Plan::new(messages, tools) {
            Ok(plan) => plan,
            Err(e) => return Box::pin(stream::once(async move { Err(e) })),
        };
        let this = self.clone();
        let mut turn = serde_json::json!({"input": plan.input});
        // Unset keeps the model's default effort.
        if let Some(effort) = options.thinking_effort {
            turn["effort"] = match effort {
                ThinkingEffort::Low => "low",
                ThinkingEffort::Medium => "medium",
                ThinkingEffort::High => "high",
            }
            .into();
        }
        if let Some(schema) = &options.output_schema {
            // Codex sends it in strict mode, which needs every object closed.
            turn["outputSchema"] = super::schema::close_objects(schema).into();
        }

        Box::pin(async_stream::try_stream! {
            let workdir = tempfile::tempdir()?;
            let catalog = workdir.path().join("models.json");
            tokio::fs::write(&catalog, this.catalog().await?).await?;
            let mut child = this
                .command(workdir.path(), &catalog)?
                .spawn()
                .context("failed to start the codex CLI")?;
            let mut stdin = child.stdin.take().expect("stdin is piped");
            let stdout = child.stdout.take().expect("stdout is piped");
            let mut stderr = child.stderr.take().expect("stderr is piped");
            // Drained alongside stdout so a full stderr pipe cannot stall the process.
            let stderr = tokio::spawn(async move {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf).await;
                buf
            });
            let mut lines = BufReader::new(stdout).lines();

            send(&mut stdin, serde_json::json!({
                "id": 0,
                "method": "initialize",
                "params": {
                    "clientInfo": {"name": "ailoy", "title": null, "version": env!("CARGO_PKG_VERSION")},
                    // Dynamic tools and item injection are experimental.
                    "capabilities": {"experimentalApi": true, "requestAttestation": false},
                },
            })).await?;
            response(&mut lines, 0).await?;
            send(&mut stdin, serde_json::json!({"method": "initialized"})).await?;

            let mut thread = serde_json::json!({
                "cwd": workdir.path(),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                // No environment: the model cannot reach the machine.
                "environments": [],
                "ephemeral": true,
                // Always set, even empty: otherwise Codex's own coding-agent instructions are used.
                "baseInstructions": plan.instructions,
                "dynamicTools": plan.tools,
                // Every tool call and the response's usage, as soon as the response completes.
                "experimentalRawEvents": true,
            });
            if let Some(model) = &this.model {
                thread["model"] = model.as_str().into();
            }
            send(&mut stdin, serde_json::json!({"id": 1, "method": "thread/start", "params": thread})).await?;
            let thread_id = response(&mut lines, 1).await?["thread"]["id"]
                .as_str()
                .context("thread/start returned no thread id")?
                .to_owned();

            if !plan.history.is_empty() {
                send(&mut stdin, serde_json::json!({
                    "id": 2,
                    "method": "thread/inject_items",
                    "params": {"threadId": thread_id, "items": plan.history},
                })).await?;
                response(&mut lines, 2).await?;
            }

            // Its response is checked by the reader, along with the turn's events.
            turn["threadId"] = thread_id.into();
            send(&mut stdin, serde_json::json!({"id": 3, "method": "turn/start", "params": turn})).await?;

            let mut reader = Reader::new(plan.tool_names, plan.history_call_ids);
            let mut finished = false;
            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                let message: serde_json::Value = serde_json::from_str(&line)
                    .with_context(|| format!("unexpected output from codex: {line}"))?;
                if let Some(out) = reader.on_message(message)? {
                    finished = out.finish_reason.is_some();
                    yield out;
                    if finished {
                        break;
                    }
                }
            }

            // The app server keeps running after a turn, and a tool call is left
            // unanswered: either way the process is stopped here.
            let _ = child.start_kill();
            let status = child.wait().await?;
            if !finished {
                let stderr = stderr.await.unwrap_or_default();
                Err(anyhow::anyhow!("codex exited with {status} before finishing: {stderr}"))?;
            }
            drop(workdir);
        })
    }
}

/// What one call sends to the app server, built from the conversation.
#[derive(Debug)]
struct Plan {
    /// System messages, joined.
    instructions: String,
    /// Dynamic tool specs for `thread/start`.
    tools: Vec<serde_json::Value>,
    /// The tools' names, to tell their calls from Codex's own.
    tool_names: Vec<String>,
    /// The earlier turns, as Responses API items for `thread/inject_items`.
    history: Vec<serde_json::Value>,
    /// The ids of the calls in the earlier turns.
    history_call_ids: Vec<String>,
    /// `turn/start` input: the latest user message, or none after tool results.
    input: Vec<serde_json::Value>,
}

impl Plan {
    fn new(messages: &[Message], tools: &[ToolDesc]) -> anyhow::Result<Self> {
        let (system, conversation): (Vec<&Message>, Vec<&Message>) =
            messages.iter().partition(|m| m.role == Role::System);
        let instructions = system
            .into_iter()
            .map(text_of)
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n");
        let tool_names = tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect::<Vec<_>>();
        let tools = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "inputSchema": tool.parameters,
                })
            })
            .collect();

        let Some((latest, earlier)) = conversation.split_last() else {
            anyhow::bail!("no message to send");
        };
        let (earlier, input) = match latest.role {
            Role::User => (
                earlier,
                vec![
                    serde_json::json!({"type": "text", "text": text_of(latest)?, "text_elements": []}),
                ],
            ),
            // The results go into the history, and the turn starts without input.
            Role::Tool => (&conversation[..], Vec::new()),
            _ => anyhow::bail!("the last message must be a user message or tool results"),
        };
        let history: Vec<serde_json::Value> = earlier
            .iter()
            .map(|message| response_items(message))
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        let history_call_ids = history
            .iter()
            .filter(|item| item["type"] == "function_call")
            .filter_map(|item| item["call_id"].as_str().map(str::to_owned))
            .collect();

        Ok(Self {
            instructions,
            tools,
            tool_names,
            history,
            history_call_ids,
            input,
        })
    }
}

/// A message as Responses API items: its text as a message, and each tool
/// call or result as an item of its own.
fn response_items(message: &Message) -> anyhow::Result<Vec<serde_json::Value>> {
    let mut items = Vec::new();
    match message.role {
        Role::User => items.push(serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text_of(message)?}],
        })),
        Role::Assistant => {
            let text = texts_of(message)?;
            if !text.is_empty() {
                items.push(serde_json::json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }));
            }
            for part in message.tool_calls.iter().flatten() {
                let Part::Function { id, function } = part else {
                    anyhow::bail!("a tool call must be a function part");
                };
                items.push(serde_json::json!({
                    "type": "function_call",
                    "call_id": id,
                    "name": function.name,
                    "arguments": serde_json::to_string(&function.arguments)?,
                }));
            }
        }
        Role::Tool => {
            let call_id = message
                .id
                .as_deref()
                .context("a tool result must carry its call id")?;
            let mut output = Vec::new();
            for part in &message.contents {
                match part {
                    Part::Text { text } => output.push(text.clone()),
                    Part::Value { value } => output.push(match value.as_str() {
                        Some(s) => s.to_owned(),
                        None => serde_json::to_string(value)?,
                    }),
                    _ => anyhow::bail!("CodexModel supports only text and value tool results"),
                }
            }
            items.push(serde_json::json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": output.join("\n"),
            }));
        }
        ref role => anyhow::bail!("CodexModel does not support {role} messages here"),
    }
    Ok(items)
}

/// Turns the app server's messages during a turn into deltas.
#[derive(Default)]
struct Reader {
    tool_names: Vec<String>,
    /// Calls from the injected history, which come back as raw items too.
    history_call_ids: Vec<String>,
    started: bool,
    /// The agent message the last text came from; a new one is set apart by a blank line.
    message_id: Option<String>,
    /// Calls to the tools in the response so far.
    calls: Vec<PartDelta>,
    /// The latest usage the app server reported for the turn.
    usage: Option<TokenUsage>,
}

impl Reader {
    fn new(tool_names: Vec<String>, history_call_ids: Vec<String>) -> Self {
        Self {
            tool_names,
            history_call_ids,
            ..Default::default()
        }
    }

    /// Handles one message. A delta with a finish reason ends the reply.
    fn on_message(
        &mut self,
        message: serde_json::Value,
    ) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let Some(method) = message["method"].as_str() else {
            // A response to one of our requests: only its error matters.
            if let Some(error) = message.get("error") {
                anyhow::bail!("codex rejected a request: {error}");
            }
            return Ok(None);
        };
        let params = &message["params"];
        let is_request = message.get("id").is_some();

        let mut out = MessageDeltaOutput::new();
        match method {
            // Codex running one of the calls, which are taken from the raw items instead.
            "item/tool/call" => return Ok(None),
            _ if is_request => anyhow::bail!("codex asked for {method}, which is not supported"),
            "rawResponseItem/completed" => {
                let item = &params["item"];
                let name = item["name"].as_str().unwrap_or_default();
                let call_id = item["call_id"].as_str().unwrap_or_default();
                if item["type"] == "function_call"
                    && item.get("namespace").is_none()
                    && self.tool_names.iter().any(|n| n == name)
                    && !self.history_call_ids.iter().any(|id| id == call_id)
                {
                    let arguments = item["arguments"].as_str().unwrap_or("{}");
                    let arguments = serde_json::from_str(arguments)
                        .unwrap_or_else(|_| serde_json::Value::String(arguments.to_owned()));
                    self.calls.push(PartDelta::Function {
                        id: Some(call_id.to_owned()),
                        function: PartDeltaFunction::WithParsedArgs {
                            name: name.to_owned(),
                            arguments: arguments.into(),
                        },
                    });
                }
                return Ok(None);
            }
            // The response is complete: if it called the tools, the reply ends with the calls.
            "rawResponse/completed" => {
                self.usage = parse_usage(&params["usage"]);
                if self.calls.is_empty() {
                    return Ok(None);
                }
                out.delta = out.delta.with_tool_calls(std::mem::take(&mut self.calls));
                out.finish_reason = Some(FinishReason::ToolCall {});
                out.usage = self.usage.take();
            }
            "item/agentMessage/delta" => {
                let mut text = params["delta"].as_str().unwrap_or_default().to_owned();
                let item = params["itemId"].as_str().map(str::to_owned);
                if self.message_id.is_some() && self.message_id != item {
                    text.insert_str(0, "\n\n");
                }
                self.message_id = item;
                out.delta = out.delta.with_contents([PartDelta::Text { text }]);
            }
            "item/reasoning/summaryTextDelta" | "item/reasoning/textDelta" => {
                out.delta.thinking = Some(params["delta"].as_str().unwrap_or_default().to_owned());
            }
            "item/started" => {
                let kind = params["item"]["type"].as_str().unwrap_or_default();
                if OWN_TOOL_ITEMS.contains(&kind) {
                    anyhow::bail!("codex used a tool of its own ({kind}): {}", params["item"]);
                }
                return Ok(None);
            }
            "thread/tokenUsage/updated" => {
                self.usage = parse_usage(&params["tokenUsage"]["last"]);
                return Ok(None);
            }
            "turn/completed" => {
                let turn = &params["turn"];
                match turn["status"].as_str() {
                    Some("completed") => {
                        out.finish_reason = Some(FinishReason::Stop {});
                        out.usage = self.usage.take();
                    }
                    status => {
                        let error = turn["error"]["message"].as_str().unwrap_or_default();
                        anyhow::bail!("codex turn {}: {error}", status.unwrap_or("ended"));
                    }
                }
            }
            // `error` reports a failed request that may still be retried; a
            // failure for good ends the turn as failed.
            _ => return Ok(None),
        }
        if !self.started {
            out.delta.role = Some(Role::Assistant);
            self.started = true;
        }
        Ok(Some(out))
    }
}

/// Writes one JSON-RPC message.
async fn send(stdin: &mut ChildStdin, message: serde_json::Value) -> anyhow::Result<()> {
    let mut line = serde_json::to_string(&message)?;
    line.push('\n');
    stdin.write_all(line.as_bytes()).await?;
    stdin.flush().await?;
    Ok(())
}

/// Reads until the response to request `id`, skipping notifications, and
/// returns its result.
async fn response(
    lines: &mut tokio::io::Lines<BufReader<tokio::process::ChildStdout>>,
    id: u64,
) -> anyhow::Result<serde_json::Value> {
    while let Some(line) = lines.next_line().await? {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        if message["id"] != id || message.get("method").is_some() {
            continue;
        }
        if let Some(error) = message.get("error") {
            anyhow::bail!("codex rejected request {id}: {error}");
        }
        return Ok(message["result"].clone());
    }
    anyhow::bail!("codex exited before answering request {id}")
}

/// The text of a message that cannot carry tool calls.
fn text_of(message: &Message) -> anyhow::Result<String> {
    if message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) {
        anyhow::bail!("only assistant messages can carry tool calls");
    }
    texts_of(message)
}

/// The text parts of a message joined; any other part is an error.
fn texts_of(message: &Message) -> anyhow::Result<String> {
    let texts = message
        .contents
        .iter()
        .map(|part| {
            part.as_text()
                .ok_or_else(|| anyhow::anyhow!("CodexModel supports only text parts"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(texts.join("\n"))
}

fn parse_usage(u: &serde_json::Value) -> Option<TokenUsage> {
    Some(TokenUsage {
        input_tokens: u["inputTokens"].as_u64()?,
        output_tokens: u["outputTokens"].as_u64()?,
        cache_creation_input_tokens: u["cacheWriteInputTokens"].as_u64(),
        cache_read_input_tokens: u["cachedInputTokens"].as_u64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_call() -> [Message; 3] {
        [
            Message::new(Role::User).with_contents([Part::text("weather?")]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "c1",
                "get_weather",
                serde_json::json!({"city": "Seoul"}),
            )]),
            Message::new(Role::Tool)
                .with_id("c1")
                .with_contents([Part::text("sunny")]),
        ]
    }

    #[test]
    fn tool_results_go_into_the_history() {
        let mut messages =
            vec![Message::new(Role::System).with_contents([Part::text("Be brief.")])];
        messages.extend(weather_call());
        let plan = Plan::new(&messages, &[]).unwrap();
        assert_eq!(plan.instructions, "Be brief.");
        assert!(plan.input.is_empty());
        assert_eq!(plan.history_call_ids, ["c1"]);
        assert_eq!(
            serde_json::Value::Array(plan.history),
            serde_json::json!([
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
                {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": "{\"city\":\"Seoul\"}"},
                {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            ])
        );
    }

    #[test]
    fn latest_user_message_is_the_input() {
        let mut messages = weather_call().to_vec();
        messages.push(Message::new(Role::User).with_contents([Part::text("and Tokyo?")]));
        let plan = Plan::new(&messages, &[]).unwrap();
        assert_eq!(plan.history.len(), 3);
        assert_eq!(
            serde_json::Value::Array(plan.input),
            serde_json::json!([{"type": "text", "text": "and Tokyo?", "text_elements": []}])
        );
    }

    /// Runs `messages` through a reader; returns the deltas until the first finish.
    fn read(messages: &[serde_json::Value]) -> anyhow::Result<Vec<MessageDeltaOutput>> {
        let mut reader = Reader::new(vec!["get_weather".to_owned()], vec!["call_0".to_owned()]);
        let mut outs = Vec::new();
        for message in messages {
            if let Some(out) = reader.on_message(message.clone())? {
                let finished = out.finish_reason.is_some();
                outs.push(out);
                if finished {
                    break;
                }
            }
        }
        Ok(outs)
    }

    /// The name and arguments of each parsed call.
    fn calls(out: &MessageDeltaOutput) -> Vec<(Option<String>, String, serde_json::Value)> {
        out.delta
            .tool_calls
            .iter()
            .map(|call| {
                let PartDelta::Function {
                    id,
                    function: PartDeltaFunction::WithParsedArgs { name, arguments },
                } = call
                else {
                    panic!("not a parsed call: {call:?}");
                };
                (
                    id.clone(),
                    name.clone(),
                    serde_json::to_value(arguments).unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn all_calls_end_the_reply_with_usage() {
        let function_call = |id: &str, name: &str, city: &str| {
            serde_json::json!({"method": "rawResponseItem/completed", "params": {"item": {
                "type": "function_call", "call_id": id, "name": name,
                "arguments": format!("{{\"city\":\"{city}\"}}"),
            }}})
        };
        let outs = read(&[
            serde_json::json!({"id": 3, "result": {"turn": {"id": "t"}}}),
            // A call from the injected history.
            function_call("call_0", "get_weather", "Busan"),
            serde_json::json!({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": "Let me check."}}),
            function_call("call_1", "get_weather", "Seoul"),
            // Codex runs the first call while the response is still streaming.
            serde_json::json!({"id": 0, "method": "item/tool/call", "params": {"callId": "call_1", "tool": "get_weather", "arguments": {"city": "Seoul"}}}),
            function_call("call_2", "get_weather", "Tokyo"),
            function_call("call_3", "not_ours", "Paris"),
            serde_json::json!({"method": "rawResponse/completed", "params": {"responseId": "r", "usage": {"totalTokens": 15, "inputTokens": 10, "cachedInputTokens": 4, "cacheWriteInputTokens": 0, "outputTokens": 5, "reasoningOutputTokens": 0}}}),
        ])
        .unwrap();
        assert_eq!(outs[0].delta.role, Some(Role::Assistant));
        let last = outs.last().unwrap();
        assert!(matches!(
            last.finish_reason,
            Some(FinishReason::ToolCall {})
        ));
        assert_eq!(
            calls(last),
            [
                (
                    Some("call_1".to_owned()),
                    "get_weather".to_owned(),
                    serde_json::json!({"city": "Seoul"})
                ),
                (
                    Some("call_2".to_owned()),
                    "get_weather".to_owned(),
                    serde_json::json!({"city": "Tokyo"})
                ),
            ]
        );
        let usage = last.usage.as_ref().unwrap();
        assert_eq!(
            (usage.input_tokens, usage.cache_read_input_tokens),
            (10, Some(4))
        );
    }

    #[test]
    fn turn_end_closes_or_fails_the_reply() {
        let outs = read(&[
            serde_json::json!({"method": "item/agentMessage/delta", "params": {"itemId": "m1", "delta": "Hi."}}),
            serde_json::json!({"method": "item/agentMessage/delta", "params": {"itemId": "m2", "delta": "Bye."}}),
            serde_json::json!({"method": "turn/completed", "params": {"turn": {"status": "completed", "error": null}}}),
        ])
        .unwrap();
        assert!(
            matches!(&outs[1].delta.contents[..], [PartDelta::Text { text }] if text == "\n\nBye.")
        );
        assert!(matches!(outs[2].finish_reason, Some(FinishReason::Stop {})));

        let err = read(&[serde_json::json!({"method": "turn/completed", "params": {"turn": {"status": "failed", "error": {"message": "401 Unauthorized"}}}})])
            .unwrap_err();
        assert!(err.to_string().contains("401"), "{err}");
    }

    #[test]
    fn catalog_drops_code_mode_and_multi_agent() {
        let catalog = direct_catalog(serde_json::json!({"models": [{
            "slug": "gpt",
            "tool_mode": "code_mode_only",
            "multi_agent_version": 2,
            "model_messages": {"multi_agent": {}, "approvals": {}},
            "experimental_supported_tools": ["clock"],
            "use_responses_lite": true,
        }]}));
        assert_eq!(
            catalog,
            serde_json::json!({"models": [{
                "slug": "gpt",
                "model_messages": {"approvals": {}},
                "experimental_supported_tools": [],
                "use_responses_lite": false,
            }]})
        );
    }

    #[test]
    fn own_tool_use_fails() {
        let err = read(&[serde_json::json!({"method": "item/started", "params": {"item": {"type": "commandExecution", "id": "x"}}})])
            .unwrap_err();
        assert!(err.to_string().contains("commandExecution"), "{err}");
    }
}
