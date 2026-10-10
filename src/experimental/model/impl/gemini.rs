//! A language model backed by the Gemini CLI in headless mode.
//!
//! Each call starts one `gemini` process in a project directory shared by all
//! calls ([`PROJECT_DIR`] under the system's temporary directory), whose
//! `.gemini/settings.json` is the same for every call: it keeps the CLI's own
//! tools and context files out, declares the tools through a discovery
//! command that prints the file named by `AILOY_GEMINI_TOOLS`, and adds a
//! `BeforeTool` hook that stops the agent before any tool runs. Keeping it the
//! same keeps the CLI's record of trusted project hooks to one entry.
//!
//! The call's own files go in a temporary directory:
//! - `system.md`, which replaces the CLI's system prompt (`GEMINI_SYSTEM_MD`);
//! - `tools.json`, the function declarations (`AILOY_GEMINI_TOOLS`);
//! - `session.jsonl`, the earlier turns as a session record (`--session-file`),
//!   with tool calls and results as function calls and responses.
//!
//! The latest user message goes to stdin, and the reply is read as one JSON
//! event per line from stdout (`--output-format stream-json`). The model
//! calls the tools as declared functions: each call is reported as a
//! `tool_use` event, the hook then stops the run, and the final `result`
//! event closes the reply with the calls and the run's usage.

use std::{path::Path, process::Stdio};

use anyhow::Context as _;
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    process::Command,
};

use crate::{
    experimental::model::InferLangModel,
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, Role, TokenUsage,
    },
    tool::ToolDesc,
};

/// The project directory every call runs in, under the system's temporary directory.
const PROJECT_DIR: &str = "ailoy-gemini";

/// The CLI's own tools, kept from the model. Names unknown to the installed
/// version are ignored. (`tools.core`, the CLI's allowlist, cannot be used
/// instead: it denies discovered tools too.)
const BUILTIN_TOOLS: &[&str] = &[
    "activate_skill",
    "ask_user",
    "complete_task",
    "enter_plan_mode",
    "exit_plan_mode",
    "get_internal_docs",
    "glob",
    "google_web_search",
    "grep_search",
    "invoke_agent",
    "list_background_processes",
    "list_directory",
    "list_mcp_resources",
    "read_background_output",
    "read_file",
    "read_many_files",
    "read_mcp_resource",
    "replace",
    "run_shell_command",
    "save_memory",
    "take_snapshot",
    "tracker_add_dependency",
    "tracker_create_task",
    "tracker_get_task",
    "tracker_list_tasks",
    "tracker_update_task",
    "tracker_visualize",
    "update_topic",
    "web_fetch",
    "write_file",
    "write_todos",
];

/// The prefix the CLI gives the names of discovered tools.
const TOOL_PREFIX: &str = "discovered_tool_";

/// The prompt when the conversation ends with tool results: the CLI needs one,
/// and the results themselves are in the session.
const CONTINUE_PROMPT: &str = "Continue from the tool results above.";

/// Runs Gemini through the `gemini` CLI instead of calling the API directly.
///
/// The CLI's login (Google account or API key) is used as is. System messages
/// replace the CLI's own system prompt; without any, the prompt is empty.
/// Tools are declared to the model, so it calls them as it would any tool.
///
/// Limits of going through the CLI:
/// - Text only: image parts are rejected. Thinking is not reported.
/// - A tool's description gets the CLI's note on discovered tools appended.
/// - After tool results, the turn goes on from a short user prompt
///   ([`CONTINUE_PROMPT`]), since the CLI needs one; before it, the CLI puts
///   a model turn noting that the previous response was interrupted.
/// - The CLI drops earlier user messages that start with `/` or `?` from the
///   session, and takes a latest message that starts with `/` as a command.
/// - Each call leaves a session record under `~/.gemini/tmp`.
/// - The discovery command and the hook run through `sh`, so not on Windows.
#[derive(Clone, Debug)]
pub struct GeminiModel {
    program: String,
    model: Option<String>,
}

impl Default for GeminiModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GeminiModel {
    pub fn new() -> Self {
        Self {
            program: "gemini".to_owned(),
            model: None,
        }
    }

    /// Model name for `--model` (e.g. `"gemini-2.5-flash"`); unset keeps the CLI default.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Name or path of the `gemini` executable.
    pub fn with_program(mut self, program: impl Into<String>) -> Self {
        self.program = program.into();
        self
    }

    /// Writes the files for one call into `workdir` and builds the command,
    /// without starting it.
    fn command(&self, plan: &Plan, workdir: &Path) -> anyhow::Result<Command> {
        let project = project_dir()?;
        let system_md = workdir.join("system.md");
        std::fs::write(&system_md, &plan.system)?;
        let tools_json = workdir.join("tools.json");
        std::fs::write(&tools_json, serde_json::to_string(&plan.tools)?)?;

        let mut command = Command::new(&self.program);
        command.args([
            "--output-format",
            "stream-json",
            // Without it, tools that need approval are kept from the model in
            // headless mode; the hook stops them before they run anyway.
            "--approval-mode",
            "yolo",
            // Names no configured server, so none is started.
            "--allowed-mcp-server-names",
            "ailoy-none",
        ]);
        if let Some(model) = &self.model {
            command.args(["--model", model.as_str()]);
        }
        if !plan.session.is_empty() {
            let session = workdir.join("session.jsonl");
            std::fs::write(&session, &plan.session)?;
            command.arg("--session-file").arg(session);
        }
        command
            // Always set, even empty: otherwise the CLI's own coding-agent prompt is used.
            .env("GEMINI_SYSTEM_MD", &system_md)
            .env("AILOY_GEMINI_TOOLS", &tools_json)
            // Without trust, the project settings are not loaded.
            .env("GEMINI_CLI_TRUST_WORKSPACE", "true")
            .current_dir(project)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Dropping a stream mid-reply ends the process with it.
            .kill_on_drop(true);
        Ok(command)
    }
}

/// The shared project directory, with its settings written if they differ.
fn project_dir() -> anyhow::Result<std::path::PathBuf> {
    let project = std::env::temp_dir().join(PROJECT_DIR);
    let settings_dir = project.join(".gemini");
    std::fs::create_dir_all(&settings_dir)?;
    let path = settings_dir.join("settings.json");
    let settings = serde_json::to_string_pretty(&settings())?;
    if std::fs::read_to_string(&path).ok().as_deref() != Some(settings.as_str()) {
        // Written aside and renamed, so a concurrent call never reads it half-written.
        let staged = tempfile::NamedTempFile::new_in(&settings_dir)?;
        std::fs::write(staged.path(), &settings)?;
        staged.persist(&path)?;
    }
    Ok(project)
}

/// The project settings, the same for every call.
fn settings() -> serde_json::Value {
    serde_json::json!({
        "tools": {
            "exclude": BUILTIN_TOOLS,
            // Quoted so the CLI passes the variable to `sh` unexpanded.
            "discoveryCommand": "sh -c 'cat \"$AILOY_GEMINI_TOOLS\"'",
            // Never run: the hook stops the agent first.
            "callCommand": "false",
        },
        "context": {
            // No such file exists, so no GEMINI.md is added to the system prompt.
            "fileName": "AILOY_NO_CONTEXT.md",
            // Keeps the directory listing out of the context the CLI adds.
            "includeDirectoryTree": false,
        },
        "hooks": {
            "BeforeTool": [{
                "matcher": ".*",
                "hooks": [{
                    "type": "command",
                    "command": "echo '{\"continue\":false,\"stopReason\":\"the call goes back to the caller\"}'",
                }],
            }],
        },
    })
}

impl InferLangModel for GeminiModel {
    fn infer(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
    ) -> BoxFuture<'static, anyhow::Result<MessageOutput>> {
        let mut deltas = self.infer_stream(messages, tools);
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
    ) -> BoxStream<'static, anyhow::Result<MessageDeltaOutput>> {
        let plan = match Plan::new(messages, tools) {
            Ok(plan) => plan,
            Err(e) => return Box::pin(stream::once(async move { Err(e) })),
        };
        let this = self.clone();

        Box::pin(async_stream::try_stream! {
            let workdir = tempfile::tempdir()?;
            let mut child = this
                .command(&plan, workdir.path())?
                .spawn()
                .context("failed to start the gemini CLI")?;
            let mut stdin = child.stdin.take().expect("stdin is piped");
            stdin.write_all(plan.prompt.as_bytes()).await?;
            // Closing stdin ends the prompt.
            drop(stdin);
            let stdout = child.stdout.take().expect("stdout is piped");
            let mut stderr = child.stderr.take().expect("stderr is piped");
            // Drained alongside stdout so a full stderr pipe cannot stall the process.
            let stderr = tokio::spawn(async move {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf).await;
                buf
            });

            let mut lines = BufReader::new(stdout).lines();
            let mut reader = Reader::default();
            let mut finished = false;
            while let Some(line) = lines.next_line().await? {
                // The CLI may print plain log lines too.
                if !line.trim_start().starts_with('{') {
                    continue;
                }
                let event: serde_json::Value = serde_json::from_str(&line)
                    .with_context(|| format!("unexpected output from gemini: {line}"))?;
                if let Some(out) = reader.on_event(event)? {
                    finished = out.finish_reason.is_some();
                    yield out;
                    if finished {
                        break;
                    }
                }
            }

            if finished {
                // The run has ended; a no-op unless it lingers.
                let _ = child.start_kill();
                let _ = child.wait().await;
            } else {
                let status = child.wait().await?;
                let stderr = stderr.await.unwrap_or_default();
                Err(anyhow::anyhow!("gemini exited with {status} before finishing: {stderr}"))?;
            }
            drop(workdir);
        })
    }
}

/// What one call hands to the CLI, built from the conversation.
#[derive(Debug)]
struct Plan {
    /// System messages, joined.
    system: String,
    /// Function declarations for the discovery command to print.
    tools: Vec<serde_json::Value>,
    /// The earlier turns as a session record, or empty when there are none.
    session: String,
    /// The prompt for stdin.
    prompt: String,
}

impl Plan {
    fn new(messages: &[Message], tools: &[ToolDesc]) -> anyhow::Result<Self> {
        let (system, conversation): (Vec<&Message>, Vec<&Message>) =
            messages.iter().partition(|m| m.role == Role::System);
        let system = system
            .into_iter()
            .map(text_of)
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n");
        let tools = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description.clone().unwrap_or_default(),
                    "parametersJsonSchema": tool.parameters,
                })
            })
            .collect();

        let Some((latest, earlier)) = conversation.split_last() else {
            anyhow::bail!("no message to send");
        };
        let (earlier, prompt) = match latest.role {
            Role::User => (earlier, text_of(latest)?),
            // The results go into the session, and the turn goes on from a short prompt.
            Role::Tool => (&conversation[..], CONTINUE_PROMPT.to_owned()),
            _ => anyhow::bail!("the last message must be a user message or tool results"),
        };
        if prompt.trim().is_empty() {
            anyhow::bail!("the user message is empty");
        }

        Ok(Self {
            system,
            tools,
            session: session_record(earlier)?,
            prompt,
        })
    }
}

/// The conversation as a session record: a metadata line, then one line per
/// message. Tool results are attached to the calls they answer.
fn session_record(conversation: &[&Message]) -> anyhow::Result<String> {
    if conversation.is_empty() {
        return Ok(String::new());
    }
    let mut records: Vec<serde_json::Value> = Vec::new();
    for message in conversation {
        match message.role {
            Role::User => records.push(serde_json::json!({
                "id": format!("m{}", records.len()),
                "type": "user",
                "content": [{"text": text_of(message)?}],
            })),
            Role::Assistant => {
                let text = texts_of(message)?;
                let mut calls = Vec::new();
                for part in message.tool_calls.iter().flatten() {
                    let Part::Function { id, function } = part else {
                        anyhow::bail!("a tool call must be a function part");
                    };
                    calls.push(serde_json::json!({
                        "id": id,
                        "name": format!("{TOOL_PREFIX}{}", function.name),
                        "args": function.arguments,
                    }));
                }
                // Without text the content is empty but present, as the CLI requires.
                let content = if text.is_empty() {
                    serde_json::json!([])
                } else {
                    serde_json::json!(text)
                };
                records.push(serde_json::json!({
                    "id": format!("m{}", records.len()),
                    "type": "gemini",
                    "content": content,
                    "toolCalls": calls,
                }));
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
                        _ => anyhow::bail!("GeminiModel supports only text and value tool results"),
                    }
                }
                let call = records
                    .iter_mut()
                    .rev()
                    .filter(|r| r["type"] == "gemini")
                    .flat_map(|r| r["toolCalls"].as_array_mut().into_iter().flatten())
                    .find(|c| c["id"] == call_id)
                    .with_context(|| format!("no call {call_id} for its result"))?;
                call["result"] = output.join("\n").into();
            }
            ref role => anyhow::bail!("GeminiModel does not support {role} messages here"),
        }
    }

    let mut lines = vec![serde_json::json!({"sessionId": "ailoy", "projectHash": "ailoy"}).to_string()];
    lines.extend(records.iter().map(|r| r.to_string()));
    Ok(lines.join("\n") + "\n")
}

/// Turns the CLI's events into deltas.
#[derive(Default)]
struct Reader {
    started: bool,
    /// Calls to the tools in the reply so far.
    calls: Vec<PartDelta>,
}

impl Reader {
    /// Handles one event. A delta with a finish reason ends the reply.
    fn on_event(&mut self, event: serde_json::Value) -> anyhow::Result<Option<MessageDeltaOutput>> {
        let mut out = MessageDeltaOutput::new();
        match event["type"].as_str() {
            // The prompt is echoed back as a `user` message first.
            Some("message") if event["role"] == "assistant" => {
                let text = event["content"].as_str().unwrap_or_default().to_owned();
                if text.is_empty() {
                    return Ok(None);
                }
                out.delta = out.delta.with_contents([PartDelta::Text { text }]);
            }
            Some("tool_use") => {
                let name = event["tool_name"].as_str().unwrap_or_default();
                let Some(name) = name.strip_prefix(TOOL_PREFIX) else {
                    anyhow::bail!("gemini used a tool of its own ({name}): {event}");
                };
                self.calls.push(PartDelta::Function {
                    id: event["tool_id"].as_str().map(str::to_owned),
                    function: PartDeltaFunction::WithParsedArgs {
                        name: name.to_owned(),
                        arguments: event["parameters"].clone().into(),
                    },
                });
                return Ok(None);
            }
            // The run is over: the reply ends with the calls, if any, and the usage.
            Some("result") => {
                if event["status"] != "success" {
                    let message = event["error"]["message"].as_str().unwrap_or_default();
                    anyhow::bail!("gemini failed: {message}");
                }
                if self.calls.is_empty() {
                    out.finish_reason = Some(FinishReason::Stop {});
                } else {
                    out.delta = out.delta.with_tool_calls(std::mem::take(&mut self.calls));
                    out.finish_reason = Some(FinishReason::ToolCall {});
                }
                out.usage = parse_usage(&event["stats"]);
            }
            // `tool_result` is the hook stopping a call; `error` events are
            // warnings, or precede a failed `result`.
            _ => return Ok(None),
        }
        if !self.started {
            out.delta.role = Some(Role::Assistant);
            self.started = true;
        }
        Ok(Some(out))
    }
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
                .ok_or_else(|| anyhow::anyhow!("GeminiModel supports only text parts"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(texts.join("\n"))
}

/// Usage summed over every model the run used.
fn parse_usage(stats: &serde_json::Value) -> Option<TokenUsage> {
    Some(TokenUsage {
        input_tokens: stats["input_tokens"].as_u64()?,
        output_tokens: stats["output_tokens"].as_u64()?,
        cache_creation_input_tokens: None,
        cache_read_input_tokens: stats["cached"].as_u64(),
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
    fn results_attach_to_their_calls_in_the_session() {
        let plan = Plan::new(&weather_call(), &[]).unwrap();
        assert_eq!(plan.prompt, CONTINUE_PROMPT);
        let lines: Vec<serde_json::Value> = plan
            .session
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            lines,
            [
                serde_json::json!({"sessionId": "ailoy", "projectHash": "ailoy"}),
                serde_json::json!({"id": "m0", "type": "user", "content": [{"text": "weather?"}]}),
                serde_json::json!({"id": "m1", "type": "gemini", "content": [], "toolCalls": [
                    {"id": "c1", "name": "discovered_tool_get_weather", "args": {"city": "Seoul"}, "result": "sunny"},
                ]}),
            ]
        );
    }

    #[test]
    fn lone_user_message_needs_no_session() {
        let messages = [
            Message::new(Role::System).with_contents([Part::text("Be brief.")]),
            Message::new(Role::User).with_contents([Part::text("hi")]),
        ];
        let plan = Plan::new(&messages, &[]).unwrap();
        assert_eq!((plan.system.as_str(), plan.prompt.as_str(), plan.session.as_str()), ("Be brief.", "hi", ""));
    }

    /// Runs `events` through a reader; returns the deltas until the first finish.
    fn read(events: &[serde_json::Value]) -> anyhow::Result<Vec<MessageDeltaOutput>> {
        let mut reader = Reader::default();
        let mut outs = Vec::new();
        for event in events {
            if let Some(out) = reader.on_event(event.clone())? {
                let finished = out.finish_reason.is_some();
                outs.push(out);
                if finished {
                    break;
                }
            }
        }
        Ok(outs)
    }

    #[test]
    fn calls_end_the_reply_with_usage() {
        let tool_use = |id: &str, city: &str| {
            serde_json::json!({"type": "tool_use", "tool_name": "discovered_tool_get_weather", "tool_id": id, "parameters": {"city": city}})
        };
        let outs = read(&[
            serde_json::json!({"type": "init", "session_id": "s", "model": "gemini"}),
            serde_json::json!({"type": "message", "role": "user", "content": "weather?"}),
            serde_json::json!({"type": "message", "role": "assistant", "content": "Let me check.", "delta": true}),
            tool_use("t1", "Seoul"),
            tool_use("t2", "Tokyo"),
            serde_json::json!({"type": "tool_result", "tool_id": "t1", "status": "error", "output": ""}),
            serde_json::json!({"type": "result", "status": "success", "stats": {"total_tokens": 15, "input_tokens": 10, "output_tokens": 5, "cached": 4, "input": 6, "duration_ms": 1, "tool_calls": 1}}),
        ])
        .unwrap();
        assert_eq!(outs[0].delta.role, Some(Role::Assistant));
        let last = outs.last().unwrap();
        assert!(matches!(last.finish_reason, Some(FinishReason::ToolCall {})));
        let names: Vec<_> = last
            .delta
            .tool_calls
            .iter()
            .map(|c| match c {
                PartDelta::Function { id, function: PartDeltaFunction::WithParsedArgs { name, .. } } => {
                    (id.clone().unwrap(), name.clone())
                }
                other => panic!("not a parsed call: {other:?}"),
            })
            .collect();
        assert_eq!(names, [("t1".to_owned(), "get_weather".to_owned()), ("t2".to_owned(), "get_weather".to_owned())]);
        let usage = last.usage.as_ref().unwrap();
        assert_eq!((usage.input_tokens, usage.cache_read_input_tokens), (10, Some(4)));
    }

    #[test]
    fn plain_reply_stops_and_failure_errors() {
        let outs = read(&[
            serde_json::json!({"type": "message", "role": "assistant", "content": "Hi.", "delta": true}),
            serde_json::json!({"type": "result", "status": "success", "stats": {"input_tokens": 3, "output_tokens": 1, "cached": 0}}),
        ])
        .unwrap();
        assert!(matches!(outs.last().unwrap().finish_reason, Some(FinishReason::Stop {})));

        let err = read(&[serde_json::json!({"type": "result", "status": "error", "error": {"type": "Error", "message": "quota"}, "stats": {}})])
            .unwrap_err();
        assert!(err.to_string().contains("quota"), "{err}");

        let err = read(&[serde_json::json!({"type": "tool_use", "tool_name": "run_shell_command", "tool_id": "x", "parameters": {}})])
            .unwrap_err();
        assert!(err.to_string().contains("run_shell_command"), "{err}");
    }
}
