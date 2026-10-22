//! A language model backed by the Claude Code CLI in headless mode (`claude -p`).
//!
//! Each call starts one `claude` process, writes the prompt to its stdin as one
//! user message (`--input-format stream-json`), so images can go with the text,
//! and reads the reply as one JSON event per line from its stdout
//! (`--output-format stream-json`).
//!
//! Claude Code's own tools and MCP servers are switched off. The tools a call is given are
//! declared to the CLI by an MCP server that runs in this process, as the Agent SDK's
//! in-process servers do: the CLI is told its name in an `initialize` control request, and
//! sends its MCP messages over the same stdin and stdout, as `mcp_message` control requests.
//! The model calls the tools natively, and the reply streams each call as a `tool_use` block.
//! The process is stopped once the reply ends with them, before the CLI would run them, so
//! the server only ever lists the tools.

use std::{collections::HashMap, io::Write as _, process::Stdio};

use anyhow::Context as _;
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    process::{Child, ChildStdin, Command},
};

use super::claude_api::image_block;
use crate::{
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, Role, TokenUsage,
    },
    model::{InferLangModel, LangModelOptions, ThinkingEffort},
    tool::ToolDesc,
};

/// The tool the CLI adds for `--json-schema`; the reply is its arguments.
const STRUCTURED_OUTPUT: &str = "StructuredOutput";
/// The in-process MCP server that declares the tools.
const SERVER: &str = "ailoy";
/// What the CLI puts before the name of each of [`SERVER`]'s tools.
const TOOL_PREFIX: &str = "mcp__ailoy__";

/// Runs Claude through the `claude` CLI instead of calling the API directly.
///
/// The CLI's login (subscription or API key) is used as is. System messages
/// replace Claude Code's own system prompt; without any, the prompt is empty.
///
/// Limits of going through the CLI:
/// - Tools reach the model as an MCP server's, named `mcp__ailoy__<name>`. The API takes
///   names of up to 64 characters, so a tool's own name can have 52.
/// - The CLI takes a single prompt, so earlier turns, tool calls and tool
///   results are written into it as a transcript, with images in user
///   messages and tool results placed where they were.
/// - System and assistant messages are text only.
/// - Consecutive prompts share a prefix, but the CLI does not read the earlier
///   turns back from the prompt cache yet: its own cache breakpoint comes
///   after the transcript, so each call writes the history to the cache again.
/// - The CLI adds context of its own to the request: the signed-in user's
///   email, the working directory and the date reach the model.
#[derive(Clone, Debug)]
pub struct ClaudeCliModel {
    program: String,
    model: Option<String>,
}

impl Default for ClaudeCliModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeCliModel {
    pub fn new() -> Self {
        Self {
            program: "claude".to_owned(),
            model: None,
        }
    }

    /// Model id for `--model` (e.g. `"claude-sonnet-5-5"`); unset keeps the CLI default.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Name or path of the `claude` executable.
    pub fn with_program(mut self, program: impl Into<String>) -> Self {
        self.program = program.into();
        self
    }

    /// Builds the command for one call, without starting it.
    fn invocation(
        &self,
        messages: &[Message],
        tools: &[ToolDesc],
        options: &LangModelOptions,
    ) -> anyhow::Result<Invocation> {
        let (system, conversation): (Vec<&Message>, Vec<&Message>) =
            messages.iter().partition(|m| m.role == Role::System);
        let system = system
            .into_iter()
            .map(text_of)
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n");
        let prompt = render_prompt(&conversation)?;
        // In a file rather than in argv, where a long prompt would hit the OS's limit on
        // argument size. An empty prompt is passed as is, as the CLI may read an empty
        // file as no prompt at all.
        let system_file = if system.is_empty() {
            None
        } else {
            // Created readable by the owner only.
            let mut file = tempfile::Builder::new()
                .prefix("ailoy-claude-system-")
                .tempfile()
                .context("failed to create the system prompt file")?;
            file.write_all(system.as_bytes())?;
            Some(file.into_temp_path())
        };

        let mut command = Command::new(&self.program);
        command.args([
            "-p",
            "--input-format",
            "stream-json",
            // Claude Code's own tools; the given ones come from the in-process server.
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            // The user's and project's settings (hooks, output style, permissions, …) stay out
            // of the model, as does the auto-memory set below.
            "--setting-sources",
            "",
            // `stream-json` under `-p` requires `--verbose`.
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]);
        if let Some(model) = &self.model {
            command.args(["--model", model.as_str()]);
        }
        // Unset keeps the CLI's own default effort.
        if let Some(effort) = options.thinking_effort {
            let effort = match effort {
                ThinkingEffort::Low => "low",
                ThinkingEffort::Medium => "medium",
                ThinkingEffort::High => "high",
            };
            command.args(["--effort", effort]);
        }
        if let Some(schema) = &options.output_schema {
            // The CLI answers through its own `StructuredOutput` tool, read in place of text.
            command.args(["--json-schema", serde_json::to_string(schema)?.as_str()]);
        }
        // Always set, even empty: otherwise Claude Code's own coding-agent prompt is used.
        match &system_file {
            Some(path) => command.arg("--system-prompt-file").arg(path.as_os_str()),
            None => command.args(["--system-prompt", ""]),
        };
        command
            // Memories the CLI saved for the working directory would reach the model.
            .env("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Dropping a stream mid-reply ends the process with it.
            .kill_on_drop(true);

        Ok(Invocation {
            command,
            prompt,
            tools: tools.to_vec(),
            _system_file: system_file,
        })
    }
}

impl InferLangModel for ClaudeCliModel {
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
        let mut invocation = match self.invocation(messages, tools, options) {
            Ok(invocation) => invocation,
            Err(e) => return Box::pin(stream::once(async move { Err(e) })),
        };
        // With a schema the reply is the `StructuredOutput` call alone; plain text the model
        // writes around it is dropped, though tool calls are still taken.
        let structured = options.output_schema.is_some();

        Box::pin(async_stream::try_stream! {
            let tools = invocation.tools.clone();
            let (mut child, mut stdin) = invocation.spawn().await?;
            let stdout = child.stdout.take().expect("stdout is piped");
            let mut stderr = child.stderr.take().expect("stderr is piped");
            // Drained alongside stdout so a full stderr pipe cannot stall the process.
            let stderr = tokio::spawn(async move {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf).await;
                buf
            });

            let mut lines = BufReader::new(stdout).lines();
            // Usage from `message_start`, completed by `message_delta` once the message ends.
            let mut start_usage = None;
            // The usage of the latest message to end, which is the reply's.
            let mut reply_usage = None;
            // Why the latest message ended, from its `message_delta`.
            let mut stop_reason = None;
            // The `tool_use` block being streamed.
            let mut tool_use: Option<ToolUse> = None;
            // The calls of the latest message to the given tools.
            let mut calls = Vec::new();
            // The arguments of the latest `StructuredOutput` call, which are the reply. Held
            // to the end, since a call the CLI rejects is followed by another.
            let mut structured_json = String::new();
            let mut started = false;
            let mut finished = false;
            while let Some(line) = lines.next_line().await? {
                if line.trim().is_empty() {
                    continue;
                }
                let event: serde_json::Value = serde_json::from_str(&line)
                    .with_context(|| format!("unexpected output from claude: {line}"))?;

                let mut out = MessageDeltaOutput::new();
                match event["type"].as_str() {
                    // The CLI asking this process, as the in-process server's host.
                    Some("control_request") => {
                        let response = control_response(&event, &tools);
                        stdin.write_all(format!("{response}\n").as_bytes()).await?;
                        stdin.flush().await?;
                        continue;
                    }
                    // The answer to the `initialize` request.
                    Some("control_response") => {
                        let response = &event["response"];
                        if response["subtype"] == "error" {
                            let error = response["error"].as_str().unwrap_or_default();
                            Err(anyhow::anyhow!("claude did not take the tools: {error}"))?;
                        }
                        continue;
                    }
                    // Partial output: text and thinking are surfaced as they come, calls once
                    // the message ends with them.
                    Some("stream_event") => {
                        let inner = &event["event"];
                        let delta = &inner["delta"];
                        match inner["type"].as_str() {
                            Some("message_start") => {
                                start_usage = parse_usage(&inner["message"]["usage"]);
                                calls.clear();
                                continue;
                            }
                            Some("content_block_start") => {
                                let block = &inner["content_block"];
                                if block["type"] == "tool_use" {
                                    tool_use = Some(ToolUse {
                                        id: block["id"].as_str().unwrap_or_default().to_owned(),
                                        name: block["name"].as_str().unwrap_or_default().to_owned(),
                                        input: String::new(),
                                    });
                                }
                                continue;
                            }
                            Some("content_block_delta") => {
                                if let Some(tool_use) = tool_use.as_mut() {
                                    if let Some(json) = delta["partial_json"].as_str() {
                                        tool_use.input.push_str(json);
                                    }
                                    continue;
                                }
                                if let Some(text) = delta["text"].as_str() {
                                    if text.is_empty() || structured {
                                        continue;
                                    }
                                    out.delta = out.delta.with_contents([PartDelta::Text {
                                        text: text.to_owned(),
                                    }]);
                                } else if let Some(thinking) = delta["thinking"].as_str() {
                                    out.delta.thinking = Some(thinking.to_owned());
                                } else {
                                    continue;
                                }
                            }
                            Some("content_block_stop") => {
                                let Some(ToolUse { id, name, input }) = tool_use.take() else {
                                    continue;
                                };
                                if name == STRUCTURED_OUTPUT {
                                    structured_json = input;
                                } else if let Some(name) = name.strip_prefix(TOOL_PREFIX) {
                                    let arguments: serde_json::Value = if input.trim().is_empty() {
                                        serde_json::json!({})
                                    } else {
                                        serde_json::from_str(&input).with_context(|| {
                                            format!("the arguments of a call to {name}: {input}")
                                        })?
                                    };
                                    calls.push(PartDelta::Function {
                                        id: Some(id),
                                        function: PartDeltaFunction::WithParsedArgs {
                                            name: name.to_owned(),
                                            arguments: arguments.into(),
                                        },
                                    });
                                }
                                continue;
                            }
                            Some("message_delta") => {
                                stop_reason = delta["stop_reason"].as_str().map(str::to_owned);
                                reply_usage = final_usage(start_usage.take(), &inner["usage"]);
                                // The reply ends with calls to the given tools. They are the
                                // caller's to run, so the run ends here, before the CLI would
                                // turn to them.
                                if stop_reason.as_deref() != Some("tool_use") || calls.is_empty() {
                                    continue;
                                }
                                out.delta = out.delta.with_tool_calls(std::mem::take(&mut calls));
                                out.finish_reason = Some(FinishReason::ToolCall {});
                                out.usage = reply_usage.take();
                                finished = true;
                            }
                            _ => continue,
                        }
                    }
                    // The run is over: close the message with its usage. The run's own adds
                    // up every request it made, such as a structured reply the CLI rejected and
                    // asked again for, so the reply's is its last message's.
                    Some("result") => {
                        let run_usage = parse_result(&event)?;
                        out.usage = reply_usage.take().or(run_usage);
                        if structured {
                            // The reply the CLI accepted; attempts it rejected are left out.
                            let json = match &event["structured_output"] {
                                serde_json::Value::Null => std::mem::take(&mut structured_json),
                                output => serde_json::to_string(output)?,
                            };
                            if !json.is_empty() {
                                out.delta = out.delta.with_contents([PartDelta::Text { text: json }]);
                            }
                        }
                        out.finish_reason = Some(parse_finish_reason(
                            event["stop_reason"].as_str().or(stop_reason.as_deref()),
                        ));
                        finished = true;
                    }
                    _ => continue,
                }
                if !started {
                    out.delta.role = Some(Role::Assistant);
                    started = true;
                }
                yield out;
                if finished {
                    break;
                }
            }

            if finished {
                // Stops a run that ended at calls, or one that waits for more input.
                let _ = child.start_kill();
                let _ = child.wait().await;
            } else {
                drop(stdin);
                let status = child.wait().await?;
                let stderr = stderr.await.unwrap_or_default();
                Err(anyhow::anyhow!("claude exited with {status} before finishing: {stderr}"))?;
            }
            // The system prompt file is kept until the process is done with it.
            drop(invocation);
        })
    }

    fn total_input_tokens(&self, usage: &TokenUsage) -> u64 {
        usage.input_with_cache()
    }
}

/// A `tool_use` block as it streams in: the call's id, the tool's name as the CLI has it,
/// and the JSON of its arguments so far.
struct ToolUse {
    id: String,
    name: String,
    input: String,
}

/// A ready-to-start `claude` command, the prompt for its stdin, and the tools its
/// in-process server declares.
struct Invocation {
    command: Command,
    /// The prompt's content blocks.
    prompt: Vec<serde_json::Value>,
    tools: Vec<ToolDesc>,
    /// The file holding the system prompt, removed when dropped; kept for as long as the
    /// process runs.
    _system_file: Option<tempfile::TempPath>,
}

impl Invocation {
    /// Starts the CLI and writes the prompt, after the request that registers the in-process
    /// server when there are tools. Stdin is handed back open, for the answers to the CLI's
    /// requests.
    async fn spawn(&mut self) -> anyhow::Result<(Child, ChildStdin)> {
        let mut child = self
            .command
            .spawn()
            .context("failed to start the claude CLI")?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let mut input = String::new();
        if !self.tools.is_empty() {
            let initialize = serde_json::json!({
                "type": "control_request",
                "request_id": "ailoy-initialize",
                "request": {"subtype": "initialize", "sdkMcpServers": [SERVER]},
            });
            input.push_str(&format!("{initialize}\n"));
        }
        let message = serde_json::json!({
            "type": "user",
            "message": {"role": "user", "content": std::mem::take(&mut self.prompt)},
        });
        input.push_str(&format!("{message}\n"));
        stdin.write_all(input.as_bytes()).await?;
        stdin.flush().await?;
        Ok((child, stdin))
    }
}

/// The answer to a control request from the CLI: the in-process server's reply to an MCP
/// message. Any other request is refused.
fn control_response(event: &serde_json::Value, tools: &[ToolDesc]) -> serde_json::Value {
    let request = &event["request"];
    let response = match request["subtype"].as_str() {
        Some("mcp_message") if request["server_name"] == SERVER => Ok(serde_json::json!({
            "mcp_response": mcp_response(&request["message"], tools),
        })),
        subtype => Err(format!(
            "ailoy does not answer {} requests",
            subtype.unwrap_or("unnamed")
        )),
    };
    let response = match response {
        Ok(response) => serde_json::json!({
            "subtype": "success", "request_id": event["request_id"], "response": response,
        }),
        Err(error) => serde_json::json!({
            "subtype": "error", "request_id": event["request_id"], "error": error,
        }),
    };
    serde_json::json!({"type": "control_response", "response": response})
}

/// The in-process server's reply to the JSON-RPC `message`. It lists the tools and runs
/// none: a run that calls them ends before the CLI would ask.
fn mcp_response(message: &serde_json::Value, tools: &[ToolDesc]) -> serde_json::Value {
    let id = &message["id"];
    // A notification takes no reply, but the control request it came in does.
    if id.is_null() {
        return serde_json::json!({"jsonrpc": "2.0", "result": {}});
    }
    let result = match message["method"].as_str() {
        Some("initialize") => serde_json::json!({
            "protocolVersion": message["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER, "version": env!("CARGO_PKG_VERSION")},
        }),
        Some("tools/list") => serde_json::json!({
            "tools": tools.iter().map(|tool| serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            })).collect::<Vec<_>>(),
        }),
        method => {
            return serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("{} is not served here", method.unwrap_or("this")),
                },
            });
        }
    };
    serde_json::json!({"jsonrpc": "2.0", "id": id, "result": result})
}

/// One turn of the transcript.
enum Turn<'a> {
    User(&'a Message),
    Assistant(&'a Message),
    Results(&'a [&'a Message]),
}

/// The start of every prompt, before the transcript.
const TRANSCRIPT_INTRO: &str = "Below is the conversation so far, as a transcript. Write the assistant's reply to its last turn: only the reply itself, without an <assistant> tag.
In the transcript, \"&\" and \"<\" in the turns' contents are escaped as \"&amp;\" and \"&lt;\"; read them as those characters, and write your reply's text unescaped.
The tools the assistant called are recorded in its turns as <tool_call> elements, and what they returned in the <tool_result> elements after them. These only record what happened: to call a tool, call it, rather than writing such an element.

";

/// The prompt as content blocks: runs of text, with each image where it was.
#[derive(Default)]
struct Content {
    blocks: Vec<serde_json::Value>,
    /// The next text starts a block of its own instead of extending the last one.
    split: bool,
}

impl Content {
    fn text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if !std::mem::take(&mut self.split)
            && let Some(last) = self.blocks.last_mut()
            && let Some(serde_json::Value::String(run)) = last.get_mut("text")
        {
            run.push_str(text);
            return;
        }
        self.blocks
            .push(serde_json::json!({"type": "text", "text": text}));
    }

    /// The text, image and value parts of a message, one line apart, with the text escaped.
    fn parts(&mut self, parts: &[Part]) -> anyhow::Result<()> {
        for (i, part) in parts.iter().enumerate() {
            if i > 0 {
                self.text("\n");
            }
            match part {
                Part::Text { text } => self.text(&escape(text)),
                Part::Value { value } => match value.as_str() {
                    Some(s) => self.text(&escape(s)),
                    None => self.text(&escape(&serde_json::to_string(value)?)),
                },
                Part::Image { image } => self.blocks.push(image_block(image).into()),
                Part::Function { .. } => {
                    anyhow::bail!("ClaudeCliModel takes no function parts in message contents")
                }
            }
        }
        Ok(())
    }

    fn turn(&mut self, turn: &Turn, names: &HashMap<&str, &str>) -> anyhow::Result<()> {
        match turn {
            Turn::User(message) => {
                if message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) {
                    anyhow::bail!("only assistant messages can carry tool calls");
                }
                self.text("<user>\n");
                self.parts(&message.contents)?;
                self.text("\n</user>");
                Ok(())
            }
            Turn::Assistant(message) => {
                let text = render_assistant(message)?;
                self.text(&format!("<assistant>\n{text}\n</assistant>"));
                Ok(())
            }
            Turn::Results(results) => self.results(results, names),
        }
    }

    /// A run of tool-result messages, a `<tool_result>` each.
    fn results(&mut self, results: &[&Message], names: &HashMap<&str, &str>) -> anyhow::Result<()> {
        for (i, message) in results.iter().enumerate() {
            if i > 0 {
                self.text("\n");
            }
            let name = message
                .id
                .as_deref()
                .and_then(|id| names.get(id))
                .copied()
                .unwrap_or("unknown");
            self.text(&format!("<tool_result name=\"{}\">\n", escape(name)));
            self.parts(&message.contents)?;
            self.text("\n</tool_result>");
        }
        Ok(())
    }
}

/// The prompt for a conversation that ends with a user message or with tool
/// results: all of it as a transcript.
///
/// Every turn is written the same way whether it is the latest or an earlier one,
/// and starts a content block of its own, so the prompt of one call is the start of
/// the next one's, block for block. (The CLI does not reuse it from the cache yet;
/// see [`ClaudeCliModel`].)
fn render_prompt(conversation: &[&Message]) -> anyhow::Result<Vec<serde_json::Value>> {
    // Tool results carry only the call id; the name comes from the call.
    let names: HashMap<&str, &str> = conversation
        .iter()
        .flat_map(|m| m.tool_calls.iter().flatten())
        .filter_map(|part| match part {
            Part::Function { id, function } => Some((id.as_str(), function.name.as_str())),
            _ => None,
        })
        .collect();

    let mut turns = Vec::new();
    let mut i = 0;
    while i < conversation.len() {
        let message = conversation[i];
        match &message.role {
            Role::User => turns.push(Turn::User(message)),
            Role::Assistant => turns.push(Turn::Assistant(message)),
            Role::Tool => {
                // Results of the calls in one turn go back together.
                let end = conversation[i..]
                    .iter()
                    .position(|m| m.role != Role::Tool)
                    .map_or(conversation.len(), |n| i + n);
                turns.push(Turn::Results(&conversation[i..end]));
                i = end;
                continue;
            }
            role => anyhow::bail!("ClaudeCliModel does not support {role} messages here"),
        }
        i += 1;
    }

    match turns.last() {
        None => anyhow::bail!("no message to send"),
        Some(Turn::Assistant(_)) => {
            anyhow::bail!("the last message must be a user message or tool results")
        }
        Some(Turn::User(message)) if message.contents.is_empty() => {
            anyhow::bail!("the user message is empty")
        }
        _ => {}
    }

    let mut prompt = Content::default();
    prompt.text(TRANSCRIPT_INTRO);
    for (i, turn) in turns.iter().enumerate() {
        if i > 0 {
            prompt.split = true;
            prompt.text("\n\n");
        }
        prompt.turn(turn, &names)?;
    }
    Ok(prompt.blocks)
}

/// An assistant turn: its text, then a `<tool_call>` for each call, with its arguments as
/// JSON; escaped as the transcript has it.
fn render_assistant(message: &Message) -> anyhow::Result<String> {
    let mut lines = vec![escape(&texts_of(message)?)];
    lines.retain(|text| !text.is_empty());
    for part in message.tool_calls.iter().flatten() {
        let Part::Function { function, .. } = part else {
            anyhow::bail!("a tool call must be a function part");
        };
        lines.push(format!(
            "<tool_call name=\"{}\">{}</tool_call>",
            escape(&function.name),
            escape(&serde_json::to_string(&function.arguments)?)
        ));
    }
    Ok(lines.join("\n"))
}

/// Escapes `&` and `<`, so that text in the transcript cannot open or close its tags.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

/// The text of a message that cannot carry tool calls.
fn text_of(message: &Message) -> anyhow::Result<String> {
    if message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty()) {
        anyhow::bail!("only assistant messages can carry tool calls");
    }
    texts_of(message)
}

/// The text parts of a message joined; any other part is an error. For system
/// and assistant messages, which go into the prompt as plain text.
fn texts_of(message: &Message) -> anyhow::Result<String> {
    let texts = message
        .contents
        .iter()
        .map(|part| {
            part.as_text().ok_or_else(|| {
                anyhow::anyhow!(
                    "ClaudeCliModel supports only text in system and assistant messages"
                )
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(texts.join("\n"))
}

/// The usage from the CLI's final `result` object, or an error if the run failed.
fn parse_result(result: &serde_json::Value) -> anyhow::Result<Option<TokenUsage>> {
    if result["is_error"].as_bool().unwrap_or(false) {
        let text = result["result"].as_str().unwrap_or_default();
        let subtype = result["subtype"].as_str().unwrap_or("error");
        anyhow::bail!("claude failed ({subtype}): {text}");
    }
    Ok(parse_usage(&result["usage"]))
}

fn parse_usage(u: &serde_json::Value) -> Option<TokenUsage> {
    Some(TokenUsage {
        input_tokens: u["input_tokens"].as_u64()?,
        output_tokens: u["output_tokens"].as_u64()?,
        cache_creation_input_tokens: u["cache_creation_input_tokens"].as_u64(),
        cache_read_input_tokens: u["cache_read_input_tokens"].as_u64(),
    })
}

/// The usage of a message once it has ended: `message_delta` carries the final output
/// count, and the input counts only in some versions, so `message_start`'s fill the rest.
fn final_usage(start: Option<TokenUsage>, u: &serde_json::Value) -> Option<TokenUsage> {
    let Some(mut usage) = start else {
        return parse_usage(u);
    };
    if let Some(n) = u["output_tokens"].as_u64() {
        usage.output_tokens = n;
    }
    if let Some(n) = u["input_tokens"].as_u64() {
        usage.input_tokens = n;
    }
    if let Some(n) = u["cache_creation_input_tokens"].as_u64() {
        usage.cache_creation_input_tokens = Some(n);
    }
    if let Some(n) = u["cache_read_input_tokens"].as_u64() {
        usage.cache_read_input_tokens = Some(n);
    }
    Some(usage)
}

/// How a reply without tool calls ended, by the API's `stop_reason`.
fn parse_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("max_tokens" | "model_context_window_exceeded") => FinishReason::Length {},
        Some("refusal") => FinishReason::Refusal {
            reason: "reason: refusal".to_owned(),
        },
        // `end_turn`, `stop_sequence`, and `tool_use` for the CLI's `StructuredOutput` call.
        _ => FinishReason::Stop {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The counts the CLI reported for a run whose system prompt came from the cache, measured
    /// on 2026-10-23 with Claude Code 2.1.308.
    #[test]
    fn total_input_tokens_adds_the_prompt_cache() {
        let usage = parse_usage(&serde_json::json!({
            "input_tokens": 10,
            "cache_read_input_tokens": 21_373,
            "cache_creation_input_tokens": 0,
            "output_tokens": 40
        }))
        .unwrap();
        assert_eq!(ClaudeCliModel::new().total_input_tokens(&usage), 21_383);
    }

    fn weather() -> ToolDesc {
        crate::tool::ToolDescBuilder::new("get_weather")
            .description("Get the weather for a city.")
            .parameters(crate::to_value!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }))
            .build()
    }

    #[test]
    fn the_server_lists_the_tools_and_runs_none() {
        let tools = [weather()];
        let request = |message: serde_json::Value| {
            serde_json::json!({
                "type": "control_request",
                "request_id": "r1",
                "request": {"subtype": "mcp_message", "server_name": "ailoy", "message": message},
            })
        };

        let response = control_response(
            &request(
                serde_json::json!({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"}}),
            ),
            &tools,
        );
        assert_eq!(response["response"]["subtype"], "success");
        assert_eq!(response["response"]["request_id"], "r1");
        let reply = &response["response"]["response"]["mcp_response"];
        assert_eq!(reply["id"], 0);
        assert_eq!(reply["result"]["protocolVersion"], "2025-11-25");

        let response = control_response(
            &request(serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})),
            &tools,
        );
        let listed = &response["response"]["response"]["mcp_response"]["result"]["tools"][0];
        assert_eq!(listed["name"], "get_weather");
        assert_eq!(listed["inputSchema"]["required"][0], "city");

        let response = control_response(
            &request(serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
            &tools,
        );
        let reply = &response["response"]["response"]["mcp_response"];
        assert!(
            reply.get("id").is_none() && reply["result"].is_object(),
            "{reply}"
        );

        let response = control_response(
            &request(serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call"})),
            &tools,
        );
        let reply = &response["response"]["response"]["mcp_response"];
        assert_eq!(reply["error"]["code"], -32601);

        let response = control_response(
            &serde_json::json!({"type": "control_request", "request_id": "r2",
                "request": {"subtype": "can_use_tool"}}),
            &tools,
        );
        assert_eq!(response["response"]["subtype"], "error");
    }

    /// Runs `events` as the CLI's output, from a script standing in for it, and returns what
    /// the model makes of them.
    #[cfg(unix)]
    async fn replay(events: &[serde_json::Value], tools: &[ToolDesc]) -> MessageOutput {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let program = dir.path().join("claude");
        let lines: String = events.iter().map(|e| format!("{e}\n")).collect();
        // Reads its input to the end after writing, as the CLI waits for more of it.
        std::fs::write(
            &program,
            format!("#!/bin/sh\ncat <<'EOF'\n{lines}EOF\nexec cat > /dev/null\n"),
        )
        .unwrap();
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();

        let model = ClaudeCliModel::new().with_program(program.to_string_lossy());
        let messages = [Message::new(Role::User).with_contents([Part::text("Weather?")])];
        model
            .infer(&messages, tools, &LangModelOptions::default())
            .await
            .unwrap()
    }

    fn event(event: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"type": "stream_event", "event": event})
    }

    /// Two calls in one reply are both taken, with the arguments as the API streams them,
    /// and the run ends before the CLI turns to them.
    #[cfg(unix)]
    #[tokio::test]
    async fn the_calls_a_reply_ends_with_are_taken() {
        let call = |index: u64, id: &str, json: [&str; 2]| {
            [
                event(
                    serde_json::json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": id,
                        "name": "mcp__ailoy__get_weather", "input": {}}}),
                ),
                event(
                    serde_json::json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json[0]}}),
                ),
                event(
                    serde_json::json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": json[1]}}),
                ),
                event(serde_json::json!({"type": "content_block_stop", "index": index})),
            ]
        };
        let mut events = vec![
            event(serde_json::json!({"type": "message_start",
                "message": {"usage": {"input_tokens": 3, "output_tokens": 1}}})),
            event(
                serde_json::json!({"type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}}),
            ),
            event(
                serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Checking both."}}),
            ),
            event(serde_json::json!({"type": "content_block_stop", "index": 0})),
        ];
        events.extend(call(1, "toolu_1", ["{\"city\": ", "\"Seoul\"}"]));
        events.extend(call(2, "toolu_2", ["{\"city\"", ": \"Busan\"}"]));
        events.extend([
            event(serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 40}})),
            // What the CLI goes on to, past the point the run ends at.
            serde_json::json!({"type": "result", "subtype": "success", "is_error": true,
                "result": "not reached"}),
        ]);

        let output = replay(&events, &[weather()]).await;
        assert_eq!(output.finish_reason, FinishReason::ToolCall {});
        assert_eq!(output.message.contents[0].as_text(), Some("Checking both."));
        let calls: Vec<_> = output
            .message
            .tool_calls
            .iter()
            .flatten()
            .filter_map(Part::as_function)
            .map(|(id, name, args)| (id.to_owned(), name.to_owned(), args.clone()))
            .collect();
        assert_eq!(
            calls,
            [
                (
                    "toolu_1".into(),
                    "get_weather".into(),
                    crate::to_value!({"city": "Seoul"})
                ),
                (
                    "toolu_2".into(),
                    "get_weather".into(),
                    crate::to_value!({"city": "Busan"})
                ),
            ]
        );
        assert_eq!(output.usage.unwrap().output_tokens, 40);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_reply_reports_its_last_messages_usage() {
        // A run of two requests, measured on 2026-10-23 with Claude Code 2.1.308: the CLI ran
        // its own tool, then answered, and its `result` added up both. The input counts are
        // the measured ones; the two output counts are made up to add up to the run's 172.
        let start = |input: u64, read: u64, written: u64| {
            event(
                serde_json::json!({"type": "message_start", "message": {"usage": {
                "input_tokens": input, "cache_read_input_tokens": read,
                "cache_creation_input_tokens": written, "output_tokens": 1}}}),
            )
        };
        let result = serde_json::json!({"type": "result", "subtype": "success",
            "is_error": false, "stop_reason": "end_turn", "usage": {"input_tokens": 18,
            "cache_read_input_tokens": 35_628, "cache_creation_input_tokens": 8_468,
            "output_tokens": 172}});
        let two_requests = vec![
            start(10, 14_235, 7_158),
            event(serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 70}})),
            start(8, 21_393, 1_310),
            event(
                serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "done"}}),
            ),
            event(serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 102}})),
            result.clone(),
        ];
        // With no message to read it from, the run's usage stands.
        for (events, output_tokens, total) in
            [(two_requests, 102, 22_711), (vec![result], 172, 44_114)]
        {
            let usage = replay(&events, &[weather()]).await.usage.unwrap();
            assert_eq!(usage.output_tokens, output_tokens, "{usage:?}");
            assert_eq!(ClaudeCliModel::new().total_input_tokens(&usage), total);
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_reply_without_calls_ends_at_the_result() {
        let events = [
            event(serde_json::json!({"type": "message_start",
                "message": {"usage": {"input_tokens": 3, "output_tokens": 1}}})),
            event(
                serde_json::json!({"type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": "Sunny."}}),
            ),
            event(serde_json::json!({"type": "message_delta",
                "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}})),
            serde_json::json!({"type": "result", "subtype": "success", "is_error": false,
                "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}}),
        ];
        let output = replay(&events, &[weather()]).await;
        assert_eq!(output.finish_reason, FinishReason::Stop {});
        assert_eq!(output.message.contents[0].as_text(), Some("Sunny."));
        assert!(output.message.tool_calls.is_none_or(|c| c.is_empty()));
    }

    #[test]
    fn transcript_pairs_results_with_calls() {
        let messages = [
            Message::new(Role::User).with_contents([Part::text("weather?")]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "c1",
                "get_weather",
                serde_json::json!({"city": "Seoul", "days": 1}),
            )]),
            Message::new(Role::Tool)
                .with_id("c1")
                .with_contents([Part::text("sunny")]),
        ];
        let blocks = render_prompt(&messages.iter().collect::<Vec<_>>()).unwrap();
        // A block per turn.
        assert_eq!(blocks.len(), 3);
        let prompt: String = blocks.iter().map(|b| b["text"].as_str().unwrap()).collect();
        assert!(prompt.contains(
            "<assistant>\n<tool_call name=\"get_weather\">{\"city\":\"Seoul\",\"days\":1}\
             </tool_call>\n</assistant>"
        ));
        assert!(prompt.ends_with("<tool_result name=\"get_weather\">\nsunny\n</tool_result>"));
    }

    #[test]
    fn images_go_where_they_were() {
        let image = || Part::image_embedded("image/png", vec![1, 2, 3].into()).unwrap();
        let messages = [
            Message::new(Role::User).with_contents([Part::text("look"), image()]),
            Message::new(Role::Assistant).with_tool_calls([Part::function(
                "c1",
                "imgread",
                serde_json::json!({"path": "a.png"}),
            )]),
            Message::new(Role::Tool)
                .with_id("c1")
                .with_contents([image()]),
        ];
        let blocks = render_prompt(&messages.iter().collect::<Vec<_>>()).unwrap();
        let types: Vec<_> = blocks.iter().map(|b| b["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            ["text", "image", "text", "text", "text", "image", "text"]
        );
        assert!(
            blocks[0]["text"]
                .as_str()
                .unwrap()
                .ends_with("<user>\nlook\n")
        );
        assert_eq!(blocks[1]["source"]["data"], "AQID");
        assert!(
            blocks[4]["text"]
                .as_str()
                .unwrap()
                .ends_with("<tool_result name=\"imgread\">\n")
        );
        assert!(
            blocks[6]["text"]
                .as_str()
                .unwrap()
                .starts_with("\n</tool_result>")
        );
    }
}
