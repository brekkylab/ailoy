//! A language model backed by the Claude Code CLI in headless mode (`claude -p`).
//!
//! Each call starts one `claude` process, writes the prompt to its stdin as one
//! user message (`--input-format stream-json`), so images can go with the text,
//! and reads the reply as one JSON event per line from its stdout
//! (`--output-format stream-json`).
//!
//! Claude Code's own tools and MCP servers are switched off. Tools are instead
//! described in the system prompt, and the model calls them in Claude's XML
//! format, which is parsed out of its reply:
//!
//! ```text
//! <function_calls>
//! <invoke name="get_weather">
//! <parameter name="city">Seoul</parameter>
//! </invoke>
//! </function_calls>
//! ```
//!
//! The process is stopped at the first `</function_calls>`, since the CLI has
//! no stop sequences and the model would otherwise go on to imagine the results.
//! Only a `<function_calls>` at the start of a line and outside fenced or
//! indented code is taken as a call, and none is looked for when no tools were given.

use std::{collections::HashMap, io::Write as _, process::Stdio};

use anyhow::Context as _;
use futures::{
    StreamExt as _,
    future::BoxFuture,
    stream::{self, BoxStream},
};
use tokio::{
    io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader},
    process::{Child, Command},
};

use super::claude_api::image_block;
use crate::{
    datatype::Value,
    experimental::model::{InferLangModel, LangModelOptions, ThinkingEffort},
    message::{
        Delta as _, FinishReason, Message, MessageDeltaOutput, MessageOutput, Part, PartDelta,
        PartDeltaFunction, Role, TokenUsage,
    },
    tool::ToolDesc,
};

/// The tool the CLI adds for `--json-schema`; the reply is its arguments.
const STRUCTURED_OUTPUT: &str = "StructuredOutput";
const CALLS_OPEN: &str = "<function_calls>";
const CALLS_CLOSE: &str = "</function_calls>";
const INVOKE_OPEN: &str = "<invoke name=\"";
const INVOKE_CLOSE: &str = "</invoke>";
const PARAM_OPEN: &str = "<parameter name=\"";
const PARAM_CLOSE: &str = "</parameter>";
/// The namespace Claude's own tool-call tags carry; the model sometimes writes it here too.
const NAMESPACE: &str = "antml:";
/// The tag names read with or without [`NAMESPACE`].
const TAGS: [&str; 3] = ["function_calls", "invoke", "parameter"];
/// The longest a stray word before a bare `<invoke>` is taken to be.
const MAX_STRAY_WORD: usize = 12;

/// Runs Claude through the `claude` CLI instead of calling the API directly.
///
/// The CLI's login (subscription or API key) is used as is. System messages
/// replace Claude Code's own system prompt; without any, the prompt is empty.
///
/// Limits of going through the CLI:
/// - Tools are described in the prompt rather than declared to the API, so a
///   call is only as reliable as the model's adherence to the format.
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

    /// Model alias or full name for `--model` (e.g. `"sonnet"`); unset keeps the CLI default.
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
        let mut system = system
            .into_iter()
            .map(text_of)
            .collect::<anyhow::Result<Vec<_>>>()?
            .join("\n\n");
        if !tools.is_empty() {
            // Laid out like the API's own tool-use system prompt: tools, then the
            // user's system prompt, then the tool configuration.
            system = [render_tools(tools), system, TOOL_CONFIG.to_owned()]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
        }
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
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            // The user's and project's settings (hooks, output style, …) stay out of the
            // model, as does the auto-memory set below.
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
        // writes around it is dropped, though tool calls in it are still taken.
        let structured = options.output_schema.is_some();

        Box::pin(async_stream::try_stream! {
            let tools = invocation.tools.clone();
            let mut child = invocation.spawn().await?;
            let stdout = child.stdout.take().expect("stdout is piped");
            let mut stderr = child.stderr.take().expect("stderr is piped");
            // Drained alongside stdout so a full stderr pipe cannot stall the process.
            let stderr = tokio::spawn(async move {
                let mut buf = String::new();
                let _ = stderr.read_to_string(&mut buf).await;
                buf
            });

            let mut lines = BufReader::new(stdout).lines();
            // Without tools there is nothing to call, so the text is not searched for calls.
            let mut scanner = CallScanner {
                plain: tools.is_empty(),
                ..Default::default()
            };
            // Usage from `message_start`, completed by `message_delta` once the message ends.
            let mut start_usage = None;
            // Why the latest message ended, from its `message_delta`.
            let mut stop_reason = None;
            // The arguments of the latest `StructuredOutput` call, which are the reply. Held
            // to the end, since a call the CLI rejects is followed by another.
            let mut structured_json = String::new();
            // Inside the `StructuredOutput` call, and past it.
            let mut in_structured = false;
            let mut structured_done = false;
            // The reply up to a closed `<function_calls>` block, held until the message's
            // final usage arrives or the model is seen going on past the block.
            let mut pending: Option<MessageDeltaOutput> = None;
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
                    // Partial output: text (minus tool calls) and thinking are surfaced.
                    Some("stream_event") => {
                        let inner = &event["event"];
                        let kind = inner["type"].as_str();
                        let delta = &inner["delta"];
                        if let Some(held) = pending.take() {
                            // Past the block, whitespace is waited through to the message's
                            // end and its usage. Anything more is the model going on to imagine
                            // the results; it is cut off with the usage known so far, whose
                            // output count falls short.
                            match kind {
                                Some("message_delta") => {
                                    out = held;
                                    out.usage = final_usage(start_usage.take(), &inner["usage"]);
                                }
                                Some("content_block_delta")
                                    if delta["text"].as_str().is_some_and(|t| t.trim().is_empty()) =>
                                {
                                    pending = Some(held);
                                    continue;
                                }
                                Some("content_block_start" | "content_block_delta") => {
                                    out = held;
                                    out.usage = start_usage.take();
                                }
                                _ => {
                                    pending = Some(held);
                                    continue;
                                }
                            }
                            finished = true;
                        } else {
                            match kind {
                                Some("message_start") => {
                                    start_usage = parse_usage(&inner["message"]["usage"]);
                                    continue;
                                }
                                Some("message_delta") => {
                                    stop_reason = delta["stop_reason"].as_str().map(str::to_owned);
                                    continue;
                                }
                                Some("content_block_start") => {
                                    let block = &inner["content_block"];
                                    in_structured = block["type"] == "tool_use"
                                        && block["name"] == STRUCTURED_OUTPUT;
                                    if in_structured {
                                        structured_json.clear();
                                    }
                                    continue;
                                }
                                Some("content_block_stop") if in_structured => {
                                    in_structured = false;
                                    structured_done = true;
                                    continue;
                                }
                                Some("content_block_delta") => {}
                                _ => continue,
                            }
                            if in_structured {
                                if let Some(json) = delta["partial_json"].as_str() {
                                    structured_json.push_str(json);
                                }
                                continue;
                            } else if structured_done {
                                // Whatever the model says after the structured reply is not part of it.
                                continue;
                            } else if let Some(piece) = delta["text"].as_str() {
                                let (text, block) = scanner.push(piece);
                                if !text.is_empty() && !structured {
                                    out.delta = out.delta.with_contents([PartDelta::Text { text }]);
                                }
                                if let Some(block) = block {
                                    out.delta = out.delta.with_tool_calls(parse_calls(&block, &tools)?);
                                    out.finish_reason = Some(FinishReason::ToolCall {});
                                    pending = Some(out);
                                    continue;
                                } else if out.delta.contents.is_empty() {
                                    continue;
                                }
                            } else if let Some(thinking) = delta["thinking"].as_str() {
                                out.delta.thinking = Some(thinking.to_owned());
                            } else {
                                continue;
                            }
                        }
                    }
                    // The run is over: close the message with its usage.
                    Some("result") => {
                        let usage = parse_result(&event)?;
                        if let Some(held) = pending.take() {
                            out = held;
                        } else {
                            let block = scanner.finish_bare();
                            if block.is_none() && scanner.in_calls {
                                Err(anyhow::anyhow!("the reply ended inside {CALLS_OPEN}"))?;
                            }
                            let text = scanner.flush();
                            if !text.is_empty() && !structured {
                                out.delta = out.delta.with_contents([PartDelta::Text { text }]);
                            }
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
                            out.finish_reason = Some(match block {
                                Some(block) => {
                                    out.delta = out.delta.with_tool_calls(parse_calls(&block, &tools)?);
                                    FinishReason::ToolCall {}
                                }
                                None => parse_finish_reason(
                                    event["stop_reason"].as_str().or(stop_reason.as_deref()),
                                ),
                            });
                        }
                        out.usage = usage;
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
            // The output ended between a closed block and the message's end.
            if let Some(mut out) = pending.take() {
                out.usage = start_usage.take();
                if !started {
                    out.delta.role = Some(Role::Assistant);
                }
                yield out;
                finished = true;
            }

            if finished {
                // Stops a reply cut short at a tool call; a no-op once the run has ended.
                let _ = child.start_kill();
                let _ = child.wait().await;
            } else {
                let status = child.wait().await?;
                let stderr = stderr.await.unwrap_or_default();
                Err(anyhow::anyhow!("claude exited with {status} before finishing: {stderr}"))?;
            }
            // The system prompt file is kept until the process is done with it.
            drop(invocation);
        })
    }
}

/// A ready-to-start `claude` command, the prompt for its stdin, and the tools
/// whose schemas type the parsed call arguments.
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
    async fn spawn(&mut self) -> anyhow::Result<Child> {
        let mut child = self
            .command
            .spawn()
            .context("failed to start the claude CLI")?;
        let message = serde_json::json!({
            "type": "user",
            "message": {"role": "user", "content": std::mem::take(&mut self.prompt)},
        });
        let mut stdin = child.stdin.take().expect("stdin is piped");
        stdin.write_all(format!("{message}\n").as_bytes()).await?;
        // Closing stdin ends the input.
        drop(stdin);
        Ok(child)
    }
}

/// Splits streamed reply text into plain text and the first `<function_calls>` block.
///
/// Only a tag at the start of a line (after up to three spaces) and outside fenced or
/// indented code opens a block, so a reply can quote the format. Text that may be the start of
/// such a tag is held back until it is known not to be, and so is trailing
/// whitespace, so the text before a call ends cleanly.
///
/// Also reads the ways the model is known to garble the block (anthropics/claude-code#49747
/// and its duplicates): the tags with Claude's own `antml:` namespace, `<invoke>`s without
/// the `<function_calls>` around them, and a stray word (`court`, `call`, …) on the line
/// where `<function_calls>` should be, which is dropped.
#[derive(Default)]
struct CallScanner {
    buf: String,
    /// Passes all text through without looking for calls, as when no tools were given.
    plain: bool,
    in_calls: bool,
    /// The calls opened with a bare `<invoke>`, the `<function_calls>` around it left out,
    /// as the model sometimes writes them. Such a block has no closing tag to stop at, so it
    /// runs to `</function_calls>` if one comes, or else to the end of the reply.
    bare: bool,
    emitted: bool,
    /// Where the text taken out of `buf` left off.
    lines: Lines,
}

/// What the scanner found in its buffer.
enum Opener {
    /// A tag at `at`, `len` bytes long; `bare` for an `<invoke>`.
    Tag {
        at: usize,
        len: usize,
        bare: bool,
    },
    /// What may be the start of a tag, from `at` to the end.
    Partial(usize),
    None,
}

impl CallScanner {
    /// Feeds one piece of text. Returns the text that is safe to show, and the
    /// inside of the `<function_calls>` block once it closes.
    fn push(&mut self, piece: &str) -> (String, Option<String>) {
        self.buf.push_str(piece);
        if self.plain {
            let len = self.buf.trim_end().len();
            return (self.take_text(len), None);
        }
        let mut text = String::new();
        if !self.in_calls {
            let held = match self.find_opener() {
                Opener::Tag {
                    at,
                    len,
                    bare: false,
                } => {
                    text = self.take_text(at);
                    self.buf.drain(..len);
                    self.in_calls = true;
                    0
                }
                // The `<invoke>` stays in the buffer: it is the block's first call.
                Opener::Tag { at, bare: true, .. } => {
                    let stray =
                        stray_word(&self.buf[..at], self.lines.at_line_start()).unwrap_or(at);
                    text = self.take_text(stray);
                    self.buf.drain(..at - stray);
                    self.in_calls = true;
                    self.bare = true;
                    0
                }
                Opener::Partial(at) => at,
                Opener::None => self.buf.len(),
            };
            if !self.in_calls {
                let len = shown_len(&self.buf[..held], self.lines.at_line_start());
                return (self.take_text(len), None);
            }
        }
        strip_namespace(&mut self.buf);
        let block = self
            .buf
            .find(CALLS_CLOSE)
            .map(|end| self.buf[..end].to_owned());
        (text, block)
    }

    /// The first tag in `buf` that opens a block, or what may become one at its end.
    fn find_opener(&self) -> Opener {
        let openers = [
            (CALLS_OPEN.to_owned(), false),
            (INVOKE_OPEN.to_owned(), true),
            (format!("<{NAMESPACE}{}", &CALLS_OPEN[1..]), false),
            (format!("<{NAMESPACE}{}", &INVOKE_OPEN[1..]), true),
        ];
        let mut lines = self.lines.clone();
        let mut pos = 0;
        loop {
            let end = self.buf[pos..].find('\n').map(|n| pos + n + 1);
            let line = &self.buf[pos..end.unwrap_or(self.buf.len())];
            let indent = line.len() - line.trim_start_matches([' ', '\t']).len();
            // Four columns of indentation make a Markdown code block, as a tab does.
            if lines.at_line_start() && indent < 4 && !line[..indent].contains('\t') {
                let rest = &line[indent..];
                for (opener, bare) in &openers {
                    if rest.starts_with(opener.as_str()) {
                        let (at, len, bare) = (pos + indent, opener.len(), *bare);
                        return Opener::Tag { at, len, bare };
                    }
                    if end.is_none() && !rest.is_empty() && opener.starts_with(rest) {
                        return Opener::Partial(pos + indent);
                    }
                }
            }
            lines.advance(line);
            match end {
                Some(end) => pos = end,
                None => return Opener::None,
            }
        }
    }

    /// At the end of the reply, the block of bare `<invoke>`s still open, up to the last
    /// `</invoke>`; whatever follows it is dropped. `None` if no call was opened that way, or
    /// none of them closed.
    fn finish_bare(&mut self) -> Option<String> {
        if !(self.in_calls && self.bare) {
            return None;
        }
        let end = self.buf.rfind(INVOKE_CLOSE)? + INVOKE_CLOSE.len();
        let block = self.buf[..end].to_owned();
        self.buf.clear();
        self.in_calls = false;
        Some(block)
    }

    /// Whatever text is still held back, at the end of the reply.
    fn flush(&mut self) -> String {
        let len = self.buf.trim_end().len();
        self.take_text(len)
    }

    /// Removes `buf[..len]` and returns it. Trailing whitespace is only ever cut
    /// at the end of the reply's text, since it is held back until more follows.
    fn take_text(&mut self, len: usize) -> String {
        let mut text: String = self.buf.drain(..len).collect();
        self.lines.advance(&text);
        if !self.emitted {
            text = text.trim_start().to_owned();
        }
        text.truncate(text.trim_end().len());
        self.emitted |= !text.is_empty();
        text
    }
}

/// Where a run of text leaves off: in fenced code or not, and on which line.
#[derive(Clone, Default)]
struct Lines {
    /// The fence character and length of the code block the text is in, if any.
    fence: Option<(char, usize)>,
    /// The text of the current line so far.
    line: String,
}

impl Lines {
    /// At the start of a line outside fenced code, whitespace aside: where a tag opens a block.
    fn at_line_start(&self) -> bool {
        self.fence.is_none() && self.line.trim().is_empty()
    }

    fn advance(&mut self, text: &str) {
        let mut segments = text.split('\n');
        if let Some(first) = segments.next() {
            self.line.push_str(first);
        }
        for segment in segments {
            self.end_line();
            self.line.push_str(segment);
        }
    }

    /// Opens or closes a fence on a line of three or more backticks or tildes.
    fn end_line(&mut self) {
        let line = self.line.trim();
        if let Some(c @ ('`' | '~')) = line.chars().next() {
            let run = line.chars().take_while(|&x| x == c).count();
            match self.fence {
                None if run >= 3 => self.fence = Some((c, run)),
                Some((fence, len)) if fence == c && run >= len && line[run..].trim().is_empty() => {
                    self.fence = None
                }
                _ => {}
            }
        }
        self.line.clear();
    }
}

/// Drops [`NAMESPACE`] from every whole tool-call tag in a block's `buf`; a tag still being
/// streamed is left as is until it is whole.
fn strip_namespace(buf: &mut String) {
    if !buf.contains(NAMESPACE) {
        return;
    }
    for tag in TAGS {
        for open in ["<", "</"] {
            *buf = buf.replace(&format!("{open}{NAMESPACE}{tag}"), &format!("{open}{tag}"));
        }
    }
}

/// Where the line holding only a stray word starts, if `text` ends with one: what the model
/// sometimes writes in place of `<function_calls>`. `at_line_start` says whether `text`
/// starts a line, since a line begun before it was already shown.
fn stray_word(text: &str, at_line_start: bool) -> Option<usize> {
    let text = text.trim_end();
    let start = match text.rfind('\n') {
        Some(newline) => newline + 1,
        None if at_line_start => 0,
        None => return None,
    };
    let word = text[start..].trim();
    (!word.is_empty()
        && word.len() <= MAX_STRAY_WORD
        && word.chars().all(|c| c.is_ascii_alphabetic()))
    .then_some(start)
}

/// How much of `text` can be shown: all but its trailing whitespace, and before that a
/// line that may be a stray word ([`stray_word`]).
fn shown_len(text: &str, at_line_start: bool) -> usize {
    match stray_word(text, at_line_start) {
        // From the newline before it, so the next push still sees the word start a line.
        Some(start) => text[..start].trim_end().len(),
        None => text.trim_end().len(),
    }
}

/// Parses the inside of a `<function_calls>` block into tool calls, typing each
/// argument by its tool's schema.
fn parse_calls(block: &str, tools: &[ToolDesc]) -> anyhow::Result<Vec<PartDelta>> {
    let mut calls = Vec::new();
    let mut rest = block;
    while let Some(start) = rest.find(INVOKE_OPEN) {
        let (name, after) = rest[start + INVOKE_OPEN.len()..]
            .split_once("\">")
            .context("malformed <invoke> tag")?;
        let (body, after) = after
            .split_once(INVOKE_CLOSE)
            .context("unterminated <invoke>")?;
        rest = after;

        let schema = tools.iter().find(|t| t.name == name).map(|t| &t.parameters);
        let keys: Vec<&str> = schema
            .and_then(|s| s.pointer("/properties"))
            .and_then(|p| p.as_object())
            .map(|p| p.keys().map(String::as_str).collect())
            .unwrap_or_default();
        let mut arguments = serde_json::Map::new();
        let mut params = body;
        let mut last = None;
        loop {
            let start = params.find(PARAM_OPEN);
            // Text after a value is the rest of it, cut at a `</parameter>` the model failed
            // to escape.
            if let Some(key) = last
                && !params[..start.unwrap_or(params.len())].trim().is_empty()
            {
                anyhow::bail!(
                    "malformed call to {name}: text follows the value of `{key}`, which may \
                     hold an unescaped {PARAM_CLOSE}"
                );
            }
            let Some(start) = start else {
                break;
            };
            let (key, after) = params[start + PARAM_OPEN.len()..]
                .split_once("\">")
                .context("malformed <parameter> tag")?;
            let (value, after) = after
                .split_once(PARAM_CLOSE)
                .context("unterminated <parameter>")?;
            params = after;
            // A value holding another parameter's opening tag is one the model failed to
            // close (anthropics/claude-code#49747); taking it as is would hand the tool a
            // corrupted value and drop that parameter, both silently.
            if let Some(other) = keys
                .iter()
                .find(|k| value.contains(&format!("{PARAM_OPEN}{k}\">")))
            {
                anyhow::bail!(
                    "malformed call to {name}: the value of `{key}` runs into the \
                     <parameter> tag of `{other}`"
                );
            }
            arguments.insert(
                key.to_owned(),
                parse_argument(&unescape(value), key, schema),
            );
            last = Some(key);
        }

        calls.push(PartDelta::Function {
            id: Some(format!("toolu_{}", uuid::Uuid::new_v4().simple())),
            function: PartDeltaFunction::WithParsedArgs {
                name: name.to_owned(),
                arguments: serde_json::Value::Object(arguments).into(),
            },
        });
    }
    if calls.is_empty() {
        anyhow::bail!("no <invoke> in {CALLS_OPEN}");
    }
    Ok(calls)
}

/// Every argument arrives as text, already unescaped: string parameters keep it as
/// is, the rest are read as JSON and fall back to the text when that fails.
fn parse_argument(value: &str, key: &str, schema: Option<&Value>) -> serde_json::Value {
    let ty = schema
        .and_then(|s| s.pointer(&format!("/properties/{key}/type")))
        .and_then(|t| t.as_str());
    if ty != Some("string")
        && let Ok(parsed) = serde_json::from_str(value.trim())
    {
        return parsed;
    }
    serde_json::Value::String(value.to_owned())
}

/// Instructions after the user's system prompt, in the place of the API's tool
/// configuration. Not part of the published template: they stand in for the
/// stop sequence and tool-result handling the CLI lacks.
const TOOL_CONFIG: &str = "After a \"</function_calls>\" block, stop immediately and wait: the results will arrive in the next message inside a \"<function_results>\" block.
When no function is needed, just answer, without remarking on the functions.";

/// The system-prompt section that describes the tools and how to call them.
///
/// Follows the template published under "Tool use system prompt" in Anthropic's
/// tool-use docs; its `{{ FORMATTING INSTRUCTIONS }}` placeholder is filled
/// with the `<function_calls>` format.
fn render_tools(tools: &[ToolDesc]) -> String {
    let mut prompt = String::from(
        "In this environment you have access to a set of tools you can use to answer the user's question.
You can invoke functions by writing a \"<function_calls>\" block like the following as part of your reply to the user:
<function_calls>
<invoke name=\"$FUNCTION_NAME\">
<parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</parameter>
...
</invoke>
<invoke name=\"$FUNCTION_NAME2\">
...
</invoke>
</function_calls>
String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
In parameter values, always write \"&\" as \"&amp;\" and \"<\" as \"&lt;\", so that a value can hold any text; values are unescaped before use.
Here are the functions available in JSONSchema format:
<functions>
",
    );
    for tool in tools {
        let function = serde_json::json!({
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        });
        prompt.push_str(&format!("<function>{function}</function>\n"));
    }
    prompt.push_str("</functions>");
    prompt
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

    /// A run of tool-result messages as one `<function_results>` block.
    fn results(&mut self, results: &[&Message], names: &HashMap<&str, &str>) -> anyhow::Result<()> {
        self.text("<function_results>\n");
        for message in results {
            let name = message
                .id
                .as_deref()
                .and_then(|id| names.get(id))
                .copied()
                .unwrap_or("unknown");
            self.text(&format!(
                "<result>\n<name>{}</name>\n<output>",
                escape(name)
            ));
            self.parts(&message.contents)?;
            self.text("</output>\n</result>\n");
        }
        self.text("</function_results>");
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
                // Results of the calls in one turn go back in one block.
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

/// An assistant turn: its text, then its tool calls as a `<function_calls>` block,
/// escaped as the transcript and the call format have it.
fn render_assistant(message: &Message) -> anyhow::Result<String> {
    let mut text = escape(&texts_of(message)?);
    let calls = message.tool_calls.iter().flatten();
    let mut block = String::new();
    for part in calls {
        let Part::Function { function, .. } = part else {
            anyhow::bail!("a tool call must be a function part");
        };
        block.push_str(&format!("{INVOKE_OPEN}{}\">\n", function.name));
        for (key, value) in function.arguments.as_object().into_iter().flatten() {
            let value = match value.as_str() {
                Some(s) => escape(s),
                None => escape(&serde_json::to_string(value)?),
            };
            block.push_str(&format!("{PARAM_OPEN}{key}\">{value}{PARAM_CLOSE}\n"));
        }
        block.push_str(INVOKE_CLOSE);
        block.push('\n');
    }
    if !block.is_empty() {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&format!("{CALLS_OPEN}\n{block}{CALLS_CLOSE}"));
    }
    Ok(text)
}

/// Escapes `&` and `<`, so that text in the transcript or in a parameter value cannot
/// open or close its tags.
fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;")
}

/// Undoes [`escape`] in a parameter value the model wrote, and `&gt;` too, which it may
/// write out of habit. Text it left unescaped passes through, these entities aside.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let (entity, c) = [("&amp;", '&'), ("&lt;", '<'), ("&gt;", '>')]
            .into_iter()
            .find(|(entity, _)| rest.starts_with(entity))
            .unwrap_or(("&", '&'));
        out.push(c);
        rest = &rest[entity.len()..];
    }
    out.push_str(rest);
    out
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

    /// Feeds `pieces` in order; returns the shown text and the closed block, if any.
    fn scan(pieces: &[&str]) -> (String, Option<String>) {
        let mut scanner = CallScanner::default();
        let mut shown = String::new();
        for piece in pieces {
            let (text, block) = scanner.push(piece);
            shown.push_str(&text);
            if block.is_some() {
                return (shown, block);
            }
        }
        shown.push_str(&scanner.flush());
        (shown, None)
    }

    #[test]
    fn plain_text_passes_through() {
        let (text, block) = scan(&["\nHello ", "<b>world</b>", "  \n"]);
        assert_eq!(text, "Hello <b>world</b>");
        assert_eq!(block, None);
    }

    #[test]
    fn tag_split_across_pieces_is_caught() {
        let (text, block) = scan(&[
            "Checking.\n\n<func",
            "tion_calls>\n<invoke name=\"a\">\n</inv",
            "oke>\n</function_",
            "calls>\nimagined result",
        ]);
        assert_eq!(text, "Checking.");
        assert_eq!(block.as_deref(), Some("\n<invoke name=\"a\">\n</invoke>\n"));
    }

    #[test]
    fn a_tag_in_indented_code_is_text() {
        let quoted = "Use this format:\n\n    <function_calls>\n    <invoke name=\"a\">\n    </invoke>\n    </function_calls>";
        let (text, block) = scan(&[quoted]);
        assert_eq!((text.as_str(), block), (quoted, None));
        let (text, block) = scan(&["Format:\n\t<invoke name=\"a\">\n\t</invoke>"]);
        assert_eq!(
            (text.as_str(), block),
            ("Format:\n\t<invoke name=\"a\">\n\t</invoke>", None)
        );

        // Up to three spaces still open a block.
        let (text, block) = scan(&["Checking.\n   <function_calls>\n</function_calls>"]);
        assert_eq!((text.as_str(), block.as_deref()), ("Checking.", Some("\n")));
    }

    #[test]
    fn a_bare_invoke_opens_a_block_that_runs_to_the_end() {
        let mut scanner = CallScanner::default();
        let (text, block) = scanner.push("Reading the skill.\n<inv");
        assert_eq!((text.as_str(), block), ("Reading the skill.", None));
        let (text, block) =
            scanner.push("oke name=\"a\">\n</invoke>\n<invoke name=\"b\">\n</invoke>\n");
        assert_eq!((text.as_str(), block), ("", None));
        assert_eq!(
            scanner.finish_bare().as_deref(),
            Some("<invoke name=\"a\">\n</invoke>\n<invoke name=\"b\">\n</invoke>")
        );
        assert_eq!(scanner.flush(), "");
    }

    #[test]
    fn a_bare_invoke_closed_by_function_calls_ends_there() {
        let (text, block) = scan(&["<invoke name=\"a\">\n</invoke>\n</function_calls>\nimagined"]);
        assert_eq!(text, "");
        assert_eq!(block.as_deref(), Some("<invoke name=\"a\">\n</invoke>\n"));
    }

    #[test]
    fn a_stray_word_before_a_bare_invoke_is_dropped() {
        let mut scanner = CallScanner::default();
        let (text, _) = scanner.push("Reading it now.\ncou");
        assert_eq!(text, "Reading it now.");
        let (text, _) = scanner.push("rt\n<invoke name=\"a\">\n</invoke>");
        assert_eq!(text, "");
        assert_eq!(
            scanner.finish_bare().as_deref(),
            Some("<invoke name=\"a\">\n</invoke>")
        );

        // A word that runs on is text, not a stray.
        let (text, block) = scan(&["call", " me later"]);
        assert_eq!((text.as_str(), block), ("call me later", None));
    }

    #[test]
    fn namespaced_tags_are_read_as_plain_ones() {
        let ns = NAMESPACE;
        let (text, block) = scan(&[
            "Checking.\n<ant",
            &format!(
                "ml:function_calls>\n<{ns}invoke name=\"a\">\n<{ns}parameter name=\"x\">1</{ns}param"
            ),
            &format!("eter>\n</{ns}invoke>\n</{ns}function_calls>"),
        ]);
        assert_eq!(text, "Checking.");
        assert_eq!(
            block.as_deref(),
            Some("\n<invoke name=\"a\">\n<parameter name=\"x\">1</parameter>\n</invoke>\n")
        );
    }

    #[test]
    fn a_value_running_into_another_parameter_is_an_error() {
        let tools = [crate::tool::ToolDescBuilder::new("f")
            .parameters(serde_json::json!({
                "type": "object",
                "properties": {"summary": {"type": "string"}, "files": {"type": "array"}}
            }))
            .build()];
        let block = "<invoke name=\"f\">\n<parameter name=\"summary\">done.</summary>\n\
                     <parameter name=\"files\">[\"a\"]</parameter>\n</invoke>";
        let err = parse_calls(block, &tools).unwrap_err().to_string();
        assert!(
            err.contains("`summary`") && err.contains("`files`"),
            "{err}"
        );
    }

    #[test]
    fn arguments_are_typed_by_schema() {
        let tools = [crate::tool::ToolDescBuilder::new("f")
            .parameters(serde_json::json!({
                "type": "object",
                "properties": {"s": {"type": "string"}, "n": {"type": "integer"}, "o": {}}
            }))
            .build()];
        let block = "<invoke name=\"f\">\n<parameter name=\"s\">42</parameter>\n\
                     <parameter name=\"n\">42</parameter>\n<parameter name=\"o\">{\"k\": [1]}</parameter>\n</invoke>";
        let calls = parse_calls(block, &tools).unwrap();
        let PartDelta::Function {
            function: PartDeltaFunction::WithParsedArgs { name, arguments },
            ..
        } = &calls[0]
        else {
            panic!("not a parsed call: {calls:?}");
        };
        assert_eq!(name, "f");
        assert_eq!(
            serde_json::to_value(arguments).unwrap(),
            serde_json::json!({"s": "42", "n": 42, "o": {"k": [1]}})
        );
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
            "<invoke name=\"get_weather\">\n<parameter name=\"city\">Seoul</parameter>\n\
             <parameter name=\"days\">1</parameter>\n</invoke>"
        ));
        assert!(prompt.ends_with(
            "<function_results>\n<result>\n<name>get_weather</name>\n<output>sunny</output>\n\
             </result>\n</function_results>"
        ));
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
        assert!(blocks[4]["text"].as_str().unwrap().ends_with("<output>"));
        assert!(blocks[6]["text"].as_str().unwrap().starts_with("</output>"));
    }
}
