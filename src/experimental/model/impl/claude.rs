//! A language model backed by the Claude Code CLI in headless mode (`claude -p`).
//!
//! Each call starts one `claude` process, writes the prompt to its stdin and
//! reads the reply as one JSON event per line from its stdout
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

use std::{collections::HashMap, process::Stdio};

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

/// Runs Claude through the `claude` CLI instead of calling the API directly.
///
/// The CLI's login (subscription or API key) is used as is. System messages
/// replace Claude Code's own system prompt; without any, the prompt is empty.
///
/// Limits of going through the CLI:
/// - Tools are described in the prompt rather than declared to the API, so a
///   call is only as reliable as the model's adherence to the format.
/// - Text only: image parts are rejected.
/// - The CLI takes a single prompt, so earlier turns, tool calls and tool
///   results are written into it as a transcript.
#[derive(Clone, Debug)]
pub struct ClaudeModel {
    program: String,
    model: Option<String>,
}

impl Default for ClaudeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaudeModel {
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

        let mut command = Command::new(&self.program);
        command.args([
            "-p",
            "--tools",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
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
        command.args(["--system-prompt", system.as_str()]);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // Dropping a stream mid-reply ends the process with it.
            .kill_on_drop(true);

        Ok(Invocation {
            command,
            prompt,
            tools: tools.to_vec(),
        })
    }
}

impl InferLangModel for ClaudeModel {
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
        let invocation = match self.invocation(messages, tools, options) {
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
            let mut scanner = CallScanner::default();
            // Input usage from `message_start`, for a reply cut short at a tool call.
            let mut start_usage = None;
            // Inside the `StructuredOutput` call, whose arguments are the reply, and past it.
            let mut in_structured = false;
            let mut structured_done = false;
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
                        if inner["type"] == "message_start" {
                            start_usage = parse_usage(&inner["message"]["usage"]);
                            continue;
                        }
                        if inner["type"] == "content_block_start" {
                            let block = &inner["content_block"];
                            in_structured = block["type"] == "tool_use"
                                && block["name"] == STRUCTURED_OUTPUT;
                            continue;
                        }
                        if inner["type"] == "content_block_stop" && in_structured {
                            in_structured = false;
                            structured_done = true;
                            continue;
                        }
                        let delta = &inner["delta"];
                        if in_structured {
                            let Some(json) = delta["partial_json"].as_str() else {
                                continue;
                            };
                            if json.is_empty() {
                                continue;
                            }
                            out.delta = out.delta.with_contents([PartDelta::Text { text: json.to_owned() }]);
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
                                out.usage = start_usage.take();
                                finished = true;
                            } else if out.delta.contents.is_empty() {
                                continue;
                            }
                        } else if let Some(thinking) = delta["thinking"].as_str() {
                            out.delta.thinking = Some(thinking.to_owned());
                        } else {
                            continue;
                        }
                    }
                    // The run is over: close the message with its usage.
                    Some("result") => {
                        let usage = parse_result(&event)?;
                        if scanner.in_calls {
                            Err(anyhow::anyhow!("the reply ended inside {CALLS_OPEN}"))?;
                        }
                        let text = scanner.flush();
                        if !text.is_empty() && !structured {
                            out.delta = out.delta.with_contents([PartDelta::Text { text }]);
                        }
                        out.finish_reason = Some(FinishReason::Stop {});
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

            if finished {
                // Stops a reply cut short at a tool call; a no-op once the run has ended.
                let _ = child.start_kill();
                let _ = child.wait().await;
            } else {
                let status = child.wait().await?;
                let stderr = stderr.await.unwrap_or_default();
                Err(anyhow::anyhow!("claude exited with {status} before finishing: {stderr}"))?;
            }
        })
    }
}

/// A ready-to-start `claude` command, the prompt for its stdin, and the tools
/// whose schemas type the parsed call arguments.
struct Invocation {
    command: Command,
    prompt: String,
    tools: Vec<ToolDesc>,
}

impl Invocation {
    async fn spawn(mut self) -> anyhow::Result<Child> {
        let mut child = self
            .command
            .spawn()
            .context("failed to start the claude CLI")?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        stdin.write_all(self.prompt.as_bytes()).await?;
        // Closing stdin ends the prompt.
        drop(stdin);
        Ok(child)
    }
}

/// Splits streamed reply text into plain text and the first `<function_calls>` block.
///
/// Text that may be the start of the tag is held back until it is known not to
/// be, and so is trailing whitespace, so the text before a call ends cleanly.
#[derive(Default)]
struct CallScanner {
    buf: String,
    in_calls: bool,
    emitted: bool,
}

impl CallScanner {
    /// Feeds one piece of text. Returns the text that is safe to show, and the
    /// inside of the `<function_calls>` block once it closes.
    fn push(&mut self, piece: &str) -> (String, Option<String>) {
        self.buf.push_str(piece);
        let mut text = String::new();
        if !self.in_calls {
            if let Some(at) = self.buf.find(CALLS_OPEN) {
                text = self.take_text(at);
                self.buf.drain(..CALLS_OPEN.len());
                self.in_calls = true;
            } else {
                let held = held_len(&self.buf);
                return (self.take_text(self.buf.len() - held), None);
            }
        }
        let block = self
            .buf
            .find(CALLS_CLOSE)
            .map(|end| self.buf[..end].to_owned());
        (text, block)
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
        if !self.emitted {
            text = text.trim_start().to_owned();
        }
        text.truncate(text.trim_end().len());
        self.emitted |= !text.is_empty();
        text
    }
}

/// How many trailing bytes of `buf` to hold back: a possible start of
/// [`CALLS_OPEN`], plus the whitespace before it.
fn held_len(buf: &str) -> usize {
    let partial = (1..CALLS_OPEN.len().min(buf.len()) + 1)
        .rev()
        .find(|&n| {
            let at = buf.len() - n;
            buf.is_char_boundary(at) && CALLS_OPEN.starts_with(&buf[at..])
        })
        .unwrap_or(0);
    let rest = &buf[..buf.len() - partial];
    partial + (rest.len() - rest.trim_end().len())
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
        let mut arguments = serde_json::Map::new();
        let mut params = body;
        while let Some(start) = params.find(PARAM_OPEN) {
            let (key, after) = params[start + PARAM_OPEN.len()..]
                .split_once("\">")
                .context("malformed <parameter> tag")?;
            let (value, after) = after
                .split_once(PARAM_CLOSE)
                .context("unterminated <parameter>")?;
            params = after;
            arguments.insert(key.to_owned(), parse_argument(value, key, schema));
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

/// Every argument arrives as text: string parameters keep it as is, the rest
/// are read as JSON and fall back to the text when that fails.
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

/// One turn of the transcript, already rendered.
enum Turn {
    User(String),
    Assistant(String),
    Results(String),
}

/// The prompt for a conversation that ends with a user message or with tool
/// results. A lone user message is sent as is; earlier turns go in front of
/// the latest one as a transcript.
fn render_prompt(conversation: &[&Message]) -> anyhow::Result<String> {
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
            Role::User => turns.push(Turn::User(text_of(message)?)),
            Role::Assistant => turns.push(Turn::Assistant(render_assistant(message)?)),
            Role::Tool => {
                // Results of the calls in one turn go back in one block.
                let end = conversation[i..]
                    .iter()
                    .position(|m| m.role != Role::Tool)
                    .map_or(conversation.len(), |n| i + n);
                turns.push(Turn::Results(render_results(
                    &conversation[i..end],
                    &names,
                )?));
                i = end;
                continue;
            }
            role => anyhow::bail!("ClaudeModel does not support {role} messages here"),
        }
        i += 1;
    }

    let Some((latest, history)) = turns.split_last() else {
        anyhow::bail!("no message to send");
    };
    let latest = match latest {
        Turn::User(text) | Turn::Results(text) => text,
        Turn::Assistant(_) => {
            anyhow::bail!("the last message must be a user message or tool results")
        }
    };
    if history.is_empty() {
        return Ok(latest.clone());
    }

    let mut prompt =
        String::from("Continue this conversation as the assistant. The earlier turns:\n\n");
    for turn in history {
        match turn {
            Turn::User(text) => prompt.push_str(&format!("<user>\n{text}\n</user>\n\n")),
            Turn::Assistant(text) => {
                prompt.push_str(&format!("<assistant>\n{text}\n</assistant>\n\n"))
            }
            Turn::Results(block) => prompt.push_str(&format!("{block}\n\n")),
        }
    }
    prompt.push_str("Now reply to the latest message:\n\n");
    prompt.push_str(latest);
    Ok(prompt)
}

/// An assistant turn: its text, then its tool calls as a `<function_calls>` block.
fn render_assistant(message: &Message) -> anyhow::Result<String> {
    let mut text = texts_of(message)?;
    let calls = message.tool_calls.iter().flatten();
    let mut block = String::new();
    for part in calls {
        let Part::Function { function, .. } = part else {
            anyhow::bail!("a tool call must be a function part");
        };
        block.push_str(&format!("{INVOKE_OPEN}{}\">\n", function.name));
        for (key, value) in function.arguments.as_object().into_iter().flatten() {
            let value = match value.as_str() {
                Some(s) => s.to_owned(),
                None => serde_json::to_string(value)?,
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

/// A run of tool-result messages as one `<function_results>` block.
fn render_results(results: &[&Message], names: &HashMap<&str, &str>) -> anyhow::Result<String> {
    let mut block = String::from("<function_results>\n");
    for message in results {
        let name = message
            .id
            .as_deref()
            .and_then(|id| names.get(id))
            .copied()
            .unwrap_or("unknown");
        let mut output = Vec::new();
        for part in &message.contents {
            match part {
                Part::Text { text } => output.push(text.clone()),
                Part::Value { value } => output.push(match value.as_str() {
                    Some(s) => s.to_owned(),
                    None => serde_json::to_string(value)?,
                }),
                _ => anyhow::bail!("ClaudeModel supports only text and value tool results"),
            }
        }
        block.push_str(&format!(
            "<result>\n<name>{name}</name>\n<output>{}</output>\n</result>\n",
            output.join("\n")
        ));
    }
    block.push_str("</function_results>");
    Ok(block)
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
                .ok_or_else(|| anyhow::anyhow!("ClaudeModel supports only text parts"))
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
        let prompt = render_prompt(&messages.iter().collect::<Vec<_>>()).unwrap();
        assert!(prompt.contains(
            "<invoke name=\"get_weather\">\n<parameter name=\"city\">Seoul</parameter>\n\
             <parameter name=\"days\">1</parameter>\n</invoke>"
        ));
        assert!(prompt.ends_with(
            "<function_results>\n<result>\n<name>get_weather</name>\n<output>sunny</output>\n\
             </result>\n</function_results>"
        ));
    }
}
