//! Builds what flows onto the screen during a run.
//!
//! # What reaches the screen differs from what lands in a file
//!
//! What the person running this wants is not SQL but **what is happening right now**.
//! And what the app knows is not what only the agent knows.
//!
//! - What the app knows exactly: which tools were called how many times, and how many
//!   rows came back. Tool results arrive as `Part::Value`, so reading them with
//!   [`Value::pointer`] leaves nothing to guess.
//! - What only the agent knows: why this query, now. That has to be said by the agent,
//!   which is why `prompt.rs` asks for one sentence per step.
//!
//! So [`Trace`] **holds tool calls until the agent speaks.** When words arrive, the
//! preceding group folds into one line and attaches under them. Turning 45 lines of
//! truncated SQL into ten-odd lines of "what was done, why, and how many are left" is
//! all this module does.
//!
//! Full command text goes to a file through [`Trace::queries`], not to the screen. On
//! screen it was cut to width and could not be reproduced; in the file nothing is cut.
//!
//! [`Value::pointer`]: ailoy::datatype::Value::pointer

use anyhow::{Context, Result};

use ailoy::{
    datatype::Value,
    message::{Message, MessageOutput, Part, PartFunction, Role},
};

/// The name of the query-log file.
///
/// The extension is not `.sql` because the contents are not SQL but `headhunting …`
/// shell commands. Flags change the answer, so stripping out the SQL alone would not
/// reproduce anything.
pub const QUERY_LOG: &str = "queries.log";

/// What the agent must not reach.
///
/// # Why this counts rather than blocks
///
/// The file tools (`read`, `write`, `glob`, `grep`) go through the console and are
/// confined to the mounted tree. **`shell` is not** — `cortex-local-console` spawns
/// `sh -c` on the host and only matches `current_dir` to the session, so an absolute path
/// goes anywhere. In a real run the agent wrote intermediate results to `/tmp` and those
/// files stayed on the host.
///
/// Narrowing the tree therefore narrows the **default path**; it does not build a wall.
/// A wall would need the console server to spawn the shell inside an isolated filesystem,
/// and that is not this example's call.
///
/// So what is checked is whether it happened. A run that read the answer key or the
/// scoring criteria cannot be used to evaluate, and letting that pass quietly would leave
/// it unknowable what a good score means.
const OFF_LIMITS: &[&str] = &[
    // Holds trap labels and verdicts.
    "ground_truth",
    // Per-posting scoring criteria: who should be picked and who should be rejected.
    "expected/",
    // The entire source pool — a way to read all of it without going through the command.
    "candidates.json",
    "narration.json",
];

/// The most calls in one group to list individually.
///
/// Past that, listing every number reads worse than reporting the largest.
const CHAIN_MAX: usize = 5;

/// How far a failure reason is cut to fit one line.
const REASON_WIDTH: usize = 90;

/// The tokens this run spent.
///
/// # Why the four are counted separately
///
/// They are priced differently: list input at 1×, cache writes at 1.25×, cache reads at
/// 0.1×. A sum of the four therefore means nothing, and only by holding them apart can
/// the cost be stated.
///
/// And counting `input_tokens` **alone** inverts the picture. Once caching takes hold
/// that figure collapses — not because fewer tokens went out but because they moved to
/// `cache_read_input_tokens`. Measured: with a 2,007-token instruction served from cache,
/// `input_tokens` fell to 12. On that figure alone the cost reads as near zero.
#[derive(Default, Clone, Copy)]
pub struct Usage {
    /// Input billed at list price (1×).
    pub input: u64,
    pub output: u64,
    /// Input written to cache. 1.25×.
    pub cache_write: u64,
    /// Input served from cache. 0.1×.
    pub cache_read: u64,
}

impl Usage {
    /// Input tokens weighted by price.
    ///
    /// The bill is proportional to this. It is printed because listing the four figures
    /// and leaving the reader to apply the weights means nobody applies them.
    pub fn effective_input(&self) -> u64 {
        (self.input as f64 + self.cache_write as f64 * 1.25 + self.cache_read as f64 * 0.1) as u64
    }

    /// The input tokens that would have been billed at list price without caching.
    ///
    /// Both what was read from cache and what was written to it would have gone out as
    /// plain input without caching. Put beside [`Self::effective_input`], the saving is a
    /// measured figure rather than a claim.
    pub fn uncached_input(&self) -> u64 {
        self.input + self.cache_write + self.cache_read
    }
}

/// Watches the tool traffic and splits what goes on screen from what lands in a file.
#[derive(Default)]
pub struct Trace {
    /// Calls awaiting a result, keyed by tool_call id.
    ///
    /// A `Vec` because even a busy turn raises only a handful, and order has to hold so
    /// the row counts on screen appear in the order the agent called them.
    pending: Vec<(String, Call)>,

    /// Calls that finished after the agent last spoke.
    group: Vec<Call>,

    /// The full command text, bound for `queries.log`.
    queries: Vec<Query>,

    /// How many calls have failed so far. Used in the run summary.
    failures: usize,

    /// Cumulative token usage.
    usage: Usage,

    /// Which assistant response this is. Attached to the query log.
    turn: usize,

    /// Shell commands that reached something off limits outside the tree. See
    /// [`OFF_LIMITS`].
    escapes: Vec<String>,
}

/// One entry as it will be written to the log file.
/// A shell command that went outside the tree. Normally there are none.
pub struct Query {
    pub turn: usize,
    pub cmd: String,
    /// How many pool commands this call held. Usually 1, or as many as were chained
    /// together in the shell.
    pub commands: usize,
}

struct Call {
    kind: Kind,
    outcome: Option<Outcome>,
}

enum Kind {
    /// A shell call that invoked `headhunting`. Holds **every command in that call**.
    ///
    /// Several rather than one because that is what real runs do. The agent packs
    /// `headhunting search --help; echo ===; headhunting read --help` into a single
    /// `shell` call, and calls it repeatedly inside loops. Reading only the first word
    /// produced a screen saying 13 calls while the query log recorded 23.
    Pool(Vec<Cmd>),
    /// A call that wrote a file. Holds the path.
    Write(String),
    /// Any other tool. Holds its name.
    Other(String),
}

/// The three `headhunting` commands.
///
/// They are told apart on screen because they are three different jobs. Search builds a
/// candidate set, read is where the judgment happens, and query asks what neither could.
/// Counted as one lump — "12 pool calls" — a run that swept broadly and one that read
/// carefully look the same.
///
/// And the query count is a signal in itself. Frequent use of the emergency exit means
/// something could not be asked through `search` or `read`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Cmd {
    Search,
    Read,
    Query,
}

enum Outcome {
    Rows(u64),
    Bytes(u64),
    Failed(String),
    /// Finished, with no scale to report.
    Done,
}

impl Trace {
    /// Takes one message off the stream and returns what to put on screen.
    ///
    /// Tool results do not go on screen directly. They attach to the pending call and
    /// come out as a row count when the group next folds.
    pub fn observe(&mut self, output: &MessageOutput) -> String {
        // Counted before the role is examined. Tool results have `usage` as `None`
        // (`message.rs`: "tool results leave this as `None`"), so they pass through.
        if let Some(u) = &output.usage {
            self.usage.input += u.input_tokens;
            self.usage.output += u.output_tokens;
            self.usage.cache_write += u.cache_creation_input_tokens.unwrap_or(0);
            self.usage.cache_read += u.cache_read_input_tokens.unwrap_or(0);
        }

        let msg = &output.message;

        if msg.role == Role::Tool {
            self.settle(msg);
            return String::new();
        }

        let mut out = String::new();

        // When the agent speaks, the previous step is over. Folding the group before
        // those words is what makes it read as "did this → got that many → so next…".
        let said = text_of(&msg.contents);
        let said = said.trim();
        if !said.is_empty() {
            out.push_str(&self.flush());
            out.push_str(said);
            out.push_str("\n\n");
        }

        self.turn += 1;
        for call in msg.tool_calls.iter().flatten() {
            if let Part::Function { id, function } = call {
                self.enqueue(id.clone(), function);
            }
        }
        out
    }

    /// Folds whatever group is left once the stream ends.
    pub fn finish(&mut self) -> String {
        self.flush()
    }

    pub fn queries(&self) -> &[Query] {
        &self.queries
    }

    pub fn failures(&self) -> usize {
        self.failures
    }

    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Writes the full command text to a file.
    ///
    /// # Why a file rather than the screen
    ///
    /// On screen it was cut to width. All 45 entries broke off mid-command — a record
    /// nothing could be reproduced from — while taking most of the screen and burying
    /// what the agent said. Nothing is cut here, so it can actually be reproduced.
    ///
    /// This lives on [`Trace`] because [`Self::queries`] does. Moving it to the caller
    /// would add one more piece of wiring to hand the list across.
    pub fn write_queries(&self, path: &std::path::Path, jd: &std::path::Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("query-log directory {}", parent.display()))?;
        }
        let mut body = format!(
            "-- headhunter query log\n\
             -- posting {}\n\
             -- {} calls holding {} commands. The turn number is which assistant response \
             made the call.\n",
            jd.display(),
            self.queries.len(),
            self.queries.iter().map(|q| q.commands).sum::<usize>()
        );
        for q in &self.queries {
            body.push_str(&format!("\n-- turn {}\n{}\n", q.turn, q.cmd));
        }
        std::fs::write(path, body).with_context(|| format!("query log {}", path.display()))
    }

    /// Shell commands that reached something off limits outside the tree. Normally empty.
    pub fn escapes(&self) -> &[String] {
        &self.escapes
    }

    fn enqueue(&mut self, id: String, f: &PartFunction) {
        let cmd = f
            .arguments
            .pointer("/cmd")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        // Separately from whether this is a pool command, check whether the shell line
        // reached something it must not.
        if let Some(hit) = OFF_LIMITS.iter().find(|needle| cmd.contains(**needle)) {
            let _ = hit;
            self.escapes.push(cmd.to_string());
        }

        let found = commands_in(cmd);
        let kind = if !found.is_empty() {
            self.queries.push(Query {
                turn: self.turn,
                cmd: cmd.to_string(),
                commands: found.len(),
            });
            Kind::Pool(found)
        } else if f.name == "write" || f.name == "edit" {
            let path = f
                .arguments
                .pointer("/path")
                .or_else(|| f.arguments.pointer("/file_path"))
                .and_then(|v| v.as_str())
                .unwrap_or("(no path)");
            Kind::Write(path.to_string())
        } else {
            Kind::Other(f.name.clone())
        };

        self.pending.push((
            id,
            Call {
                kind,
                outcome: None,
            },
        ));
    }

    /// Attaches a tool result to the call that made it.
    ///
    /// The pairing is exact: `Part::Function`'s `id` and the tool-result message's
    /// `Message::id` are the same value (`with_id` in `src/tool/func.rs`).
    fn settle(&mut self, msg: &Message) {
        let Some(id) = msg.id.as_deref() else { return };
        let Some(at) = self.pending.iter().position(|(k, _)| k == id) else {
            return;
        };
        let (_, mut call) = self.pending.remove(at);

        let value = msg.contents.iter().find_map(|p| match p {
            Part::Value { value } => Some(value),
            _ => None,
        });
        call.outcome = Some(value.map_or(Outcome::Done, |v| outcome_of(&call.kind, v)));
        if matches!(call.outcome, Some(Outcome::Failed(_))) {
            self.failures += 1;
        }
        self.group.push(call);
    }

    /// Folds the gathered calls into a few lines.
    fn flush(&mut self) -> String {
        if self.group.is_empty() {
            return String::new();
        }
        let group = std::mem::take(&mut self.group);
        let mut out = String::new();

        // The three commands are counted apart. Lumped together, a run that swept broadly
        // and one that read carefully look like the same line.
        for command in [Cmd::Search, Cmd::Read, Cmd::Query] {
            let calls: usize = group
                .iter()
                .map(|c| match &c.kind {
                    Kind::Pool(cmds) => cmds.iter().filter(|x| **x == command).count(),
                    _ => 0,
                })
                .sum();
            if calls == 0 {
                continue;
            }
            let sizes: Vec<u64> = group
                .iter()
                .filter_map(|c| match (&c.kind, &c.outcome) {
                    (Kind::Pool(cmds), Some(Outcome::Rows(n))) if cmds == &[command] => Some(*n),
                    _ => None,
                })
                .collect();
            out.push_str(&format!(
                "  {} {calls}×{}\n",
                name_of(command),
                sizes_note(command, &sizes)
            ));
        }

        // Failures are listed individually. Passing them over quietly leaves it unclear
        // why the next search narrowed — an FTS5 hyphen error disappeared exactly that way
        // and left not one line in the prose. They come after the count because this line
        // answers why two calls produced one row figure.
        for call in &group {
            if let Some(Outcome::Failed(why)) = &call.outcome {
                out.push_str(&format!("  ⚠ {} failed · {why}\n", label(&call.kind)));
            }
        }

        // Files are listed individually. One artifact appearing is an event in itself.
        for call in &group {
            if let (Kind::Write(path), Some(outcome)) = (&call.kind, &call.outcome) {
                let how = match outcome {
                    Outcome::Bytes(n) => size(*n),
                    _ => String::new(),
                };
                out.push_str(&format!("  wrote {path}  {how}\n"));
            }
        }

        for (name, n) in tally(&group) {
            out.push_str(&format!("  {name} {n}×\n"));
        }

        if !out.is_empty() {
            out.push('\n');
        }
        out
    }
}

/// Counts the other tools by name.
fn tally(group: &[Call]) -> Vec<(&str, usize)> {
    let mut out: Vec<(&str, usize)> = Vec::new();
    for call in group {
        if let Kind::Other(name) = &call.kind {
            match out.iter_mut().find(|(n, _)| *n == name.as_str()) {
                Some((_, count)) => *count += 1,
                None => out.push((name.as_str(), 1)),
            }
        }
    }
    out
}

fn label(kind: &Kind) -> &str {
    match kind {
        // A mixed call cannot carry one name. This is used on the failure line, so it
        // says only that something in the pool failed.
        Kind::Pool(cmds) => match cmds.as_slice() {
            [one] => name_of(*one),
            _ => "pool",
        },
        Kind::Write(_) => "wrote",
        Kind::Other(name) => name,
    }
}

fn name_of(cmd: Cmd) -> &'static str {
    match cmd {
        Cmd::Search => "search",
        Cmd::Read => "read",
        Cmd::Query => "query",
    }
}

/// How many times, and by which command, this shell line reaches the pool.
///
/// The agent may also call `cd` or `ls`, so the tool name alone does not decide it. And
/// **one call carries several commands** — chained with semicolons, repeated inside a
/// loop, piped into shell tools. So all of them are read, not the first word.
///
/// Shell syntax is not parsed; only words are read. If the word after `headhunting` is a
/// subcommand, that counts as one reach. A `headhunting` inside quotes could be
/// miscounted, but that is far rarer than the mismatch from not counting at all.
fn commands_in(cmd: &str) -> Vec<Cmd> {
    let mut out = Vec::new();
    let mut words = cmd.split_whitespace().peekable();
    while let Some(word) = words.next() {
        // A pipe or semicolon may be attached: `… | headhunting`, `;headhunting`
        if word.trim_start_matches(['|', ';', '&', '(', '`', '$']) != "headhunting" {
            continue;
        }
        match words.peek().copied() {
            Some("search") => out.push(Cmd::Search),
            Some("read") => out.push(Cmd::Read),
            Some("query") => out.push(Cmd::Query),
            _ => {}
        }
    }
    out
}

/// Reads scale or failure out of what a tool returned.
fn outcome_of(kind: &Kind, v: &Value) -> Outcome {
    // The shell family gives an exit_code. If it is not 0, the first stderr line has to
    // reach the screen.
    if let Some(code) = v.pointer("/exit_code").and_then(|x| x.as_integer())
        && code != 0
    {
        let why = v.pointer("/stderr").and_then(|x| x.as_str()).unwrap_or("");
        return Outcome::Failed(first_line(why));
    }
    if let Some(err) = v.pointer("/error").and_then(|x| x.as_str()) {
        return Outcome::Failed(first_line(err));
    }
    if let Some(n) = v.pointer("/bytes_written").and_then(|x| x.as_unsigned()) {
        return Outcome::Bytes(n);
    }
    // Scale is read only when there was one command. A mixed call's output concatenates
    // several answers, and counting that as one command's rows would state a wrong figure
    // with confidence.
    if let Kind::Pool(cmds) = kind
        && let [cmd] = cmds.as_slice()
        && let Some(stdout) = v.pointer("/stdout").and_then(|x| x.as_str())
        && let Some(n) = rows_in(*cmd, stdout)
    {
        return Outcome::Rows(n);
    }
    Outcome::Done
}

/// Reads scale out of an answer. What is counted differs per command.
///
/// `read` answers in blocks, not a table, so a line count says nothing about scale — a
/// person with three positions runs past ten lines while one with two takes six. People
/// are counted instead.
///
/// `search` and `query` answer with a table. The command appends `-- N of M rows`
/// **only when truncated**, so with that line the total is read, and without it the lines
/// minus the header are the rows. Counting what was truncated would make the narrowing
/// figure wrong end to end.
fn rows_in(cmd: Cmd, stdout: &str) -> Option<u64> {
    let s = stdout.trim_end();
    if s.trim().is_empty() {
        return None;
    }
    if cmd == Cmd::Read {
        // Each person has one line starting with an id. The `<id>  — not in the pool`
        // line for an absent id is one too, so it counts as well — one answer per id
        // asked, which is right.
        let people = s
            .lines()
            .filter(|l| l.starts_with("urn:li:person:"))
            .count();
        return Some(people as u64);
    }
    if let Some(note) = s
        .lines()
        .next_back()
        .filter(|l| l.starts_with("-- ") && l.ends_with(" rows"))
        && let Some(total) = note.split_whitespace().rev().nth(1)
        && let Ok(n) = total.parse()
    {
        return Some(n);
    }
    // `search` puts a spelling line before the table and separates it with a blank line.
    // The table starts after the last blank line — counting that line would add two rows
    // to every search.
    let table = s.rsplit_once("\n\n").map_or(s, |(_, rest)| rest);
    Some(table.lines().count().saturating_sub(1) as u64)
}

/// Scale in one phrase. `read` counts people; the rest count rows.
fn sizes_note(cmd: Cmd, sizes: &[u64]) -> String {
    let unit = if cmd == Cmd::Read { "people" } else { "rows" };
    rows_note(sizes, unit)
}

fn rows_note(rows: &[u64], unit: &str) -> String {
    if rows.is_empty() {
        return String::new();
    }
    if rows.len() <= CHAIN_MAX {
        // Commas, not arrows. Nothing guarantees that calls in one group narrow each
        // other, and an arrow would assert a relation that is not there.
        let each = rows
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!(" · {each} {unit}")
    } else {
        format!(
            " · up to {} {unit}",
            rows.iter().max().copied().unwrap_or(0)
        )
    }
}

fn first_line(s: &str) -> String {
    let line = s.trim().lines().next().unwrap_or("").trim();
    if line.chars().count() > REASON_WIDTH {
        format!("{}…", line.chars().take(REASON_WIDTH).collect::<String>())
    } else {
        line.to_string()
    }
}

/// Bytes as a size a person reads.
pub fn size(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

/// A token count as a figure a person reads.
pub fn toks(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    }
}

fn text_of(parts: &[Part]) -> String {
    parts
        .iter()
        .filter_map(|part| match part {
            Part::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a tool result from JSON.
    ///
    /// `ailoy::to_value!` is not used because that macro requires `indexmap` as a direct
    /// dependency. Going through [`Value`]'s `Deserialize` is cheaper than adding a
    /// dependency for one test.
    fn result(json: &str) -> Value {
        serde_json::from_str(json).expect("the test fixture has to be valid JSON")
    }

    /// One assistant response, carrying words and tool calls together.
    fn says(text: &str, calls: &[(&str, &str, &str)]) -> MessageOutput {
        let mut msg = Message::new(Role::Assistant);
        if !text.is_empty() {
            msg.contents = vec![Part::text(text)];
        }
        if !calls.is_empty() {
            msg.tool_calls = Some(
                calls
                    .iter()
                    .map(|(id, name, args)| Part::Function {
                        id: (*id).to_string(),
                        function: PartFunction {
                            name: (*name).to_string(),
                            arguments: result(args),
                        },
                    })
                    .collect(),
            );
        }
        wrap(msg)
    }

    /// One tool result. It attaches back to its call by `id`.
    fn returns(id: &str, json: &str) -> MessageOutput {
        wrap(
            Message::new(Role::Tool)
                .with_contents([Part::value(result(json))])
                .with_id(id.to_string()),
        )
    }

    fn wrap(message: Message) -> MessageOutput {
        MessageOutput {
            message,
            finish_reason: ailoy::message::FinishReason::Stop {},
            usage: None,
            depth: None,
            source_agent: None,
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // The console-rendering harness
    //
    // Feeds a real query log through [`Trace`] to build the screen. It is for previewing
    // that screen without running the app, and the formatting matches what the app would
    // produce because it goes through the same code.
    //
    // One input per line.
    //
    //     # <sentence>         what the agent said
    //     <command>\t<rows>     a pool call. ERR in place of rows means it failed
    //     > <path>\t<bytes>     a file write
    //
    //     HEADHUNTER_SCRIPT=<path> \
    //       cargo test -p headhunter -- --ignored --nocapture render_console
    // ────────────────────────────────────────────────────────────────────────
    #[test]
    #[ignore = "for looking at the screen. It asserts nothing"]
    fn render_console() {
        let Ok(path) = std::env::var("HEADHUNTER_SCRIPT") else {
            eprintln!("HEADHUNTER_SCRIPT is not set");
            return;
        };
        let script = std::fs::read_to_string(&path).expect("the script has to be readable");

        let mut t = Trace::default();
        let mut screen = String::new();
        let mut n = 0usize;

        for line in script.lines() {
            if let Some(said) = line.strip_prefix("# ") {
                screen.push_str(&t.observe(&says(said, &[])));
                continue;
            }
            n += 1;
            let id = format!("c{n}");
            let (head, tail) = line.rsplit_once('\t').expect("it has to split on a tab");

            if let Some(path) = head.strip_prefix("> ") {
                let args = format!(r#"{{"path":"{path}","content":"…"}}"#);
                screen.push_str(&t.observe(&says("", &[(&id, "write", &args)])));
                screen.push_str(&t.observe(&returns(
                    &id,
                    &format!(r#"{{"ok":true,"bytes_written":{tail}}}"#),
                )));
                continue;
            }

            let args = serde_json::json!({ "cmd": head }).to_string();
            screen.push_str(&t.observe(&says("", &[(&id, "shell", &args)])));
            let result = if tail == "ERR" {
                r#"{"stdout":"","stderr":"Error: no such column\n","exit_code":1}"#.to_string()
            } else {
                format!(
                    r#"{{"stdout":"{{\"columns\":[],\"rows\":[],\"total_rows\":{tail}}}","stderr":"","exit_code":0}}"#
                )
            };
            screen.push_str(&t.observe(&returns(&id, &result)));
        }
        screen.push_str(&t.finish());

        // Tokens: the call count comes from the number of pool calls, priced with
        // measured figures. Fixed prefix = 2,007 instruction (measured) + ~1,200 tool
        // schema + 450 posting.
        let calls = (t.queries().len() + 5) as u64;
        let fixed = 3_657u64;
        let (mut history, mut u) = (0u64, Usage::default());
        for turn in 0..calls {
            let grew = if turn < 8 { 1_400 } else { 700 };
            u.input += 12;
            u.output += if turn + 4 >= calls { 3_000 } else { 250 };
            if turn == 0 {
                u.cache_write += fixed;
            } else {
                u.cache_read += fixed + history;
                u.cache_write += grew;
            }
            history += grew;
        }

        println!("  posting  eval/jd/backend-rust.md");
        println!("  model    anthropic/claude-sonnet-5");
        println!("  tree     /Users/…/ailoy/examples/headhunter");
        println!("  out      run-manual/backend-rust");
        println!("  max      32000 tokens/response\n");
        print!("{screen}");
        println!("--- run summary ---");
        println!(
            "{} turns · {} pool calls · {} failed · {} tool calls · finish Stop",
            calls * 2 - 1,
            t.queries().len(),
            t.failures(),
            n
        );
        println!(
            "tokens  input {} · output {} · cache write {} · cache read {}",
            toks(u.input),
            toks(u.output),
            toks(u.cache_write),
            toks(u.cache_read)
        );
        println!(
            "        effective input {} (uncached it would be {})",
            toks(u.effective_input()),
            toks(u.uncached_input())
        );
        println!("4 artifacts");
        for (p, b) in [
            ("00-shortlist.md", 7571u64),
            ("01-하은-성.md", 1864),
            ("02-reese-whitlock.md", 1893),
            ("03-채원-노.md", 1820),
        ] {
            println!("  {:>8}  {}", size(b), p);
        }
        println!(
            "query log  run-manual/backend-rust/queries.log  ({} calls, untruncated)",
            t.queries().len()
        );
        println!("score     eval/run_eval.py --score run-manual/backend-rust");
    }

    /// An assistant response carrying usage.
    fn billed(input: u64, output: u64, write: u64, read: u64) -> MessageOutput {
        let mut o = wrap(Message::new(Role::Assistant));
        o.usage = Some(ailoy::message::TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: Some(write),
            cache_read_input_tokens: Some(read),
        });
        o
    }

    /// The four figures accumulate separately, and the weighted value comes out.
    ///
    /// The numbers are measured. Sending a request with a 2,007-token instruction twice
    /// gives `creation 2007` on the first and `read 2007` on the second.
    #[test]
    fn usage_accumulates_per_price_tier() {
        let mut t = Trace::default();
        t.observe(&billed(12, 8, 2007, 0)); // first — writes the cache
        t.observe(&billed(12, 8, 0, 2007)); // second — reads it
        let u = t.usage();

        assert_eq!(u.input, 24);
        assert_eq!(u.output, 16);
        assert_eq!(u.cache_write, 2007);
        assert_eq!(u.cache_read, 2007);

        // 24 + 2007×1.25 + 2007×0.1 = 24 + 2508.75 + 200.7 = 2733
        assert_eq!(u.effective_input(), 2733);
        // Without caching both would have sent 2007 at list price.
        assert_eq!(u.uncached_input(), 4038);
    }

    /// Tool results have no `usage`, so they do not accumulate.
    #[test]
    fn tool_results_do_not_bill() {
        let mut t = Trace::default();
        t.observe(&billed(100, 20, 0, 0));
        t.observe(&returns("c1", r#"{"stdout":"x","exit_code":0}"#));
        assert_eq!(t.usage().input, 100);
    }

    /// On `input` alone the cost reads as near zero once caching takes hold.
    ///
    /// What this test holds is not the arithmetic but **what has to reach the screen**.
    /// In a 46-turn run the list-price input was 550 while the effective input was 150K.
    /// Printing only the first reads as caching having removed the cost.
    #[test]
    fn the_headline_number_is_effective_not_input() {
        let mut t = Trace::default();
        for _ in 0..46 {
            t.observe(&billed(12, 400, 900, 21_000));
        }
        let u = t.usage();
        assert_eq!(u.input, 552);
        assert!(
            u.effective_input() > u.input * 100,
            "effective input {} has to exceed list input {} a hundredfold",
            u.effective_input(),
            u.input
        );
        assert!(u.uncached_input() > u.effective_input() * 5);
    }

    /// What one run looks like on screen.
    ///
    /// What this holds is not the formatting but **the reading order**. The agent's words
    /// come first and the result of the calls those words asked for attaches beneath them.
    /// Flushing tool calls immediately inverts that, putting the row count ahead of the
    /// sentence explaining it.
    #[test]
    fn the_screen_reads_as_reason_then_result() {
        let mut t = Trace::default();
        let mut screen = String::new();

        for output in [
            says(
                "Gating on the must-haves alone to build the candidate set.",
                &[
                    (
                        "c1",
                        "shell",
                        r#"{"cmd":"headhunting search --skill rust --city Seoul,Seongnam --min-years 4"}"#,
                    ),
                    (
                        "c2",
                        "shell",
                        r#"{"cmd":"headhunting search --skill rust --columns salary"}"#,
                    ),
                ],
            ),
            returns(
                "c1",
                r#"{"stdout":"skill \"rust\" -> Rust(52)\n\nid  name\nx  a\ny  b\n-- 2 of 56 rows\n","stderr":"","exit_code":0}"#,
            ),
            returns(
                "c2",
                r#"{"stdout":"","stderr":"headhunting search: unknown column \"salary\"\n","exit_code":2}"#,
            ),
            says(
                "Reading the leading candidates side by side.",
                &[(
                    "c3",
                    "shell",
                    r#"{"cmd":"headhunting read urn:li:person:aaaa urn:li:person:bbbb"}"#,
                )],
            ),
            returns(
                "c3",
                r#"{"stdout":"urn:li:person:aaaa  A\n  skills  x\n\nurn:li:person:bbbb  B\n  skills  y\n","stderr":"","exit_code":0}"#,
            ),
            says(
                "The top three are settled. Writing the shortlist.",
                &[("c4", "write", r#"{"path":"00-shortlist.md","content":"…"}"#)],
            ),
            returns("c4", r#"{"ok":true,"bytes_written":9624}"#),
        ] {
            screen.push_str(&t.observe(&output));
        }
        screen.push_str(&t.finish());

        assert_eq!(
            screen,
            "Gating on the must-haves alone to build the candidate set.\n\
             \n\
             \u{20}\u{20}search 2× · 56 rows\n\
             \u{20}\u{20}⚠ search failed · headhunting search: unknown column \"salary\"\n\
             \n\
             Reading the leading candidates side by side.\n\
             \n\
             \u{20}\u{20}read 1× · 2 people\n\
             \n\
             The top three are settled. Writing the shortlist.\n\
             \n\
             \u{20}\u{20}wrote 00-shortlist.md  9.4KB\n\
             \n",
        );
    }

    #[test]
    fn a_truncated_answer_reports_the_total_not_what_was_shown() {
        // Even with `--limit` carrying back two rows, the figure on screen is 426.
        // Counting what was truncated makes the narrowing figure wrong end to end.
        let out = "id  name\nx  a\ny  b\n-- 2 of 426 rows\n";
        assert_eq!(rows_in(Cmd::Search, out), Some(426));
    }

    /// `search` puts a spelling line before the table. It must not count as a row.
    #[test]
    fn the_spelling_line_is_not_a_row() {
        let out = "skill \"rust\" \u{2192} Rust(3) rust-lang(1)\n\nid  name\nx  a\ny  b\n";
        assert_eq!(rows_in(Cmd::Search, out), Some(2));
    }

    /// `read` answers in blocks, not a table. People are counted, not lines.
    #[test]
    fn reading_counts_people_not_lines() {
        let out = "urn:li:person:aaaa  Riley\n  headline   x\n  skills     y\n\n\
                   urn:li:person:bbbb  Casey\n  headline   z\n";
        assert_eq!(rows_in(Cmd::Read, out), Some(2));
    }

    #[test]
    fn a_query_without_a_truncation_note_is_counted_by_its_lines() {
        let out = "id  name\nx  a\ny  b\n";
        assert_eq!(rows_in(Cmd::Query, out), Some(2));
    }

    #[test]
    fn only_the_apps_own_command_counts_as_reaching_the_pool() {
        // The agent may call `ls` or `cat` too. Counting those as pool calls would stop
        // the figure on screen from meaning how often the pool was reached.
        assert_eq!(
            commands_in("headhunting search --skill rust"),
            vec![Cmd::Search]
        );
        assert_eq!(
            commands_in("  headhunting read urn:li:person:a"),
            vec![Cmd::Read]
        );
        assert_eq!(
            commands_in("headhunting query 'SELECT 1'"),
            vec![Cmd::Query]
        );
        assert_eq!(commands_in("ls -la"), vec![]);
        // The old name. Left in, it would quietly count as a pool call.
        assert_eq!(commands_in("sqlite data/headhunter.db 'SELECT 1'"), vec![]);
    }

    /// **One `shell` call carries several commands.** A real run did exactly that:
    /// reading only the first word, the screen said 13 calls while the query log held 23.
    #[test]
    fn several_commands_in_one_shell_call_are_all_counted() {
        assert_eq!(
            commands_in("headhunting search --help; echo ===; headhunting read --help"),
            vec![Cmd::Search, Cmd::Read]
        );
    }

    /// What comes after a pipe is a shell tool, not another pool call.
    #[test]
    fn what_comes_after_a_pipe_is_not_another_query() {
        assert_eq!(
            commands_in("headhunting search --mentions rust --limit 200 | head -5"),
            vec![Cmd::Search]
        );
    }

    /// Calling it repeatedly inside a loop has to count that many times.
    #[test]
    fn a_loop_that_calls_it_repeatedly_counts_each_call() {
        let cmd = "for s in a b; do\n  echo \"-- $s --\"\n  headhunting search --mentions \"$s\" --limit 5\ndone\nheadhunting read urn:li:person:x";
        assert_eq!(commands_in(cmd), vec![Cmd::Search, Cmd::Read]);
    }

    /// **`shell` is not confined to the mount.**
    ///
    /// `cortex-local-console` spawns `sh -c` on the host and only matches `current_dir`
    /// to the session (`cortex-console-servers/local/src/server/mod.rs`). The file tools go
    /// through the console and stay in the tree; the shell goes anywhere by absolute path —
    /// in a real run the agent wrote intermediate results to `/tmp` and those files stayed
    /// on the host.
    ///
    /// So instead of blocking it, **whether it happened is checked.** A run that reached
    /// the answer key or the scoring criteria through the shell cannot be evaluated, and
    /// that must not pass quietly.
    #[test]
    fn reaching_the_answer_key_through_the_shell_is_caught() {
        let mut t = Trace::default();
        t.observe(&says(
            "Checking the answer.",
            &[(
                "c1",
                "shell",
                r#"{"cmd":"cat /Users/x/examples/headhunter/data/ground_truth.json"}"#,
            )],
        ));
        assert_eq!(t.escapes().len(), 1, "{:?}", t.escapes());
        assert!(t.escapes()[0].contains("ground_truth"));
    }

    #[test]
    fn the_scoring_criteria_are_off_limits_too() {
        let mut t = Trace::default();
        t.observe(&says(
            "",
            &[(
                "c1",
                "shell",
                r#"{"cmd":"grep -r rust ../../eval/expected/"}"#,
            )],
        ));
        assert_eq!(t.escapes().len(), 1);
    }

    /// Pool commands and ordinary shell tools must not trip it. A false alarm is what
    /// makes a real one ignorable.
    #[test]
    fn ordinary_work_does_not_trip_the_check() {
        let mut t = Trace::default();
        for cmd in [
            r#"{"cmd":"headhunting search --mentions rust | head -5"}"#,
            r#"{"cmd":"wc -l /tmp/ids_rust.txt"}"#,
            r#"{"cmd":"cat in/schema.sql"}"#,
        ] {
            t.observe(&says("", &[("c", "shell", cmd)]));
        }
        assert!(t.escapes().is_empty(), "{:?}", t.escapes());
    }

    /// A mixed call reports no scale. Its output concatenates several answers, and
    /// counting that as one command's rows would state a wrong figure with confidence.
    #[test]
    fn a_mixed_call_reports_no_size() {
        let v = result(r#"{"stdout":"id  name\nx  a\n","stderr":"","exit_code":0}"#);
        let kind = Kind::Pool(vec![Cmd::Search, Cmd::Read]);
        assert!(matches!(outcome_of(&kind, &v), Outcome::Done));
    }

    #[test]
    fn a_nonzero_exit_is_a_failure_with_its_reason() {
        let v = result(r#"{"stdout":"","stderr":"Error: no such column: lang\n","exit_code":1}"#);
        match outcome_of(&Kind::Pool(vec![Cmd::Search]), &v) {
            Outcome::Failed(why) => assert_eq!(why, "Error: no such column: lang"),
            _ => panic!("it has to read as a failure"),
        }
    }

    #[test]
    fn a_write_reports_its_size() {
        let v = result(r#"{"ok":true,"bytes_written":9624}"#);
        match outcome_of(&Kind::Write("x.md".into()), &v) {
            Outcome::Bytes(n) => assert_eq!(size(n), "9.4KB"),
            _ => panic!("it has to read as bytes"),
        }
    }

    #[test]
    fn many_queries_collapse_to_the_largest() {
        assert_eq!(rows_note(&[2, 7, 83], "rows"), " · 2, 7, 83 rows");
        assert_eq!(
            rows_note(&[1, 2, 3, 4, 5, 600], "rows"),
            " · up to 600 rows"
        );
        // `read` counts people. The same number means something else with another unit.
        assert_eq!(sizes_note(Cmd::Read, &[7]), " · 7 people");
    }
}
