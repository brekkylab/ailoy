//! Memory: a folder on this host whose `MEMORY.md` holds short notes on the user, which the
//! system message shows and the `memory` tool changes.

use std::{
    io,
    path::{Path, PathBuf},
};

use anyhow::Context as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    datatype::Value,
    tool::{ToolDesc, ToolDescBuilder, ToolFunc},
};

/// The file in a memory's folder that holds its entries.
const FILE_NAME: &str = "MEMORY.md";

/// The file in a memory's folder that edits lock, one at a time, so none is lost.
const LOCK_NAME: &str = ".MEMORY.md.lock";

/// Between two entries of the file, and of the memory as the prompt shows it: a blank line, so
/// entries read as paragraphs and never hold one.
const SEPARATOR: &str = "\n\n";

/// How a date leads an entry: `[2026-11-17] `, the host's date when the entry was last written.
const STAMP_LEN: usize = "[YYYY-MM-DD] ".len();

/// The most characters the file may hold unless [`Memory::with_limit`] says otherwise.
const DEFAULT_LIMIT: usize = 3_000;

/// A folder on this host whose `MEMORY.md` an agent keeps short notes on the user in, with
/// the `memory` tool, and that its system message shows. See
/// [`AgentSpec::memory`](super::AgentSpec::memory).
///
/// The notes are entries, paragraphs apart, each led by the host's date when it was last
/// written, as `[2026-11-17] Drinks tea`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Memory {
    /// The folder, relative to the current directory when the agent is built unless absolute.
    dir: PathBuf,

    /// How many characters `MEMORY.md` may hold.
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

impl Memory {
    /// The memory kept in the folder `dir`, holding at most 3,000 characters.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            limit: DEFAULT_LIMIT,
        }
    }

    /// How many characters the file may hold.
    pub fn with_limit(mut self, chars: usize) -> Self {
        self.limit = chars;
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// This memory with its folder absolute and made, with those on the way to it, so that a
    /// later change of the current directory still finds it and a folder that cannot be made
    /// shows when the agent is built rather than on the first entry.
    pub(super) fn prepared(&self) -> anyhow::Result<Self> {
        let made = std::path::absolute(&self.dir).and_then(|dir| {
            std::fs::create_dir_all(&dir)?;
            Ok(dir)
        });
        let dir =
            made.with_context(|| format!("making the memory folder {}", self.dir.display()))?;
        Ok(Self {
            dir,
            limit: self.limit,
        })
    }

    fn file(&self) -> PathBuf {
        self.dir.join(FILE_NAME)
    }

    /// The entries as the file holds them now; none for a file not written yet.
    fn read(&self) -> io::Result<Vec<String>> {
        match std::fs::read_to_string(self.file()) {
            Ok(text) => Ok(parse_entries(&text)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Make `ops` in order to the file as it is now, all or none, and answer as the tool does.
    ///
    /// Edits wait for each other, from this process or another, on a lock file in the folder,
    /// so two at once both land; on a file system that cannot lock a file, as some network
    /// ones, an edit goes on without waiting. The file is written to a temporary one beside it
    /// and moved into place, so a reader never sees half of it; it is the owner's alone to read.
    fn edit(&self, ops: &[Op]) -> Value {
        let io_error = |e: io::Error| {
            crate::to_value!({
                "error": format!("the memory's MEMORY.md could not be read or written: {e}"),
                "phase": "io",
            })
        };
        let lock_path = self.dir.join(LOCK_NAME);
        let lock = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
        {
            Ok(lock) => lock,
            Err(e) => return io_error(e),
        };
        if let Err(e) = lock.lock() {
            log::warn!(
                "could not lock {}: {e}; editing the memory without waiting for other edits",
                lock_path.display()
            );
        }

        let current = match self.read() {
            Ok(current) => current,
            Err(e) => return io_error(e),
        };
        let entries = match edited(&current, ops, self.limit, &today()) {
            Ok(entries) => entries,
            Err(reason) => {
                return crate::to_value!({
                    "error": format!("{reason}. Nothing was changed."),
                    "phase": "validation",
                    "memory": shown(&current, self.limit),
                });
            }
        };
        if entries != current
            && let Err(e) = self.write(&entries)
        {
            return io_error(e);
        }
        crate::to_value!({
            "result": "saved",
            "memory": shown(&entries, self.limit),
            "note": "Your system prompt still shows the memory from the start of this conversation; this shows up from the next one.",
        })
    }

    fn write(&self, entries: &[String]) -> io::Result<()> {
        let mut text = entries.join(SEPARATOR);
        if !text.is_empty() {
            text.push('\n');
        }
        let mut temp = tempfile::NamedTempFile::new_in(&self.dir)?;
        io::Write::write_all(&mut temp, text.as_bytes())?;
        temp.as_file().sync_all()?;
        temp.persist(self.file()).map_err(|e| e.error)?;
        Ok(())
    }
}

impl From<PathBuf> for Memory {
    fn from(dir: PathBuf) -> Self {
        Self::new(dir)
    }
}

impl From<&Path> for Memory {
    fn from(dir: &Path) -> Self {
        Self::new(dir)
    }
}

impl From<String> for Memory {
    fn from(dir: String) -> Self {
        Self::new(dir)
    }
}

impl From<&str> for Memory {
    fn from(dir: &str) -> Self {
        Self::new(dir)
    }
}

/// The memory's section of the system message, from the file as it is now.
pub(super) fn render_memory(memory: &Memory) -> anyhow::Result<String> {
    let entries = memory
        .read()
        .with_context(|| format!("reading {}", memory.file().display()))?;
    Ok(format!(
        "# Memory\n\n\
         Your memory of this user, kept with the `memory` tool across conversations, each \
         entry led by the date it was last written. It was loaded when this conversation \
         started; your changes show up next time.\n\n\
         [{}/{} chars]\n{}",
        used(&entries),
        memory.limit,
        if entries.is_empty() {
            "(empty)".to_string()
        } else {
            entries.join(SEPARATOR)
        }
    ))
}

/// The `memory` tool, bound to `memory`.
///
/// The memory is captured in the tool's function rather than taken as an argument, so the
/// model never names a folder and cannot name another.
pub(super) fn memory_tool(memory: &Memory) -> (ToolDesc, ToolFunc) {
    let memory = memory.clone();
    let func = crate::tool_func!(async |args: Value| -> Value with [memory = memory.clone()] {
        let ops = match ops(&args) {
            Ok(ops) => ops,
            Err(error) => return crate::to_value!({ "error": error, "phase": "validation" }),
        };
        match tokio::task::spawn_blocking(move || memory.edit(&ops)).await {
            Ok(out) => out,
            Err(e) => crate::to_value!({ "error": e.to_string(), "phase": "io" }),
        }
    });
    (memory_tool_desc(), func)
}

fn memory_tool_desc() -> ToolDesc {
    ToolDescBuilder::new("memory")
        .description(concat!(
            "Keep short notes that are put in your system prompt at the start of every future ",
            "conversation: what you learn about the user that would still help in a later ",
            "conversation (who they are, how they like to be helped, their circumstances and ",
            "plans) and your own notes (agreements, what to avoid, lessons learned). A fact the ",
            "user mentions in passing counts, even inside a request about something else. Leave ",
            "out what only this conversation needs: the request itself, details and progress. ",
            "Each entry is one fact: a new fact is a new entry, even about the same ",
            "user. When a fact changes, replace that fact's entry instead of adding another, and ",
            "never rewrite an entry to fit another fact into it. Each entry starts with the date ",
            "it was last written, as [YYYY-MM-DD], which is added for you: write only the fact, ",
            "and use the dates to tell what may be out of date. ",
            "The memory has a size limit; when it is full, make room in the same call: shorten ",
            "entries first, and remove one only when shortening is not enough, then tell the user ",
            "what you removed. The changes in one call are made in order, all or none. ",
            "Your system prompt keeps the memory as it was when this conversation started; ",
            "changes show up from the next one.",
        ))
        .parameters(crate::to_value!({
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "description": "The changes to make, in order; one change is a list of one.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["add", "replace", "remove"],
                                "description": "'add' a new fact as its own entry; 'replace' the entry containing old_text with content, when its fact changed; 'remove' the entry containing old_text."
                            },
                            "content": {
                                "type": "string",
                                "description": "One fact, without a date: the entry to add, or the whole new text of the replaced entry."
                            },
                            "old_text": {
                                "type": "string",
                                "description": "A short piece of the entry to replace or remove, long enough that only that entry contains it."
                            }
                        },
                        "required": ["action"]
                    }
                }
            },
            "required": ["operations"]
        }))
        .build()
}

/// One change to the memory.
enum Op {
    Add { content: String },
    Replace { old_text: String, content: String },
    Remove { old_text: String },
}

/// The changes the tool's arguments ask for.
fn ops(args: &Value) -> Result<Vec<Op>, String> {
    let ops = match args.pointer("/operations") {
        Some(Value::Array(ops)) if !ops.is_empty() => ops,
        _ => return Err("operations must be a non-empty array of changes".into()),
    };
    ops.iter().map(op).collect()
}

/// One change as the model spelled it.
fn op(value: &Value) -> Result<Op, String> {
    let field = |name: &str| value.pointer(&format!("/{name}")).and_then(|v| v.as_str());
    let required = |name: &'static str| {
        field(name)
            .map(str::to_string)
            .ok_or(format!("missing parameter: {name}"))
    };
    Ok(match field("action") {
        Some("add") => Op::Add {
            content: required("content")?,
        },
        Some("replace") => Op::Replace {
            old_text: required("old_text")?,
            content: required("content")?,
        },
        Some("remove") => Op::Remove {
            old_text: required("old_text")?,
        },
        other => {
            return Err(format!(
                "action must be add, replace or remove, not {other:?}"
            ));
        }
    })
}

/// The memory's entries and how full the file is, as the tool shows them.
fn shown(entries: &[String], limit: usize) -> Value {
    crate::to_value!({
        "usage": format!("{}/{limit} chars", used(entries)),
        "entries": Value::array(entries.to_vec()),
    })
}

/// The host's date, as entries are stamped with it.
pub(super) fn today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// Characters in the file, as its limit counts them.
fn used(entries: &[String]) -> usize {
    entries.join(SEPARATOR).chars().count()
}

/// `current` with `ops` made in order, or the reason none can be; what is added or replaced is
/// stamped with `today`.
///
/// The limit is checked on the result, so one call can make room and fill it, and only when
/// the file grew: one already over its limit (edited by hand) can still shrink.
fn edited(
    current: &[String],
    ops: &[Op],
    limit: usize,
    today: &str,
) -> Result<Vec<String>, String> {
    let mut entries = current.to_vec();
    for op in ops {
        match op {
            Op::Add { content } => entries.push(entry(content, today)?),
            Op::Replace { old_text, content } => {
                let content = entry(content, today)?;
                let i = find(&entries, old_text)?;
                entries[i] = content;
            }
            Op::Remove { old_text } => {
                let i = find(&entries, old_text)?;
                entries.remove(i);
            }
        }
    }
    let size = used(&entries);
    if size > limit && size > used(current) {
        return Err(format!(
            "the memory would be {size} characters, over its limit of {limit}; shorten \
             entries in the same call, or remove one if that is not enough"
        ));
    }
    Ok(entries)
}

/// The file's entries: its paragraphs, trimmed. A line of only spaces separates too, since a
/// person may edit the file.
fn parse_entries(text: &str) -> Vec<String> {
    let mut entries = vec![String::new()];
    for line in text.lines() {
        let current = entries.last_mut().expect("never empty");
        if line.trim().is_empty() {
            entries.push(String::new());
        } else {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
    }
    entries
        .into_iter()
        .map(|e| e.trim().to_string())
        .filter(|e| !e.is_empty())
        .collect()
}

/// `content` as one entry stamped with `today`: trimmed, with blank lines dropped so it reads
/// back as one paragraph, and without a date of its own leading it (one copied from an entry);
/// refused if nothing is left.
fn entry(content: &str, today: &str) -> Result<String, String> {
    let content = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let content = body(content.trim()).trim();
    if content.is_empty() {
        return Err("an entry cannot be empty".into());
    }
    Ok(format!("[{today}] {content}"))
}

/// `entry` without the date that leads it, if one does.
fn body(entry: &str) -> &str {
    let b = entry.as_bytes();
    let stamped = b.len() >= STAMP_LEN
        && b[0] == b'['
        && b[11] == b']'
        && b[12] == b' '
        && b[5] == b'-'
        && b[8] == b'-'
        && [1, 2, 3, 4, 6, 7, 9, 10]
            .iter()
            .all(|&i| b[i].is_ascii_digit());
    if stamped { &entry[STAMP_LEN..] } else { entry }
}

/// The index of the entry `old_text` means. Both are compared without the date that leads
/// them and with whitespace runs as one space: an entry that is the whole of `old_text` is the
/// one meant (the first, if repeated), else the only entry containing it.
fn find(entries: &[String], old_text: &str) -> Result<usize, String> {
    let needle = squash(body(old_text.trim()));
    if needle.is_empty() {
        return Err("old_text cannot be empty".into());
    }
    let bodies: Vec<String> = entries.iter().map(|e| squash(body(e))).collect();
    if let Some(i) = bodies.iter().position(|b| *b == needle) {
        return Ok(i);
    }
    let hits: Vec<usize> = (0..bodies.len())
        .filter(|&i| bodies[i].contains(&needle))
        .collect();
    match hits[..] {
        [i] => Ok(i),
        [] => Err(format!("no entry contains {old_text:?}")),
        _ => Err(format!(
            "{} entries contain {old_text:?}; give a longer old_text that only one contains",
            hits.len()
        )),
    }
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use futures::StreamExt as _;

    use super::*;
    use crate::{message::Message, to_value};

    fn add(content: &str) -> Op {
        Op::Add {
            content: content.into(),
        }
    }

    fn replace(old_text: &str, content: &str) -> Op {
        Op::Replace {
            old_text: old_text.into(),
            content: content.into(),
        }
    }

    fn remove(old_text: &str) -> Op {
        Op::Remove {
            old_text: old_text.into(),
        }
    }

    const DAY: &str = "2026-11-17";

    /// `text` as an entry written on [`DAY`].
    fn dated(text: &str) -> String {
        format!("[{DAY}] {text}")
    }

    fn entries(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|s| s.to_string()).collect()
    }

    fn apply(start: &[String], ops: &[Op]) -> Result<Vec<String>, String> {
        edited(start, ops, DEFAULT_LIMIT, DAY)
    }

    /// A memory in a fresh folder of its own.
    fn fresh() -> Memory {
        Memory::new(tempfile::tempdir().unwrap().keep()).with_limit(100)
    }

    fn value(msg: &Message) -> Value {
        msg.contents[0].as_value().unwrap().clone()
    }

    /// One call of the tool, which needs no sandbox.
    async fn call(func: &ToolFunc, args: Value) -> Value {
        let output = func
            .call_pure(args, "1")
            .expect("a pure tool")
            .next()
            .await
            .unwrap();
        value(&output.message)
    }

    fn phase(out: &Value) -> Option<&str> {
        out.pointer("/phase").and_then(|v| v.as_str())
    }

    fn saved(out: &Value) -> bool {
        out.pointer("/result").and_then(|v| v.as_str()) == Some("saved")
    }

    #[test]
    fn test_entries_round_trip_through_the_file_format() {
        let texts = entries(&["Name is Minsu", "Two\nlines"]);
        let file = texts.join(SEPARATOR) + "\n";
        assert_eq!(parse_entries(&file), texts);
        assert_eq!(parse_entries("a\n  \nb\n\n\n\nc\n"), ["a", "b", "c"]);
        assert!(parse_entries("").is_empty());
    }

    #[test]
    fn test_a_memory_reads_from_a_spec_with_its_default_limit() {
        let memory: Memory = serde_json::from_value(serde_json::json!({ "dir": "/m" })).unwrap();
        assert_eq!(memory, Memory::new("/m"));
        let memory: Memory =
            serde_json::from_value(serde_json::json!({ "dir": "/m", "limit": 500 })).unwrap();
        assert_eq!(memory.limit(), 500);
        assert!(
            serde_json::from_value::<Memory>(serde_json::json!({ "dir": "/m", "limit": -1 }))
                .is_err()
        );
    }

    #[test]
    fn test_edits_add_replace_and_remove_by_old_text() {
        let start = entries(&["Name is Minsu", "Drinks tea"]);
        // Whitespace-lenient, not case-blind.
        assert!(apply(&start, &[replace("drinks   TEA", "x")]).is_err());

        let after = apply(
            &start,
            &[
                replace("Drinks\n tea", "Drinks only oat-milk lattes"),
                remove("Minsu"),
                add("Prefers short answers\n\n  \nin Korean"),
            ],
        )
        .unwrap();
        assert_eq!(
            after,
            [
                dated("Drinks only oat-milk lattes"),
                dated("Prefers short answers\nin Korean")
            ]
        );
    }

    /// What is added or replaced is led by the day it was written, not by a date the model
    /// copied from an entry.
    #[test]
    fn test_entries_are_led_by_the_day_they_were_written() {
        let start = entries(&["Lives in Seoul"]);
        let after = apply(
            &start,
            &[
                replace("Seoul", "[2026-11-10] Lives in Busan"),
                add("Married in (2027-03)"),
            ],
        )
        .unwrap();
        assert_eq!(
            after,
            [dated("Lives in Busan"), dated("Married in (2027-03)")]
        );
        assert_eq!(body(&dated("Drinks tea")), "Drinks tea");
        assert_eq!(body("[2026-1-10] Drinks tea"), "[2026-1-10] Drinks tea");
        assert_eq!(body("Drinks tea"), "Drinks tea");
    }

    /// An edit that cannot be made changes nothing.
    #[test]
    fn test_an_edit_that_cannot_be_made_changes_nothing() {
        let start = entries(&["Lives in Seoul", "Works in Seoul"]);
        let e = apply(&start, &[add("new"), remove("Seoul")]).unwrap_err();
        assert!(e.contains("2 entries"), "{e}");

        for bad in [add("  "), add("\n \n"), remove(" "), remove("Busan")] {
            assert!(apply(&start, &[bad]).is_err());
        }
    }

    /// `old_text` and the entries are compared without their dates: a date the entry no longer
    /// has does not keep it from being found, and an entry that is the whole of `old_text` wins
    /// over one that only contains it.
    #[test]
    fn test_old_text_is_matched_on_the_entries_without_their_dates() {
        let start = entries(&[
            &dated("Lives in Seoul, Gangnam"),
            &dated("Likes tea"),
            &dated("Likes tea with milk"),
        ]);
        assert_eq!(
            apply(
                &start,
                &[replace("[2026-11-10] Lives in Seoul", "Lives in Busan")]
            )
            .unwrap()[0],
            dated("Lives in Busan")
        );
        assert!(apply(&start, &[remove("[2026-11-10] Lives in Daegu")]).is_err());
        // The date alone is in no entry's text.
        assert!(apply(&start, &[remove(DAY)]).is_err());

        assert!(apply(&start, &[remove("tea")]).is_err());
        assert_eq!(
            apply(&start, &[replace(" Likes  tea", "Likes coffee")]).unwrap()[1..],
            [dated("Likes coffee"), dated("Likes tea with milk")]
        );
        assert_eq!(
            apply(&start, &[remove(&dated("Likes tea"))]).unwrap()[1..],
            [dated("Likes tea with milk")]
        );

        // An entry repeated is removed one copy at a time.
        let twice = entries(&["Drinks tea", "Drinks  tea"]);
        assert_eq!(
            apply(&twice, &[remove("Drinks tea")]).unwrap(),
            ["Drinks  tea"]
        );
    }

    #[test]
    fn test_the_limit_applies_to_the_result_and_only_to_growth() {
        let start = entries(&[&"a".repeat(95)]);
        let e = edited(&start, &[add("bbbbb")], 100, DAY).unwrap_err();
        assert!(e.contains("over its limit of 100"), "{e}");
        assert_eq!(
            edited(&start, &[remove("aaa"), add("bbbbb")], 100, DAY).unwrap(),
            [dated("bbbbb")]
        );

        // Over its limit already (edited by hand): may shrink, may not grow.
        let over = entries(&[&"a".repeat(80), &"b".repeat(80)]);
        assert!(edited(&over, &[replace("aaa", "a")], 100, DAY).is_ok());
        assert!(edited(&over, &[add("c")], 100, DAY).is_err());
    }

    #[test]
    fn test_render_shows_the_entries_and_how_full_the_file_is() {
        let memory = fresh().with_limit(500);
        assert!(
            render_memory(&memory)
                .unwrap()
                .ends_with("[0/500 chars]\n(empty)")
        );
        std::fs::write(memory.file(), "Name is Minsu\n\nDrinks tea\n").unwrap();
        let text = render_memory(&memory).unwrap();
        assert!(text.starts_with("# Memory\n"), "{text}");
        assert!(text.contains("`memory` tool"), "{text}");
        assert!(
            text.ends_with("[25/500 chars]\nName is Minsu\n\nDrinks tea"),
            "{text}"
        );
    }

    /// Preparing makes the folder and every missing one on the way, and the folder absolute,
    /// but not the file.
    #[test]
    fn test_prepared_makes_the_folders_but_not_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("agent1/agent2");
        let memory = Memory::new(&folder).prepared().unwrap();
        assert!(folder.is_dir());
        assert_eq!(memory.file(), folder.join("MEMORY.md"));
        assert!(!memory.file().exists());
        assert!(memory.dir().is_absolute());

        // A folder that cannot be made, here because a file has its name, fails now.
        let blocked = dir.path().join("a-file");
        std::fs::write(&blocked, "").unwrap();
        assert!(Memory::new(&blocked).prepared().is_err());
    }

    #[tokio::test]
    async fn test_memory_tool_edits_the_file() {
        let memory = fresh();
        let (desc, func) = memory_tool(&memory);
        assert_eq!(desc.name, "memory");
        assert!(!memory.file().exists());

        let out = call(
            &func,
            to_value!({ "operations": [{ "action": "add", "content": "- Name is Minsu" }] }),
        )
        .await;
        assert!(saved(&out), "{out:?}");
        assert_eq!(
            out.pointer("/memory/usage").unwrap().as_str(),
            Some("28/100 chars")
        );

        let out = call(
            &func,
            to_value!({ "operations": [
                { "action": "replace", "old_text": "Minsu", "content": "Name is Minsu Kim" },
                { "action": "add", "content": "Answer briefly" }
            ] }),
        )
        .await;
        assert!(saved(&out), "{out:?}");
        let day = today();
        assert_eq!(
            std::fs::read_to_string(memory.file()).unwrap(),
            format!("[{day}] Name is Minsu Kim\n\n[{day}] Answer briefly\n")
        );
    }

    /// A refusal changes nothing and shows the entries as they stand, so the model can retry
    /// from them.
    #[tokio::test]
    async fn test_memory_tool_refusal_shows_the_entries() {
        let memory = fresh();
        let (_, func) = memory_tool(&memory);
        call(
            &func,
            to_value!({ "operations": [{ "action": "add", "content": "Drinks tea" }] }),
        )
        .await;
        let tea = format!("[{}] Drinks tea", today());

        for op in [
            to_value!({ "action": "remove", "old_text": "coffee" }),
            to_value!({ "action": "add", "content": "x".repeat(100) }),
        ] {
            let out = call(&func, to_value!({ "operations": [op] })).await;
            assert_eq!(phase(&out), Some("validation"), "{out:?}");
            assert_eq!(
                out.pointer("/memory/entries/0").unwrap().as_str(),
                Some(tea.as_str())
            );
        }

        for args in [
            to_value!({}),
            to_value!({ "action": "add", "content": "x" }),
            to_value!({ "operations": [] }),
            to_value!({ "operations": "[]" }),
            to_value!({ "operations": [{ "action": "add" }] }),
            to_value!({ "operations": [{ "action": "append", "content": "x" }] }),
        ] {
            let out = call(&func, args.clone()).await;
            assert_eq!(phase(&out), Some("validation"), "{args:?}");
        }
        assert_eq!(memory.read().unwrap(), [tea]);
    }

    /// Edits made at once, as a model's parallel tool calls are, wait for each other: every
    /// one lands.
    #[test]
    fn test_edits_at_once_all_land() {
        let memory = fresh().with_limit(DEFAULT_LIMIT);
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let memory = memory.clone();
                std::thread::spawn(move || memory.edit(&[add(&format!("Fact {i}"))]))
            })
            .collect();
        for t in threads {
            assert!(saved(&t.join().unwrap()));
        }
        let mut got: Vec<String> = memory
            .read()
            .unwrap()
            .iter()
            .map(|e| body(e).to_string())
            .collect();
        got.sort();
        assert_eq!(got, (0..8).map(|i| format!("Fact {i}")).collect::<Vec<_>>());
    }
}
