use std::{
    io,
    path::{Path, PathBuf},
};

/// Between two entries of the file, and of the memory as the prompt shows it: a blank line, so
/// entries read as paragraphs and never hold one.
const SEPARATOR: &str = "\n\n";

/// The file in a memory's folder that holds its entries.
pub const FILE_NAME: &str = "MEMORY.md";

/// The most characters the file may hold unless [`Memory::with_limit`] says otherwise.
const DEFAULT_LIMIT: usize = 3_000;

/// A folder on this host whose `MEMORY.md` an agent keeps short notes in, with the `memory`
/// tool, and that its system message shows from the next session on.
///
/// The notes are entries, paragraphs apart. The folder, and those on the way to it, are made
/// when the agent is, and the file on the first entry; until then the memory is empty.
///
/// ```rust,no_run
/// # use ailoy::memory::{MemoryEdit, Memory};
/// # fn f() -> anyhow::Result<()> {
/// let memory = Memory::new("/data/agent1").prepared()?;
/// memory.edit(&[MemoryEdit::Add { content: "Name is Minsu".into() }])?;
/// assert_eq!(memory.read()?, ["Name is Minsu"]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    dir: PathBuf,
    limit: usize,
}

impl Memory {
    /// The memory kept in the folder `dir`, relative to the current directory unless absolute,
    /// holding at most 3,000 characters.
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

    /// The folder, as given or, once [`prepared`](Self::prepared), absolute.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The file in the folder that holds the entries, [`FILE_NAME`].
    pub fn file(&self) -> PathBuf {
        self.dir.join(FILE_NAME)
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    /// This memory with its folder absolute and made, with those on the way to it, so that a
    /// later change of the current directory still finds it and a folder that cannot be made
    /// shows now rather than on the first entry.
    pub fn prepared(self) -> io::Result<Self> {
        let dir = std::path::absolute(&self.dir)?;
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, ..self })
    }

    /// The entries as the file holds them now; none for a file not written yet.
    pub fn read(&self) -> io::Result<Vec<String>> {
        match std::fs::read_to_string(self.file()) {
            Ok(text) => Ok(parse_entries(&text)),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Make `edits` in order to the file as it is now, all or none, and answer with the entries
    /// after them.
    ///
    /// The limit is checked on the result, so one batch can make room and fill it, and only when
    /// the file grew: one already over its limit (edited by hand) can still shrink. The file is
    /// written to a temporary one beside it and moved into place, so a reader never sees half
    /// of it.
    pub fn edit(&self, edits: &[MemoryEdit]) -> Result<Vec<String>, EditError> {
        let current = self.read()?;
        let edited = edited(&current, edits, self.limit)?;
        if edited != current {
            let mut text = edited.join(SEPARATOR);
            if !text.is_empty() {
                text.push('\n');
            }
            let temp = self.dir.join(format!(".{FILE_NAME}.tmp"));
            std::fs::write(&temp, text)?;
            std::fs::rename(&temp, self.file())?;
        }
        Ok(edited)
    }

    /// The memory's section of the system message.
    pub fn render(&self, entries: &[String]) -> String {
        format!(
            "# Memory\n\n\
             Your memory of this user, kept with the `memory` tool across conversations. It was \
             loaded when this conversation started; your changes show up next time.\n\n\
             [{}/{} chars]\n{}",
            used(entries),
            self.limit,
            if entries.is_empty() {
                "(empty)".to_string()
            } else {
                entries.join(SEPARATOR)
            }
        )
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

/// One change to the memory, as the `memory` tool asks for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryEdit {
    Add {
        content: String,
    },
    /// The entry containing `old_text` becomes `content`.
    Replace {
        old_text: String,
        content: String,
    },
    /// The entry containing `old_text` goes.
    Remove {
        old_text: String,
    },
}

/// Why [`Memory::edit`] changed nothing.
#[derive(Debug, thiserror::Error)]
pub enum EditError {
    /// The edits cannot be made as asked; the file is as it was.
    #[error(transparent)]
    Refused(#[from] EditRefusal),

    /// The file could not be read or written.
    #[error("the memory file could not be read or written: {0}")]
    Io(#[from] io::Error),
}

/// Why the edits were refused: what the model asked for cannot be made.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct EditRefusal {
    pub reason: String,
    /// The entries as they stand, so the request can be made again from them.
    pub entries: Vec<String>,
}

/// Characters in the file, as its limit counts them.
pub(crate) fn used(entries: &[String]) -> usize {
    entries.join(SEPARATOR).chars().count()
}

/// `current` with `edits` made in order, or the reason none can be.
fn edited(
    current: &[String],
    edits: &[MemoryEdit],
    limit: usize,
) -> Result<Vec<String>, EditRefusal> {
    let refuse = |reason: String| EditRefusal {
        reason,
        entries: current.to_vec(),
    };
    let mut entries = current.to_vec();
    for edit in edits {
        match edit {
            MemoryEdit::Add { content } => {
                let content = entry(content).map_err(refuse)?;
                if !entries.iter().any(|e| squash(e) == squash(&content)) {
                    entries.push(content);
                }
            }
            MemoryEdit::Replace { old_text, content } => {
                let content = entry(content).map_err(refuse)?;
                let hits = find(&entries, old_text).map_err(refuse)?;
                remove_all(&mut entries, &hits[1..]);
                let elsewhere = (0..entries.len())
                    .any(|i| i != hits[0] && squash(&entries[i]) == squash(&content));
                if elsewhere {
                    entries.remove(hits[0]);
                } else {
                    entries[hits[0]] = content;
                }
            }
            MemoryEdit::Remove { old_text } => {
                let hits = find(&entries, old_text).map_err(refuse)?;
                remove_all(&mut entries, &hits);
            }
        }
    }
    let size = used(&entries);
    if size > limit && size > used(current) {
        return Err(refuse(format!(
            "the memory would be {size} characters, over its limit of {limit}; shorten \
             entries in the same call, or remove one if that is not enough"
        )));
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

/// `content` as one entry: trimmed, with blank lines dropped so it reads back as one
/// paragraph; refused if nothing is left.
fn entry(content: &str) -> Result<String, String> {
    let content = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let content = content.trim();
    if content.is_empty() {
        return Err("an entry cannot be empty".into());
    }
    Ok(content.to_string())
}

/// The entry containing `old_text`, whitespace runs counting as one space: its index, or the
/// indices of every copy when the matches are the same entry repeated.
fn find(entries: &[String], old_text: &str) -> Result<Vec<usize>, String> {
    let needle = squash(old_text);
    if needle.is_empty() {
        return Err("old_text cannot be empty".into());
    }
    let hits: Vec<usize> = (0..entries.len())
        .filter(|&i| squash(&entries[i]).contains(&needle))
        .collect();
    match hits.as_slice() {
        [] => Err(format!("no entry contains {old_text:?}")),
        [first, rest @ ..]
            if rest
                .iter()
                .all(|&i| squash(&entries[i]) == squash(&entries[*first])) =>
        {
            Ok(hits)
        }
        _ => Err(format!(
            "{} entries contain {old_text:?}; give a longer old_text that only one contains",
            hits.len()
        )),
    }
}

/// Remove the entries at `at`, which is ascending.
fn remove_all(entries: &mut Vec<String>, at: &[usize]) {
    for &i in at.iter().rev() {
        entries.remove(i);
    }
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(content: &str) -> MemoryEdit {
        MemoryEdit::Add {
            content: content.into(),
        }
    }

    fn replace(old_text: &str, content: &str) -> MemoryEdit {
        MemoryEdit::Replace {
            old_text: old_text.into(),
            content: content.into(),
        }
    }

    fn remove(old_text: &str) -> MemoryEdit {
        MemoryEdit::Remove {
            old_text: old_text.into(),
        }
    }

    fn entries(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|s| s.to_string()).collect()
    }

    /// A memory in a fresh folder of its own.
    fn fresh() -> Memory {
        Memory::new(tempfile::tempdir().unwrap().keep())
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
    fn test_edits_add_replace_and_remove_by_old_text() {
        let start = entries(&["Name is Minsu", "Drinks tea"]);
        // Whitespace-lenient, not case-blind.
        assert!(edited(&start, &[replace("drinks   TEA", "x")], DEFAULT_LIMIT).is_err());

        let after = edited(
            &start,
            &[
                replace("Drinks\n tea", "Drinks only oat-milk lattes"),
                remove("Minsu"),
                add("Prefers short answers\n\n  \nin Korean"),
            ],
            DEFAULT_LIMIT,
        )
        .unwrap();
        assert_eq!(
            after,
            [
                "Drinks only oat-milk lattes",
                "Prefers short answers\nin Korean"
            ]
        );
    }

    /// An edit that cannot be made changes nothing, and the refusal shows the entries as they
    /// stand.
    #[test]
    fn test_an_edit_that_cannot_be_made_changes_nothing() {
        let start = entries(&["Lives in Seoul", "Works in Seoul"]);
        let e = edited(&start, &[add("new"), remove("Seoul")], DEFAULT_LIMIT).unwrap_err();
        assert!(e.reason.contains("2 entries"), "{e}");
        assert_eq!(e.entries, start);

        for bad in [add("  "), add("\n \n"), remove(" "), remove("Busan")] {
            assert!(
                edited(&start, std::slice::from_ref(&bad), DEFAULT_LIMIT).is_err(),
                "{bad:?}"
            );
        }
    }

    /// The same entry repeated, spacing aside, is one entry: every copy goes, and adding it
    /// again adds nothing.
    #[test]
    fn test_repeated_entries_count_as_one() {
        let twice = entries(&["Drinks tea", "Drinks  tea", "Lives in Seoul"]);
        assert_eq!(
            edited(&twice, &[remove("tea")], DEFAULT_LIMIT).unwrap(),
            ["Lives in Seoul"]
        );
        let once = entries(&["Drinks tea"]);
        assert_eq!(
            edited(&once, &[add("Drinks  tea ")], DEFAULT_LIMIT).unwrap(),
            once
        );
    }

    #[test]
    fn test_the_limit_applies_to_the_result_and_only_to_growth() {
        let start = entries(&[&"a".repeat(95)]);
        let e = edited(&start, &[add("bbbbb")], 100).unwrap_err();
        assert!(e.reason.contains("over its limit of 100"), "{e}");
        assert_eq!(
            edited(&start, &[remove("aaa"), add("bbbbb")], 100).unwrap(),
            ["bbbbb"]
        );

        // Over its limit already (edited by hand): may shrink, may not grow.
        let over = entries(&[&"a".repeat(80), &"b".repeat(80)]);
        assert!(edited(&over, &[replace("aaa", "a")], 100).is_ok());
        assert!(edited(&over, &[add("c")], 100).is_err());
    }

    #[test]
    fn test_render_shows_the_entries_and_how_full_the_file_is() {
        let memory = Memory::new("/m").with_limit(500);
        let text = memory.render(&entries(&["Name is Minsu", "Drinks tea"]));
        assert!(text.starts_with("# Memory\n"), "{text}");
        assert!(text.contains("`memory` tool"), "{text}");
        assert!(
            text.ends_with("[25/500 chars]\nName is Minsu\n\nDrinks tea"),
            "{text}"
        );
        assert!(memory.render(&[]).ends_with("[0/500 chars]\n(empty)"));
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

    /// A file not written yet reads as no entries; the first edit makes it, and it reads back.
    #[test]
    fn test_edits_are_written_and_read_back() {
        let memory = fresh();
        assert!(memory.read().unwrap().is_empty());
        assert!(!memory.file().exists());

        memory
            .edit(&[add("- Name is Minsu"), add("Drinks tea")])
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(memory.file()).unwrap(),
            "- Name is Minsu\n\nDrinks tea\n"
        );
        memory.edit(&[remove("tea")]).unwrap();
        assert_eq!(memory.read().unwrap(), ["- Name is Minsu"]);

        let e = memory.edit(&[remove("coffee")]).unwrap_err();
        assert!(matches!(e, EditError::Refused(_)), "{e}");
        assert_eq!(memory.read().unwrap(), ["- Name is Minsu"]);
    }
}
