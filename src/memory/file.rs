use std::{
    io,
    path::{Path, PathBuf},
};

/// Between two entries of the file, and of the memory as the prompt shows it: a blank line, so
/// entries read as paragraphs and never hold one.
const SEPARATOR: &str = "\n\n";

/// The file in a memory's folder that holds its entries.
pub const FILE_NAME: &str = "MEMORY.md";

/// The file in a memory's folder that edits lock, one at a time, so none is lost.
const LOCK_NAME: &str = ".MEMORY.md.lock";

/// How a date leads an entry: `[2026-11-17] `, the host's date when the entry was last written.
const STAMP_LEN: usize = "[YYYY-MM-DD] ".len();

/// The most characters the file may hold unless [`Memory::with_limit`] says otherwise.
const DEFAULT_LIMIT: usize = 3_000;

/// A folder on this host whose `MEMORY.md` an agent keeps short notes in, with the `memory`
/// tool, and that its system message shows from the next session on.
///
/// The notes are entries, paragraphs apart, each led by the host's date when it was last
/// written, as `[2026-11-17] Drinks tea`. The folder, and those on the way to it, are made
/// when the agent is, and the file on the first entry; until then the memory is empty.
///
/// ```rust,no_run
/// # use ailoy::memory::{MemoryEdit, Memory};
/// # fn f() -> anyhow::Result<()> {
/// let memory = Memory::new("/data/agent1").prepared()?;
/// memory.edit(&[MemoryEdit::Add { content: "Name is Minsu".into() }])?;
/// // Led by the day it was written, as `[2026-11-17] Name is Minsu`.
/// assert!(memory.read()?[0].ends_with("] Name is Minsu"));
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
    /// of it. It keeps its permissions, and a `MEMORY.md` that is a symbolic link stays one:
    /// the file it names is the one replaced. A file the first entry makes is the owner's
    /// alone to read.
    ///
    /// Edits wait for each other, from this process or another, on a lock file in the folder:
    /// each reads the file another has just written, so two at once both land. On a file
    /// system that cannot lock a file, as some network ones, an edit goes on without waiting.
    pub fn edit(&self, edits: &[MemoryEdit]) -> Result<Edited, EditError> {
        let lock_path = self.dir.join(LOCK_NAME);
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)?;
        if let Err(e) = lock.lock() {
            log::warn!(
                "could not lock {}: {e}; editing the memory without waiting for other edits",
                lock_path.display()
            );
        }

        let current = self.read()?;
        let edited = edited(&current, edits, self.limit, &today())?;
        if edited.entries != current {
            let mut text = edited.entries.join(SEPARATOR);
            if !text.is_empty() {
                text.push('\n');
            }
            // The file a symbolic link names, so the link is left as it is.
            let target = match std::fs::canonicalize(self.file()) {
                Ok(target) => target,
                Err(e) if e.kind() == io::ErrorKind::NotFound => self.file(),
                Err(e) => return Err(e.into()),
            };
            let beside = target.parent().unwrap_or(&self.dir);
            let mut temp = tempfile::NamedTempFile::new_in(beside)?;
            io::Write::write_all(&mut temp, text.as_bytes())?;
            if let Ok(old) = std::fs::metadata(&target) {
                temp.as_file().set_permissions(old.permissions())?;
            }
            temp.as_file().sync_all()?;
            temp.persist(&target).map_err(|e| e.error)?;
        }
        Ok(edited)
    }

    /// The memory's section of the system message.
    pub fn render(&self, entries: &[String]) -> String {
        format!(
            "# Memory\n\n\
             Your memory of this user, kept with the `memory` tool across conversations, each \
             entry led by the date it was last written. It was loaded when this conversation \
             started; your changes show up next time.\n\n\
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

/// What [`Memory::edit`] made of the edits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edited {
    /// The entries after the edits.
    pub entries: Vec<String>,
    /// What the edits did other than as asked, such as a fact added that was already there.
    pub notes: Vec<String>,
}

/// Why [`Memory::edit`] changed nothing.
#[derive(Debug, thiserror::Error)]
pub enum EditError {
    /// The edits cannot be made as asked; the file is as it was.
    #[error(transparent)]
    Refused(#[from] EditRefusal),

    /// The file could not be read or written.
    #[error("the memory's MEMORY.md could not be read or written: {0}")]
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

/// The host's date, as entries are stamped with it.
pub(crate) fn today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
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

/// Whether two entries say the same thing: their dates and spacing aside.
fn same(a: &str, b: &str) -> bool {
    squash(body(a)) == squash(body(b))
}

/// Characters in the file, as its limit counts them.
pub(crate) fn used(entries: &[String]) -> usize {
    entries.join(SEPARATOR).chars().count()
}

/// `current` with `edits` made in order, or the reason none can be; what is added or replaced
/// is stamped with `today`. A fact added again is kept once, where it was, with `today`'s
/// date, as a replace with the same text would leave it.
fn edited(
    current: &[String],
    edits: &[MemoryEdit],
    limit: usize,
    today: &str,
) -> Result<Edited, EditRefusal> {
    let refuse = |reason: String| EditRefusal {
        reason,
        entries: current.to_vec(),
    };
    let mut entries = current.to_vec();
    let mut notes = Vec::new();
    for edit in edits {
        match edit {
            MemoryEdit::Add { content } => {
                let content = entry(content, today).map_err(refuse)?;
                let copies: Vec<usize> = (0..entries.len())
                    .filter(|&i| same(&entries[i], &content))
                    .collect();
                match copies.split_first() {
                    None => entries.push(content),
                    Some((&first, rest)) => {
                        notes.push(format!(
                            "{:?} was already an entry; it is kept once, now dated {today}",
                            body(&content)
                        ));
                        remove_all(&mut entries, rest);
                        entries[first] = content;
                    }
                }
            }
            MemoryEdit::Replace { old_text, content } => {
                let content = entry(content, today).map_err(refuse)?;
                let hits = find(&entries, old_text).map_err(refuse)?;
                remove_all(&mut entries, &hits[1..]);
                let elsewhere =
                    (0..entries.len()).any(|i| i != hits[0] && same(&entries[i], &content));
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
    Ok(Edited { entries, notes })
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

/// The entry containing `old_text`, whitespace runs counting as one space: its index, or the
/// indices of every copy when the matches are the same entry repeated. Of several entries
/// containing it, one that it is the whole of is the one meant, since no longer `old_text`
/// could pick that one out.
fn find(entries: &[String], old_text: &str) -> Result<Vec<usize>, String> {
    let needle = squash(old_text);
    if needle.is_empty() {
        return Err("old_text cannot be empty".into());
    }
    let one = |hits: &[usize]| match hits {
        [first, rest @ ..] => rest.iter().all(|&i| same(&entries[i], &entries[*first])),
        [] => false,
    };
    let hits: Vec<usize> = (0..entries.len())
        .filter(|&i| squash(&entries[i]).contains(&needle))
        .collect();
    if hits.is_empty() {
        return Err(format!("no entry contains {old_text:?}"));
    }
    if one(&hits) {
        return Ok(hits);
    }
    let whole: Vec<usize> = hits
        .iter()
        .copied()
        .filter(|&i| same(&entries[i], old_text))
        .collect();
    if one(&whole) {
        return Ok(whole);
    }
    Err(format!(
        "{} entries contain {old_text:?}; give a longer old_text that only one contains",
        hits.len()
    ))
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

    const DAY: &str = "2026-11-17";

    /// `text` as an entry written on [`DAY`].
    fn dated(text: &str) -> String {
        format!("[{DAY}] {text}")
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
        assert!(edited(&start, &[replace("drinks   TEA", "x")], DEFAULT_LIMIT, DAY).is_err());

        let after = edited(
            &start,
            &[
                replace("Drinks\n tea", "Drinks only oat-milk lattes"),
                remove("Minsu"),
                add("Prefers short answers\n\n  \nin Korean"),
            ],
            DEFAULT_LIMIT,
            DAY,
        )
        .unwrap()
        .entries;
        assert_eq!(
            after,
            [
                dated("Drinks only oat-milk lattes"),
                dated("Prefers short answers\nin Korean")
            ]
        );
    }

    /// What is added or replaced is led by the day it was written, not by a date the model
    /// copied from an entry; an entry's date leaves it the same fact as before.
    #[test]
    fn test_entries_are_led_by_the_day_they_were_written() {
        let start = entries(&["[2026-11-10] Drinks tea", "Lives in Seoul"]);
        let after = edited(
            &start,
            &[
                add("[2026-11-10] Drinks tea"),
                replace("Seoul", "[2026-11-10] Lives in Busan"),
                add("Married in (2027-03)"),
            ],
            DEFAULT_LIMIT,
            DAY,
        )
        .unwrap()
        .entries;
        assert_eq!(
            after,
            [
                dated("Drinks tea"),
                dated("Lives in Busan"),
                dated("Married in (2027-03)")
            ]
        );
        assert_eq!(body(&dated("Drinks tea")), "Drinks tea");
        assert_eq!(body("[2026-1-10] Drinks tea"), "[2026-1-10] Drinks tea");
        assert_eq!(body("Drinks tea"), "Drinks tea");
    }

    /// An edit that cannot be made changes nothing, and the refusal shows the entries as they
    /// stand.
    #[test]
    fn test_an_edit_that_cannot_be_made_changes_nothing() {
        let start = entries(&["Lives in Seoul", "Works in Seoul"]);
        let e = edited(&start, &[add("new"), remove("Seoul")], DEFAULT_LIMIT, DAY).unwrap_err();
        assert!(e.reason.contains("2 entries"), "{e}");
        assert_eq!(e.entries, start);

        for bad in [add("  "), add("\n \n"), remove(" "), remove("Busan")] {
            assert!(
                edited(&start, std::slice::from_ref(&bad), DEFAULT_LIMIT, DAY).is_err(),
                "{bad:?}"
            );
        }
    }

    /// The same entry repeated, spacing aside, is one entry: every copy goes, and adding it
    /// again keeps it once, where it was, dated the day it was added again, and says so.
    #[test]
    fn test_repeated_entries_count_as_one() {
        let twice = entries(&["Drinks tea", "Drinks  tea", "Lives in Seoul"]);
        assert_eq!(
            edited(&twice, &[remove("tea")], DEFAULT_LIMIT, DAY)
                .unwrap()
                .entries,
            ["Lives in Seoul"]
        );
        let start = entries(&["[2026-11-10] Drinks tea", "Lives in Seoul", "Drinks  tea"]);
        let again = edited(&start, &[add("Drinks tea")], DEFAULT_LIMIT, DAY).unwrap();
        assert_eq!(
            again.entries,
            [dated("Drinks tea"), "Lives in Seoul".into()]
        );
        assert_eq!(again.notes.len(), 1);
        assert!(
            again.notes[0].contains("already an entry"),
            "{:?}",
            again.notes
        );

        let new = edited(&start, &[add("Lives in Busan")], DEFAULT_LIMIT, DAY).unwrap();
        assert!(new.notes.is_empty());
    }

    /// An entry that another one contains whole is still picked out by its own text.
    #[test]
    fn test_an_entry_inside_another_is_picked_out_by_its_whole_text() {
        let start = entries(&[&dated("Likes tea"), &dated("Likes tea with milk")]);
        assert!(edited(&start, &[remove("tea")], DEFAULT_LIMIT, DAY).is_err());
        assert_eq!(
            edited(
                &start,
                &[replace(" Likes  tea", "Likes coffee")],
                DEFAULT_LIMIT,
                DAY
            )
            .unwrap()
            .entries,
            [dated("Likes coffee"), dated("Likes tea with milk")]
        );
        assert_eq!(
            edited(&start, &[remove(&dated("Likes tea"))], DEFAULT_LIMIT, DAY)
                .unwrap()
                .entries,
            [dated("Likes tea with milk")]
        );
    }

    #[test]
    fn test_the_limit_applies_to_the_result_and_only_to_growth() {
        let start = entries(&[&"a".repeat(95)]);
        let e = edited(&start, &[add("bbbbb")], 100, DAY).unwrap_err();
        assert!(e.reason.contains("over its limit of 100"), "{e}");
        assert_eq!(
            edited(&start, &[remove("aaa"), add("bbbbb")], 100, DAY)
                .unwrap()
                .entries,
            [dated("bbbbb")]
        );

        // Over its limit already (edited by hand): may shrink, may not grow.
        let over = entries(&[&"a".repeat(80), &"b".repeat(80)]);
        assert!(edited(&over, &[replace("aaa", "a")], 100, DAY).is_ok());
        assert!(edited(&over, &[add("c")], 100, DAY).is_err());
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
        let day = today();
        assert_eq!(
            std::fs::read_to_string(memory.file()).unwrap(),
            format!("[{day}] - Name is Minsu\n\n[{day}] Drinks tea\n")
        );
        memory.edit(&[remove("tea")]).unwrap();
        let name = [format!("[{day}] - Name is Minsu")];
        assert_eq!(memory.read().unwrap(), name);

        let e = memory.edit(&[remove("coffee")]).unwrap_err();
        assert!(matches!(e, EditError::Refused(_)), "{e}");
        assert_eq!(memory.read().unwrap(), name);
    }

    /// An edit keeps the file's permissions, and writes through a symbolic link rather than
    /// over it.
    #[cfg(unix)]
    #[test]
    fn test_an_edit_keeps_the_files_permissions_and_its_link() {
        use std::os::unix::fs::PermissionsExt as _;

        let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        let memory = fresh();
        memory.edit(&[add("Drinks tea")]).unwrap();
        assert_eq!(mode(&memory.file()), 0o600);
        std::fs::set_permissions(memory.file(), std::fs::Permissions::from_mode(0o644)).unwrap();
        memory.edit(&[add("Lives in Seoul")]).unwrap();
        assert_eq!(mode(&memory.file()), 0o644);

        let elsewhere = tempfile::tempdir().unwrap();
        let real = elsewhere.path().join("notes.md");
        std::fs::write(&real, "Drinks tea\n").unwrap();
        let linked = fresh();
        std::os::unix::fs::symlink(&real, linked.file()).unwrap();
        linked.edit(&[add("Lives in Seoul")]).unwrap();
        assert!(
            std::fs::symlink_metadata(linked.file())
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(
            std::fs::read_to_string(&real)
                .unwrap()
                .contains("Lives in Seoul")
        );
    }

    /// Edits made at once, as a model's parallel tool calls are, wait for each other: every
    /// one lands.
    #[test]
    fn test_edits_at_once_all_land() {
        let memory = fresh();
        let threads: Vec<_> = (0..8)
            .map(|i| {
                let memory = memory.clone();
                std::thread::spawn(move || memory.edit(&[add(&format!("Fact {i}"))]).unwrap())
            })
            .collect();
        for t in threads {
            t.join().unwrap();
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
