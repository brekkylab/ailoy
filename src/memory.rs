//! What an agent remembers of a user, kept in one folder in the console.
//!
//! ```text
//! <dir>/mem.sqlite   the history: what was said in past conversations, line by line (`mem`)
//! <dir>/USER.md      profile: who the user is
//! <dir>/NOTES.md    profile: the agent's own notes
//! ```
//!
//! Everything runs as a command on the console (`mem` must be on its `PATH`), so the folder
//! can be anywhere a command reaches, mounted or not.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::console::ConsoleClient;

/// Between two entries of a profile file, and of the profile as the prompt shows it: a blank
/// line, so entries read as paragraphs and never hold one.
const SEPARATOR: &str = "\n\n";

/// A message longer than this goes into the history as several lines, since a search answers
/// with whole lines.
pub const MAX_LINE_CHARS: usize = 1_500;

/// Checks the folder in `$1` exists, exiting [`NO_FOLDER`] if not, and leaves it in `$d` with
/// the script's own arguments from `$1` on.
const PRELUDE: &str = "d=$1; shift; [ -d \"$d\" ] || exit 3\n";
const NO_FOLDER: i32 = 3;

/// `mem`'s exit status for arguments it does not know: a `mem` older than what was asked.
const MEM_USAGE: i32 = 2;

/// Who said a line of the history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Speaker {
    User,
    Assistant,
}

impl Speaker {
    pub fn as_str(self) -> &'static str {
        match self {
            Speaker::User => "user",
            Speaker::Assistant => "assistant",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        [Speaker::User, Speaker::Assistant]
            .into_iter()
            .find(|speaker| speaker.as_str() == s)
    }
}

/// One line of the history, as `mem --json` answers with it.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct Record {
    pub id: String,
    pub text: String,
    /// UTC, `YYYY-MM-DDTHH:MM:SSZ`.
    pub written_at: String,
    /// `None` for a line stored without one, as every line is by a `mem` without `--role`.
    #[serde(default)]
    pub role: Option<Speaker>,
    /// The conversation the line was said in; `None` for a line stored without one.
    #[serde(default)]
    pub conversation: Option<String>,
}

/// One of the two profile files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileTarget {
    /// `USER.md`: who the user is.
    User,
    /// `NOTES.md`: the agent's own notes.
    Notes,
}

impl ProfileTarget {
    pub const ALL: [ProfileTarget; 2] = [ProfileTarget::User, ProfileTarget::Notes];

    pub fn file(self) -> &'static str {
        match self {
            ProfileTarget::User => "USER.md",
            ProfileTarget::Notes => "NOTES.md",
        }
    }

    /// The name the `profile` tool takes.
    pub fn as_str(self) -> &'static str {
        match self {
            ProfileTarget::User => "user",
            ProfileTarget::Notes => "notes",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|target| target.as_str() == s)
    }

    /// The most characters the file may hold unless [`Memory::with_limits`] says otherwise.
    pub fn default_limit(self) -> usize {
        match self {
            ProfileTarget::User => 2_000,
            ProfileTarget::Notes => 3_000,
        }
    }
}

/// The two profile files, each as its entries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Profile {
    pub user: Vec<String>,
    pub notes: Vec<String>,
}

impl Profile {
    pub fn entries(&self, target: ProfileTarget) -> &[String] {
        match target {
            ProfileTarget::User => &self.user,
            ProfileTarget::Notes => &self.notes,
        }
    }

    fn entries_mut(&mut self, target: ProfileTarget) -> &mut Vec<String> {
        match target {
            ProfileTarget::User => &mut self.user,
            ProfileTarget::Notes => &mut self.notes,
        }
    }

    /// Characters in `target`'s file, as its limit counts them.
    pub fn used(&self, target: ProfileTarget) -> usize {
        self.entries(target).join(SEPARATOR).chars().count()
    }

    /// This profile with `edits` made in order, or no change at all.
    ///
    /// Limits are checked on the result, so one batch can make room and fill it, and only for
    /// a file that grew: one already over its limit (edited by hand) can still shrink.
    pub fn edited(
        &self,
        edits: &[ProfileEdit],
        limit: impl Fn(ProfileTarget) -> usize,
    ) -> Result<Profile, EditRefusal> {
        let mut profile = self.clone();
        for edit in edits {
            let target = edit.target();
            let refuse = |reason: String| EditRefusal { target, reason };
            let entries = profile.entries_mut(target);
            match edit {
                ProfileEdit::Add { content, .. } => {
                    let content = entry(content).map_err(refuse)?;
                    if !entries.iter().any(|e| squash(e) == squash(&content)) {
                        entries.push(content);
                    }
                }
                ProfileEdit::Replace {
                    old_text, content, ..
                } => {
                    let content = entry(content).map_err(refuse)?;
                    let hits = find(entries, old_text).map_err(refuse)?;
                    remove_all(entries, &hits[1..]);
                    let elsewhere = (0..entries.len())
                        .any(|i| i != hits[0] && squash(&entries[i]) == squash(&content));
                    if elsewhere {
                        entries.remove(hits[0]);
                    } else {
                        entries[hits[0]] = content;
                    }
                }
                ProfileEdit::Remove { old_text, .. } => {
                    let hits = find(entries, old_text).map_err(refuse)?;
                    remove_all(entries, &hits);
                }
            }
        }
        for target in ProfileTarget::ALL {
            let (used, limit) = (profile.used(target), limit(target));
            if used > limit && used > self.used(target) {
                return Err(EditRefusal {
                    target,
                    reason: format!(
                        "the {} profile would be {used} characters, over its limit of {limit}; \
                         remove or shorten entries in the same call",
                        target.as_str()
                    ),
                });
            }
        }
        Ok(profile)
    }
}

/// One change to the profile, as the `profile` tool asks for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileEdit {
    Add {
        target: ProfileTarget,
        content: String,
    },
    /// The entry containing `old_text` becomes `content`.
    Replace {
        target: ProfileTarget,
        old_text: String,
        content: String,
    },
    /// The entry containing `old_text` goes.
    Remove {
        target: ProfileTarget,
        old_text: String,
    },
}

impl ProfileEdit {
    pub fn target(&self) -> ProfileTarget {
        match self {
            ProfileEdit::Add { target, .. }
            | ProfileEdit::Replace { target, .. }
            | ProfileEdit::Remove { target, .. } => *target,
        }
    }
}

/// Why [`Profile::edited`] changed nothing.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{reason}")]
pub struct EditRefusal {
    /// The file the refused edit was for.
    pub target: ProfileTarget,
    pub reason: String,
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("the memory folder {0} does not exist")]
    NoFolder(String),

    /// The console's `mem` is older than something asked of it.
    #[error("the `mem` in this console is too old for {0}")]
    Unsupported(&'static str),

    /// No line in the history has this id.
    #[error("no entry has the id {0}")]
    UnknownId(String),

    /// What the failing command said.
    #[error("{0}")]
    Failed(String),
}

/// What one agent remembers: a folder in the console, with or without a profile.
///
/// The folder must exist; it is never created, so a mistyped path cannot become an empty
/// memory. The files in it are made on first write. It should usually be under a mount, since
/// what a session writes elsewhere is gone when it ends, and written by one agent at a time,
/// since the history is read back in the order it was written.
///
/// ```rust,no_run
/// # use ailoy::memory::{Memory, Speaker};
/// # async fn f(console: &mut ailoy::console::ConsoleClient) -> anyhow::Result<()> {
/// let memory = Memory::new("/work/memory");
/// memory.append(console, "conversation-1", [(Speaker::User, "I switched to oat milk")]).await?;
/// let found = memory.search(console, "oat milk", 10, None, None).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    dir: String,
    profile: bool,
    limits: [usize; 2],
}

impl Memory {
    /// The memory in the folder `dir`: the history only, without a profile.
    pub fn new(dir: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            profile: false,
            limits: ProfileTarget::ALL.map(ProfileTarget::default_limit),
        }
    }

    /// Whether to keep `USER.md` and `NOTES.md` (and give the `profile` tool) besides the
    /// history. Off by default.
    pub fn with_profile(mut self, profile: bool) -> Self {
        self.profile = profile;
        self
    }

    /// How many characters `USER.md` and `NOTES.md` may each hold.
    pub fn with_limits(mut self, user: usize, notes: usize) -> Self {
        self.limits = [user, notes];
        self
    }

    pub fn limit(&self, target: ProfileTarget) -> usize {
        self.limits[target as usize]
    }

    pub fn dir(&self) -> &str {
        &self.dir
    }

    /// The history store's path.
    pub fn store(&self) -> String {
        format!("{}/mem.sqlite", self.dir.trim_end_matches('/'))
    }

    pub fn has_profile(&self) -> bool {
        self.profile
    }

    /// The history lines with any of `query`'s words, best first; only `role`'s when given, and
    /// none of `exclude_conversation`'s. A folder without a store yet has none.
    pub async fn search(
        &self,
        console: &mut ConsoleClient,
        query: &str,
        limit: usize,
        role: Option<Speaker>,
        exclude_conversation: Option<&str>,
    ) -> Result<Vec<Record>, MemoryError> {
        let mut args = vec!["-n".to_string(), limit.to_string(), "--json".to_string()];
        if let Some(role) = role {
            args.extend(["--role".to_string(), role.as_str().to_string()]);
        }
        if let Some(conversation) = exclude_conversation {
            args.extend([
                "--exclude-conversation".to_string(),
                conversation.to_string(),
            ]);
        }
        // `--` so a query starting with `-` is not read as an option.
        args.extend(["--".to_string(), query.to_string()]);
        let script = r#"[ -e "$d/mem.sqlite" ] || exit 0
exec mem search "$d/mem.sqlite" "$@""#;
        let out = self.sh(console, script, &args).await?;
        let newer = (role.is_some() || exclude_conversation.is_some())
            .then_some("searching by role or conversation");
        records(&succeeded(out, newer)?)
    }

    /// The line `id` with up to `n` lines on each side, in the order written.
    pub async fn around(
        &self,
        console: &mut ConsoleClient,
        id: &str,
        n: usize,
    ) -> Result<Vec<Record>, MemoryError> {
        let args = ["-n", &n.to_string(), "--json", "--", id].map(str::to_string);
        let script = r#"exec mem around "$d/mem.sqlite" "$@""#;
        let out = self.sh(console, script, &args).await?;
        // `mem` refuses an id it does not have with the same exit as any failure, so by its words.
        if out.code != 0 && out.stderr.contains("no memory with id") {
            return Err(MemoryError::UnknownId(id.to_string()));
        }
        records(&succeeded(out, Some("reading around a line"))?)
    }

    /// Add what was said in `conversation` to the end of the history, each line with who said
    /// it, making the store on first use. Returns how many lines were written.
    ///
    /// On a `mem` without `--role` nothing is written, rather than lines without a speaker.
    pub async fn append(
        &self,
        console: &mut ConsoleClient,
        conversation: &str,
        said: impl IntoIterator<Item = (Speaker, impl AsRef<str>)>,
    ) -> Result<usize, MemoryError> {
        let mut args = vec![conversation.to_string()];
        for (speaker, text) in said {
            for line in split_line(text.as_ref(), MAX_LINE_CHARS) {
                args.extend([speaker.as_str().to_string(), line]);
            }
        }
        if args.len() == 1 {
            return Ok(0);
        }
        // `$1` is the conversation, then role and text pairs. One `mem insert` per line, since
        // each has its own role; the first failure stops it.
        let script = r#"s="$d/mem.sqlite" c=$1
shift
[ -e "$s" ] || mem init "$s" >/dev/null || exit
while [ $# -gt 0 ]; do
  mem insert "$s" --role "$1" --conversation "$c" -- "$2" >/dev/null || exit
  shift 2
done"#;
        let out = self.sh(console, script, &args).await?;
        succeeded(
            out,
            Some("keeping who said a line and in which conversation"),
        )?;
        Ok(args.len() / 2)
    }

    /// Both profile files as they are now; a file not written yet has no entries.
    pub async fn read_profile(&self, console: &mut ConsoleClient) -> Result<Profile, MemoryError> {
        let mut profile = Profile::default();
        for target in ProfileTarget::ALL {
            let script = r#"[ ! -f "$d/$1" ] || cat "$d/$1""#;
            let out = self.sh(console, script, [target.file()]).await?;
            *profile.entries_mut(target) = parse_entries(&succeeded(out, None)?);
        }
        Ok(profile)
    }

    /// Write the files where `edited` differs from `current`.
    ///
    /// All files are written to temporary ones before any is moved into place, so a failed
    /// write changes nothing and a reader never sees half a file.
    pub async fn write_profile(
        &self,
        console: &mut ConsoleClient,
        current: &Profile,
        edited: &Profile,
    ) -> Result<(), MemoryError> {
        let mut args = Vec::new();
        for target in ProfileTarget::ALL {
            if edited.entries(target) != current.entries(target) {
                let mut text = edited.entries(target).join(SEPARATOR);
                if !text.is_empty() {
                    text.push('\n');
                }
                args.extend([target.file().to_string(), text]);
            }
        }
        if args.is_empty() {
            return Ok(());
        }
        let script = r#"clean() { rm -f "$d/.USER.md.tmp" "$d/.NOTES.md.tmp"; }
clean
while [ $# -gt 0 ]; do
  printf '%s' "$2" > "$d/.$1.tmp" || { clean; exit 1; }
  shift 2
done
for f in USER.md NOTES.md; do
  [ ! -f "$d/.$f.tmp" ] || mv "$d/.$f.tmp" "$d/$f" || exit
done"#;
        let out = self.sh(console, script, &args).await?;
        succeeded(out, None).map(|_| ())
    }

    /// The profile's section of the system message. How to search the history is in the
    /// `search_history` tool's description, so a memory without a profile adds no section.
    pub fn render(&self, profile: &Profile) -> String {
        let mut text = String::from(
            "# Memory\n\n\
             Your profile of this user, kept with the `profile` tool across conversations. It is \
             a lossy summary loaded when this conversation started; your changes show up next \
             time. Check exact numbers, dates and changed facts with `search_history` even if \
             the profile mentions them.",
        );
        for (target, heading) in [
            (ProfileTarget::User, "About the user"),
            (ProfileTarget::Notes, "Your notes"),
        ] {
            let entries = profile.entries(target);
            text.push_str(&format!(
                "\n\n## {heading} [{}/{} chars]\n",
                profile.used(target),
                self.limit(target),
            ));
            text.push_str(&if entries.is_empty() {
                "(empty)".to_string()
            } else {
                entries.join(SEPARATOR)
            });
        }
        text
    }

    /// Run `script` after [`PRELUDE`] with `args` as its `$1`, `$2`, …: passed as argv, so
    /// nothing needs quoting.
    async fn sh(
        &self,
        console: &mut ConsoleClient,
        script: &str,
        args: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> Result<Output, MemoryError> {
        let mut argv = vec![
            "sh".to_string(),
            "-c".to_string(),
            format!("{PRELUDE}{script}"),
            "sh".to_string(),
            self.dir.clone(),
        ];
        argv.extend(args.into_iter().map(|a| a.as_ref().to_string()));

        // No timeout: only a large store is slow, and the caller still wants the answer.
        let result = console.exec(&argv, None).await.map_err(|e| {
            MemoryError::Failed(format!("the console could not run a memory command: {e}"))
        })?;
        if result.code == NO_FOLDER {
            return Err(MemoryError::NoFolder(self.dir.clone()));
        }
        Ok(Output {
            code: result.code,
            stdout: String::from_utf8_lossy(&result.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&result.stderr).trim().to_string(),
        })
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

/// Converted with [`Path::to_string_lossy`] since argv is `String`; a non-UTF-8 path names a
/// folder the first command will report missing. Applies to `&Path` too.
impl From<PathBuf> for Memory {
    fn from(dir: PathBuf) -> Self {
        Self::new(dir.to_string_lossy().into_owned())
    }
}

impl From<&Path> for Memory {
    fn from(dir: &Path) -> Self {
        Self::new(dir.to_string_lossy().into_owned())
    }
}

struct Output {
    code: i32,
    stdout: String,
    stderr: String,
}

/// `out`'s stdout if it exited 0. A usage error means the `mem` lacks `newer`, when the
/// command asked for something only a newer one has.
fn succeeded(out: Output, newer: Option<&'static str>) -> Result<String, MemoryError> {
    match (out.code, newer) {
        (0, _) => Ok(out.stdout),
        (MEM_USAGE, Some(what)) => Err(MemoryError::Unsupported(what)),
        (code, _) if out.stderr.is_empty() => Err(MemoryError::Failed(format!(
            "the memory command failed (exit {code})"
        ))),
        _ => Err(MemoryError::Failed(out.stderr)),
    }
}

/// `mem --json` output: one record per line.
fn records(out: &str) -> Result<Vec<Record>, MemoryError> {
    out.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .map_err(|e| MemoryError::Failed(format!("`mem` answered with {line:?}: {e}")))
        })
        .collect()
}

/// A profile file's entries: its paragraphs, trimmed. A line of only spaces separates too,
/// since a person may edit the file.
pub fn parse_entries(text: &str) -> Vec<String> {
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

/// `text` trimmed, in pieces of at most `max` characters: whole paragraphs where they fit,
/// then whole lines, then words, and only a word longer than `max` is cut.
pub fn split_line(text: &str, max: usize) -> Vec<String> {
    let mut pieces = Vec::new();
    let text = text.trim();
    if !text.is_empty() {
        pack(text, max, &["\n\n", "\n", " "], &mut pieces);
    }
    pieces
}

/// Pack `text`'s parts between `seps[0]` into pieces of at most `max` characters, splitting a
/// part still too long at the next separator.
fn pack(text: &str, max: usize, seps: &[&str], pieces: &mut Vec<String>) {
    let len = |s: &str| s.chars().count();
    if len(text) <= max {
        pieces.push(text.to_string());
        return;
    }
    let Some((sep, rest)) = seps.split_first() else {
        let chars: Vec<char> = text.chars().collect();
        pieces.extend(chars.chunks(max).map(|c| c.iter().collect::<String>()));
        return;
    };
    // Words split at any whitespace and join back with a space.
    let parts: Vec<&str> = if *sep == " " {
        text.split_whitespace().collect()
    } else {
        text.split(sep)
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .collect()
    };
    let mut current = String::new();
    for part in parts {
        if !current.is_empty() && len(&current) + len(sep) + len(part) <= max {
            current.push_str(sep);
            current.push_str(part);
            continue;
        }
        if !current.is_empty() {
            pieces.push(std::mem::take(&mut current));
        }
        if len(part) <= max {
            current = part.to_string();
        } else {
            pack(part, max, rest, pieces);
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_console;

    use ProfileTarget::{Notes, User};

    fn add(target: ProfileTarget, content: &str) -> ProfileEdit {
        ProfileEdit::Add {
            target,
            content: content.into(),
        }
    }

    fn replace(target: ProfileTarget, old_text: &str, content: &str) -> ProfileEdit {
        ProfileEdit::Replace {
            target,
            old_text: old_text.into(),
            content: content.into(),
        }
    }

    fn remove(target: ProfileTarget, old_text: &str) -> ProfileEdit {
        ProfileEdit::Remove {
            target,
            old_text: old_text.into(),
        }
    }

    /// `profile` with `edits` made under the default limits.
    fn edit(profile: &Profile, edits: &[ProfileEdit]) -> Result<Profile, EditRefusal> {
        profile.edited(edits, ProfileTarget::default_limit)
    }

    fn profile(user: &[&str]) -> Profile {
        Profile {
            user: user.iter().map(|s| s.to_string()).collect(),
            notes: Vec::new(),
        }
    }

    #[test]
    fn test_entries_round_trip_through_the_file_format() {
        let entries = vec!["Name is Minsu".to_string(), "Two\nlines".to_string()];
        let file = entries.join(SEPARATOR) + "\n";
        assert_eq!(parse_entries(&file), entries);
        assert_eq!(parse_entries("a\n  \nb\n\n\n\nc\n"), ["a", "b", "c"]);
        assert!(parse_entries("").is_empty());
    }

    /// A blank line inside an entry would split it in the file, so it is dropped.
    #[test]
    fn test_an_entry_keeps_no_blank_line() {
        let edited = edit(
            &profile(&[]),
            &[add(User, "Name is Minsu\n\n  \nLives in Seoul")],
        )
        .unwrap();
        assert_eq!(edited.user, ["Name is Minsu\nLives in Seoul"]);
        let file = edited.user.join(SEPARATOR);
        assert_eq!(parse_entries(&file), edited.user);
    }

    #[test]
    fn test_edits_add_replace_and_remove_by_old_text() {
        let start = profile(&["Name is Minsu", "Drinks tea"]);
        // Whitespace-lenient, not case-blind.
        let e = edit(&start, &[replace(User, "drinks   TEA", "unused")]).unwrap_err();
        assert_eq!(e.target, User);

        let edited = edit(
            &start,
            &[
                replace(User, "Drinks\n tea", "Drinks only oat-milk lattes"),
                remove(User, "Minsu"),
                add(Notes, "Prefers short answers"),
            ],
        )
        .unwrap();
        assert_eq!(edited.user, ["Drinks only oat-milk lattes"]);
        assert_eq!(edited.notes, ["Prefers short answers"]);
    }

    #[test]
    fn test_an_edit_that_cannot_be_made_changes_nothing() {
        let start = profile(&["Lives in Seoul", "Works in Seoul"]);
        let e = edit(&start, &[add(User, "new"), remove(User, "Seoul")]).unwrap_err();
        assert!(e.reason.contains("2 entries"), "{e}");

        for bad in [add(User, "  "), add(User, "\n \n"), remove(User, " ")] {
            assert!(edit(&start, std::slice::from_ref(&bad)).is_err(), "{bad:?}");
        }
    }

    /// The same entry repeated, spacing aside, is one entry: every copy goes, and adding it
    /// again adds nothing.
    #[test]
    fn test_repeated_entries_count_as_one() {
        let twice = profile(&["Drinks tea", "Drinks  tea", "Lives in Seoul"]);
        assert_eq!(
            edit(&twice, &[remove(User, "tea")]).unwrap().user,
            ["Lives in Seoul"]
        );
        assert_eq!(
            edit(&twice, &[replace(User, "tea", "Lives in Seoul")])
                .unwrap()
                .user,
            ["Lives in Seoul"]
        );
        let once = profile(&["Drinks tea"]);
        assert_eq!(edit(&once, &[add(User, "Drinks  tea ")]).unwrap(), once);
    }

    #[test]
    fn test_limits_apply_to_the_result_and_only_to_growth() {
        let start = profile(&[&"a".repeat(1_995)]);
        let e = edit(&start, &[add(User, "bbbbb")]).unwrap_err();
        assert!(e.reason.contains("over its limit of 2000"), "{e}");
        let edited = edit(&start, &[remove(User, "aaa"), add(User, "bbbbb")]).unwrap();
        assert_eq!(edited.user, ["bbbbb"]);

        // Over its limit already: may shrink, may not grow, and does not block the other file.
        let over = profile(&[&"a".repeat(1_100), &"b".repeat(1_100)]);
        assert!(edit(&over, &[add(Notes, "note")]).is_ok());
        assert!(edit(&over, &[replace(User, "aaa", "a")]).is_ok());
        assert!(edit(&over, &[add(User, "c")]).is_err());
    }

    #[test]
    fn test_render_shows_the_profile_as_the_tool_spells_it() {
        let text = Memory::new("/m")
            .with_profile(true)
            .render(&profile(&["Name is Minsu", "Drinks tea"]));
        assert!(text.starts_with("# Memory\n"), "{text}");
        assert!(text.contains("`search_history`") && text.contains("`profile` tool"));
        assert!(
            text.contains("## About the user [25/2000 chars]\nName is Minsu\n\nDrinks tea"),
            "{text}"
        );
        assert!(text.ends_with("## Your notes [0/3000 chars]\n(empty)"));

        let small = Memory::new("/m").with_profile(true).with_limits(500, 800);
        let text = small.render(&Profile::default());
        assert!(
            text.contains("[0/500 chars]") && text.contains("[0/800 chars]"),
            "{text}"
        );
        let e = profile(&[])
            .edited(&[add(User, &"a".repeat(501))], |t| small.limit(t))
            .unwrap_err();
        assert!(e.reason.contains("limit of 500"), "{e}");
    }

    #[test]
    fn test_split_line_keeps_paragraphs_whole_where_they_fit() {
        assert!(split_line(" \n ", 10).is_empty());
        assert_eq!(split_line(" short ", 10), ["short"]);
        assert_eq!(
            split_line("aaaa\n\nbbbb\n\ncccc", 10),
            ["aaaa\n\nbbbb", "cccc"]
        );
        assert_eq!(split_line("aa bb cc dd\nee", 5), ["aa bb", "cc dd", "ee"]);
        assert_eq!(split_line("가나\t다라 마바", 5), ["가나 다라", "마바"]);
        assert_eq!(split_line("가나다라마바", 4), ["가나다라", "마바"]);
        let long = "word ".repeat(1_000);
        assert!(
            split_line(&long, MAX_LINE_CHARS)
                .iter()
                .all(|p| p.chars().count() <= MAX_LINE_CHARS)
        );
    }

    /// A fresh folder on the test console's mount, kept past this function.
    fn folder() -> Memory {
        let dir = tempfile::tempdir().unwrap().keep();
        Memory::new(dir.to_string_lossy().into_owned())
    }

    async fn has_role(console: &mut ConsoleClient) -> bool {
        let help = console
            .exec(["mem", "insert", "--help"], None)
            .await
            .unwrap();
        String::from_utf8_lossy(&help.stdout).contains("--role")
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_profile_is_written_and_read_back_through_the_console() {
        let mut console = test_console().await;
        let memory = folder();

        let current = memory.read_profile(&mut console).await.unwrap();
        assert_eq!(current, Profile::default(), "no files yet");
        let edited = edit(
            &current,
            &[
                add(User, "- Name is Minsu"),
                add(Notes, "it's \"quoted\" $HOME"),
            ],
        )
        .unwrap();
        memory
            .write_profile(&mut console, &current, &edited)
            .await
            .unwrap();
        assert_eq!(memory.read_profile(&mut console).await.unwrap(), edited);

        let file = Path::new(memory.dir()).join("USER.md");
        assert_eq!(std::fs::read_to_string(file).unwrap(), "- Name is Minsu\n");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_a_missing_folder_is_reported_and_not_made() {
        let mut console = test_console().await;
        let dir = tempfile::tempdir().unwrap().keep().join("missing");
        let memory = Memory::new(dir.to_string_lossy().into_owned());

        for e in [
            memory.read_profile(&mut console).await.unwrap_err(),
            memory
                .search(&mut console, "tea", 10, None, None)
                .await
                .unwrap_err(),
            memory
                .append(&mut console, "c1", [(Speaker::User, "tea")])
                .await
                .unwrap_err(),
        ] {
            assert!(matches!(e, MemoryError::NoFolder(_)), "{e}");
        }
        assert!(!dir.exists());
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_a_folder_without_a_store_has_no_history() {
        let mut console = test_console().await;
        let found = folder()
            .search(&mut console, "tea", 10, None, None)
            .await
            .unwrap();
        assert!(found.is_empty());
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_append_keeps_who_said_each_line() {
        let mut console = test_console().await;
        let memory = folder();
        let said = [
            (Speaker::User, "- I switched to oat milk"),
            (Speaker::Assistant, "Noted: oat milk from now on."),
        ];

        if !has_role(&mut console).await {
            let e = memory.append(&mut console, "c1", said).await.unwrap_err();
            assert!(matches!(e, MemoryError::Unsupported(_)), "{e}");
            let found = memory
                .search(&mut console, "oat", 10, None, None)
                .await
                .unwrap();
            assert!(found.is_empty(), "nothing written: {found:?}");
            return;
        }

        assert_eq!(memory.append(&mut console, "c1", said).await.unwrap(), 2);
        let user = memory
            .search(&mut console, "oat", 10, Some(Speaker::User), None)
            .await
            .unwrap();
        assert_eq!(user.len(), 1);
        assert_eq!(user[0].text, "- I switched to oat milk");
        assert_eq!(user[0].role, Some(Speaker::User));
        assert_eq!(user[0].conversation.as_deref(), Some("c1"));

        // Another conversation's lines are not this one's neighbours, and can be left out.
        memory
            .append(&mut console, "c2", [(Speaker::User, "oat milk again")])
            .await
            .unwrap();
        let others = memory
            .search(&mut console, "oat", 10, None, Some("c1"))
            .await
            .unwrap();
        assert_eq!(others.len(), 1);
        assert_eq!(others[0].conversation.as_deref(), Some("c2"));

        let around = memory.around(&mut console, &user[0].id, 2).await.unwrap();
        assert_eq!(around.len(), 2);
        assert_eq!(around[1].role, Some(Speaker::Assistant));
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_reads_lines_stored_without_a_role() {
        let mut console = test_console().await;
        let memory = folder();
        for argv in [
            vec!["mem", "init", &memory.store()],
            vec!["mem", "insert", &memory.store(), "User drinks tea"],
        ] {
            let out = console.exec(&argv, None).await.unwrap();
            assert_eq!(out.code, 0, "{}", String::from_utf8_lossy(&out.stderr));
        }

        let found = memory
            .search(&mut console, "tea", 10, None, None)
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].text, "User drinks tea");
        assert_eq!(found[0].role, None);
        assert!(
            found[0].written_at.ends_with('Z'),
            "{}",
            found[0].written_at
        );
    }
}
