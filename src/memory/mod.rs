//! Memories, as `mem` keeps them.
//!
//! A memory store is one file — the same file `mem init` makes — and reading or writing it
//! is a command run on a [`Console`]. That is the whole of what this module is: `mem`
//! already holds the store, so nothing here opens a database, links `rusqlite`, or knows
//! what a row looks like. It spells one command line and reads the lines that come back.
//!
//! Which means the store has to be reachable from the console it is handed: `mem` is a
//! name on the far end's `PATH` — attached by the console server or delegated to it —
//! and the memory file is a path the far end can open, not one this process resolves.
//!
//! Bringing a store into being is not here. `mem init` is a command of its own precisely
//! so a mistyped name cannot become an empty store that answers every search with
//! nothing, and a wrapper that quietly created one would undo that.

use std::path::{Path, PathBuf};

use crate::console::Console;

/// One memory store, named by the file it is.
///
/// A path and nothing else. There is no handle to hold: `mem` opens the file per command
/// and closes it again, so this stays cheap to clone and safe to hand to two callers —
/// what serializes access to the store is the console lock, the same as for any other
/// command an agent runs.
///
/// Which console it runs on is not here either, and is passed per call. An
/// [`Agent`](crate::agent::Agent) holding a memory holds a console slot separately, and
/// binding the two here would be a second answer to which console a tool runs in.
///
/// ```rust,no_run
/// # use ailoy::memory::Memory;
/// # async fn f(console: &mut ailoy::console::Console) -> anyhow::Result<()> {
/// let memory = Memory::new("/work/notes.sqlite");
/// memory.insert(console, ["User switched to oat milk"]).await?;
/// let found = memory.search(console, "what does the user drink?").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    memfile: String,
}

impl Memory {
    /// The store at `memfile`, which `mem init` is expected to have made already.
    ///
    /// Nothing is checked here — a path is a path until a command is run with it, and the
    /// console that would answer whether the file is there is not one of the arguments.
    /// A store that is not there is heard from the first [`search`](Self::search) or
    /// [`insert`](Self::insert).
    pub fn new(memfile: impl Into<String>) -> Self {
        Self {
            memfile: memfile.into(),
        }
    }

    /// The store, spelled as the caller spelled it.
    pub fn memfile(&self) -> &str {
        &self.memfile
    }

    /// The memories nearest `query`, nearest first.
    ///
    /// `mem search <memfile> <query>` on the console, and its stdout split into lines: one
    /// memory per line, as `mem` prints them. Nothing near the query is an empty `Vec`
    /// rather than an error — the store was read and holds nothing near this, which is an
    /// answer.
    ///
    /// The argv goes to [`Console::exec`] as three arguments, so no shell sees them and
    /// neither the path nor the query needs quoting or escaping.
    ///
    /// # Errors
    ///
    /// A store that is not there, a file that is not a store, and a `mem` the console
    /// cannot run are all errors here: `mem` says which on stderr and exits non-zero, and
    /// that sentence is what comes back. So is a console that could not carry the call at
    /// all.
    pub async fn search(
        &self,
        console: &mut Console,
        query: impl AsRef<str>,
    ) -> anyhow::Result<Vec<String>> {
        self.run(console, "search", [query.as_ref()]).await
    }

    /// Write `memories` into the store, and hear back the ones it now holds.
    ///
    /// `mem insert <memfile> <memories>...` — one memory per argument, because argv is
    /// already a list and a separator inside one argument would be a memory nobody could
    /// write. What is handed in is what the store holds, word for word: nothing here
    /// shortens, splits or rephrases it, and nothing reads it. The deciding is the
    /// caller's and has already happened by the time this is called.
    ///
    /// It does not deduplicate, either. Told the same thing twice the store holds it
    /// twice — two callers writing one sentence about different things is a case no
    /// comparison of text could tell from a repeat.
    ///
    /// Nothing to write is a no-op and an empty `Vec`, which is the answer a caller with
    /// an empty list wants rather than a case it has to check for before it can call at
    /// all.
    ///
    /// # Errors
    ///
    /// A store that is not there is one, and nothing is created on the way past — `mem
    /// init` makes a store. So is a memory that is nothing but whitespace: `mem` refuses
    /// the whole line rather than dropping it from a list that would then look complete,
    /// so either every memory here is written or none is.
    pub async fn insert(
        &self,
        console: &mut Console,
        memories: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memories: Vec<String> = memories
            .into_iter()
            .map(|m| m.as_ref().to_string())
            .collect();

        // An empty list still goes to `mem`, which answers it with nothing and a zero.
        // Short-circuiting here would be a second answer to that question, and one that
        // stops saying whether the store was even there.
        self.run(console, "insert", &memories).await
    }

    /// One `mem` command against this store, and the lines it answered with.
    ///
    /// Both commands above are the same shape — a subcommand, the store, then what the
    /// subcommand takes — and both answer in the same one: memories, one per line. So how
    /// a refusal is read and how output becomes memories is written once, here.
    async fn run(
        &self,
        console: &mut Console,
        subcommand: &str,
        rest: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memfile = &self.memfile;

        let mut argv = vec!["mem".to_string(), subcommand.to_string(), memfile.clone()];
        argv.extend(rest.into_iter().map(|arg| arg.as_ref().to_string()));

        // No timeout: these are reads and writes of one local file, and what would make
        // one slow is a store large enough that the caller still wants the answer.
        let result = console
            .exec(&argv, None)
            .await
            .map_err(|e| anyhow::anyhow!("`mem {subcommand} {memfile}` could not be run: {e}"))?;

        if result.code != 0 {
            // `mem` names the store it refused on and why, which is the sentence worth
            // passing on. Its exit code carries nothing this caller can act on beyond the
            // failure itself — the command either happened or it did not.
            let said = String::from_utf8_lossy(&result.stderr);
            let said = said.trim();
            anyhow::bail!(if said.is_empty() {
                format!("`mem {subcommand} {memfile}` failed (exit {})", result.code)
            } else {
                said.to_string()
            });
        }

        // A memory is a line, and a command answering with nothing is no lines at all.
        // Blank lines are dropped rather than returned as empty memories: `mem` refuses
        // to store one, so a blank line here is the trailing newline and not a memory.
        Ok(String::from_utf8_lossy(&result.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.to_string())
            .collect())
    }
}

/// A store named by a `String`, which is the name a `Memory` already is.
impl From<String> for Memory {
    fn from(memfile: String) -> Self {
        Self::new(memfile)
    }
}

/// The same for a borrowed name, since a path written in the call is the common way to
/// name a store.
impl From<&str> for Memory {
    fn from(memfile: &str) -> Self {
        Self::new(memfile)
    }
}

/// A store named by a `PathBuf`, for callers that built the path rather than wrote it.
///
/// The path is spelled with [`Path::to_string_lossy`], because what is passed on is an
/// argument to `mem` and argv here is `String`. A path that is not UTF-8 therefore names
/// a store the far end will not find, and says so on the first command rather than here.
impl From<PathBuf> for Memory {
    fn from(memfile: PathBuf) -> Self {
        Self::new(memfile.to_string_lossy().into_owned())
    }
}

/// The same for a borrowed path.
impl From<&Path> for Memory {
    fn from(memfile: &Path) -> Self {
        Self::new(memfile.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_console;

    /// An empty store, made through the same console the test then uses — so what is
    /// asserted is a store `mem` wrote, not a fixture that resembles one.
    async fn store(console: &mut Console) -> Memory {
        let dir = tempfile::tempdir().unwrap();
        // Leaked rather than dropped: the file has to outlive this function, and the
        // directory goes when the test process does.
        let path = dir
            .keep()
            .join("notes.sqlite")
            .to_string_lossy()
            .to_string();

        let init = console.exec(["mem", "init", &path], None).await.unwrap();
        assert_eq!(init.code, 0, "{}", String::from_utf8_lossy(&init.stderr));

        Memory::new(path)
    }

    /// A store with memories in it, put there by `mem` itself rather than by
    /// [`Memory::insert`] — so a search test that fails is a search that failed.
    async fn filled(console: &mut Console, memories: &[&str]) -> Memory {
        let memory = store(console).await;

        let mut argv = vec![
            "mem".to_string(),
            "insert".to_string(),
            memory.memfile().to_string(),
        ];
        argv.extend(memories.iter().map(|m| m.to_string()));
        let insert = console.exec(&argv, None).await.unwrap();
        assert_eq!(
            insert.code,
            0,
            "{}",
            String::from_utf8_lossy(&insert.stderr)
        );

        memory
    }

    /// A store that was never made, for the tests about what that answers with.
    fn missing() -> Memory {
        let dir = tempfile::tempdir().unwrap();
        Memory::new(
            dir.keep()
                .join("missing.sqlite")
                .to_string_lossy()
                .to_string(),
        )
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_finds_the_nearest_memory() {
        let mut console = test_console().await;
        let memory = filled(
            &mut console,
            &["User drinks tea", "User switched to oat milk"],
        )
        .await;

        let found = memory.search(&mut console, "oat milk").await.unwrap();
        assert_eq!(found, ["User switched to oat milk"]);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_with_nothing_near_is_empty() {
        let mut console = test_console().await;
        let memory = filled(&mut console, &["User drinks tea"]).await;

        let found = memory.search(&mut console, "almond").await.unwrap();
        assert!(found.is_empty(), "nothing near the query: {found:?}");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_on_a_store_that_is_not_there_is_an_error() {
        let mut console = test_console().await;
        let memory = missing();

        let e = memory
            .search(&mut console, "oat milk")
            .await
            .expect_err("there is no such store");
        assert!(e.to_string().contains(memory.memfile()), "{e}");
    }

    /// What was written comes back, in the order it was given — and is in the store, not
    /// merely echoed: asked for straight after, it is there.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_writes_what_it_was_given() {
        let mut console = test_console().await;
        let memory = store(&mut console).await;

        let written = memory
            .insert(
                &mut console,
                ["User switched to oat milk", "User drinks tea"],
            )
            .await
            .unwrap();
        assert_eq!(written, ["User switched to oat milk", "User drinks tea"]);

        let found = memory.search(&mut console, "oat milk").await.unwrap();
        assert_eq!(found, ["User switched to oat milk"]);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_with_nothing_to_write_writes_nothing() {
        let mut console = test_console().await;
        let memory = store(&mut console).await;

        let written = memory
            .insert(&mut console, Vec::<String>::new())
            .await
            .unwrap();
        assert!(written.is_empty(), "no memories, so no lines: {written:?}");
    }

    /// One unusable memory fails the line rather than being dropped from a list that then
    /// looks complete — and the store is left as it was.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_refuses_an_empty_memory() {
        let mut console = test_console().await;
        let memory = store(&mut console).await;

        memory
            .insert(&mut console, ["User drinks tea", "   "])
            .await
            .expect_err("a memory is a statement, and the second is empty");

        let found = memory.search(&mut console, "tea").await.unwrap();
        assert!(found.is_empty(), "nothing was written: {found:?}");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_into_a_store_that_is_not_there_is_an_error() {
        let mut console = test_console().await;
        let memory = missing();

        let e = memory
            .insert(&mut console, ["User drinks tea"])
            .await
            .expect_err("there is no such store");
        assert!(e.to_string().contains(memory.memfile()), "{e}");
        assert!(
            !std::path::Path::new(memory.memfile()).exists(),
            "a failed insert made nothing"
        );
    }
}
