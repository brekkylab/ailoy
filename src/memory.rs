//! Memories, as `mem` keeps them.
//!
//! A store is one file made by `mem init`; every read or write is a `mem` command run on a
//! [`ConsoleClient`]. Nothing here opens the database or knows its schema.
//!
//! So `mem` must be on the console side's `PATH`, and the memory file is a path the console
//! side opens, not one this process resolves.
//!
//! Stores are never created here: keeping `mem init` separate stops a mistyped name from
//! silently becoming an empty store.

use std::path::{Path, PathBuf};

use crate::console::ConsoleClient;

/// One memory store, named by its file.
///
/// Just a path: `mem` opens the file per command, so this is cheap to clone and share;
/// the console lock serializes access. The console is passed per call so an
/// [`Agent`](crate::agent::Agent)'s own console slot stays the one source of truth.
///
/// ```rust,no_run
/// # use ailoy::memory::Memory;
/// # async fn f(console: &mut ailoy::console::ConsoleClient) -> anyhow::Result<()> {
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
    /// The store at `memfile`, expected to exist already (`mem init`).
    ///
    /// Unchecked, since checking needs a console; a missing store surfaces on the first
    /// [`search`](Self::search) or [`insert`](Self::insert).
    pub fn new(memfile: impl Into<String>) -> Self {
        Self {
            memfile: memfile.into(),
        }
    }

    /// The store path as given.
    pub fn memfile(&self) -> &str {
        &self.memfile
    }

    /// The memories nearest `query`, nearest first.
    ///
    /// Runs `mem search <memfile> <query>`, one memory per output line. Nothing near the
    /// query is an empty `Vec`, not an error.
    ///
    /// Passed as argv, not through a shell, so nothing needs quoting.
    ///
    /// # Errors
    ///
    /// A missing store, a non-store file, or an unrunnable `mem` (the error is `mem`'s
    /// stderr), or a console failure.
    pub async fn search(
        &self,
        console: &mut ConsoleClient,
        query: impl AsRef<str>,
    ) -> anyhow::Result<Vec<String>> {
        self.run(console, "search", [query.as_ref()]).await
    }

    /// Write `memories` into the store verbatim, returning the lines `mem` reports.
    ///
    /// Runs `mem insert <memfile> <memories>...`, one memory per argument so no separator
    /// can collide with memory text. No deduplication: identical text may mean different
    /// things. An empty list writes nothing and returns an empty `Vec`.
    ///
    /// # Errors
    ///
    /// A missing store (none is created), or any whitespace-only memory, in which case
    /// nothing is written.
    pub async fn insert(
        &self,
        console: &mut ConsoleClient,
        memories: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memories: Vec<String> = memories
            .into_iter()
            .map(|m| m.as_ref().to_string())
            .collect();

        // An empty list still runs `mem`, so a missing store is still reported.
        self.run(console, "insert", &memories).await
    }

    /// Run `mem <subcommand> <memfile> <rest>...` and return its non-blank output lines.
    async fn run(
        &self,
        console: &mut ConsoleClient,
        subcommand: &str,
        rest: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memfile = &self.memfile;

        let mut argv = vec!["mem".to_string(), subcommand.to_string(), memfile.clone()];
        argv.extend(rest.into_iter().map(|arg| arg.as_ref().to_string()));

        // No timeout: only a large store makes this slow, and the caller still wants it.
        let result = console
            .exec(&argv, None)
            .await
            .map_err(|e| anyhow::anyhow!("`mem {subcommand} {memfile}` could not be run: {e}"))?;

        if result.code != 0 {
            // `mem`'s stderr names the store and reason; the exit code adds nothing.
            let said = String::from_utf8_lossy(&result.stderr);
            let said = said.trim();
            anyhow::bail!(if said.is_empty() {
                format!("`mem {subcommand} {memfile}` failed (exit {})", result.code)
            } else {
                said.to_string()
            });
        }

        // `mem` never stores a blank memory, so blank lines are just trailing newlines.
        Ok(String::from_utf8_lossy(&result.stdout)
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.to_string())
            .collect())
    }
}

impl From<String> for Memory {
    fn from(memfile: String) -> Self {
        Self::new(memfile)
    }
}

impl From<&str> for Memory {
    fn from(memfile: &str) -> Self {
        Self::new(memfile)
    }
}

/// Converted with [`Path::to_string_lossy`] since argv is `String`; a non-UTF-8 path
/// names a store that the first command will report missing. Applies to `&Path` too.
impl From<PathBuf> for Memory {
    fn from(memfile: PathBuf) -> Self {
        Self::new(memfile.to_string_lossy().into_owned())
    }
}

impl From<&Path> for Memory {
    fn from(memfile: &Path) -> Self {
        Self::new(memfile.to_string_lossy().into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_console;

    /// An empty store made by `mem init` on the test's console, not a fixture.
    async fn store(console: &mut ConsoleClient) -> Memory {
        let dir = tempfile::tempdir().unwrap();
        // Kept so the file outlives this function.
        let path = dir
            .keep()
            .join("notes.sqlite")
            .to_string_lossy()
            .to_string();

        let init = console.exec(["mem", "init", &path], None).await.unwrap();
        assert_eq!(init.code, 0, "{}", String::from_utf8_lossy(&init.stderr));

        Memory::new(path)
    }

    /// A store filled by `mem` directly, not [`Memory::insert`], so search tests don't
    /// depend on insert.
    async fn filled(console: &mut ConsoleClient, memories: &[&str]) -> Memory {
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

    /// A store that was never made.
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

    /// Written memories are returned in order and are actually searchable afterward.
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

    /// One blank memory fails the whole insert and leaves the store unchanged.
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
