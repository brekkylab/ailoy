//! Memories, as `mem` keeps them.
//!
//! A store is one file made by `mem init`; every read or write is a `mem` command run in a
//! [`Sandbox`], so `mem` must be on the sandbox's `PATH` and the memory file is a path the
//! sandbox opens. Nothing here opens the database or knows its schema.

use std::path::{Path, PathBuf};

use crate::console::Sandbox;

/// One memory store, named by its file.
///
/// Just a path: `mem` opens the file per command, so this is cheap to clone and share;
/// the sandbox lock serializes access. The sandbox is passed per call so an
/// [`Agent`](crate::agent::Agent)'s own sandbox slot stays the one source of truth.
///
/// ```rust,no_run
/// # use ailoy::memory::Memory;
/// # async fn f(sandbox: &ailoy::console::Sandbox) -> anyhow::Result<()> {
/// let memory = Memory::new("/work/notes.sqlite");
/// memory.insert(sandbox, ["User switched to oat milk"]).await?;
/// let found = memory.search(sandbox, "what does the user drink?").await?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Memory {
    memfile: String,
}

impl Memory {
    /// The store at `memfile`, expected to exist already (`mem init`); never created here, so a
    /// mistyped name cannot silently become an empty store.
    ///
    /// Unchecked, since checking needs a sandbox; a missing store surfaces on the first
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
    /// stderr), or a sandbox failure.
    pub async fn search(
        &self,
        sandbox: &Sandbox,
        query: impl AsRef<str>,
    ) -> anyhow::Result<Vec<String>> {
        self.run(sandbox, "search", [query.as_ref()]).await
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
        sandbox: &Sandbox,
        memories: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memories: Vec<String> = memories
            .into_iter()
            .map(|m| m.as_ref().to_string())
            .collect();

        // An empty list still runs `mem`, so a missing store is still reported.
        self.run(sandbox, "insert", &memories).await
    }

    /// Run `mem <subcommand> <memfile> <rest>...` and return its non-blank output lines.
    async fn run(
        &self,
        sandbox: &Sandbox,
        subcommand: &str,
        rest: impl IntoIterator<Item = impl AsRef<str>>,
    ) -> anyhow::Result<Vec<String>> {
        let memfile = &self.memfile;

        let mut argv = vec!["mem".to_string(), subcommand.to_string(), memfile.clone()];
        argv.extend(rest.into_iter().map(|arg| arg.as_ref().to_string()));

        // No timeout: only a large store makes this slow, and the caller still wants it.
        let result = crate::console::exec(sandbox, &argv, None)
            .await
            .map_err(|e| anyhow::anyhow!("`mem {subcommand} {memfile}` could not be run: {e}"))?;

        if result.code != 0 {
            // `mem`'s stderr names the store and reason; the exit code adds nothing.
            let said = result.stderr.trim();
            anyhow::bail!(if said.is_empty() {
                format!("`mem {subcommand} {memfile}` failed (exit {})", result.code)
            } else {
                said.to_string()
            });
        }

        // `mem` never stores a blank memory, so blank lines are just trailing newlines.
        Ok(result
            .stdout
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
    use crate::test_sandbox;

    /// A path in a test sandbox's `/tmp` that no other call answers.
    fn fresh_path(name: &str) -> String {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("/tmp/{name}-{n}.sqlite")
    }

    /// An empty store made by `mem init` on the test's sandbox, not a fixture.
    async fn store(sandbox: &Sandbox) -> Memory {
        let path = fresh_path("notes");

        let init = crate::console::exec(sandbox, ["mem", "init", &path], None)
            .await
            .unwrap();
        assert_eq!(init.code, 0, "{}", init.stderr);

        Memory::new(path)
    }

    /// A store filled by `mem` directly, not [`Memory::insert`], so search tests don't
    /// depend on insert.
    async fn filled(sandbox: &Sandbox, memories: &[&str]) -> Memory {
        let memory = store(sandbox).await;

        let mut argv = vec![
            "mem".to_string(),
            "insert".to_string(),
            memory.memfile().to_string(),
        ];
        argv.extend(memories.iter().map(|m| m.to_string()));
        let insert = crate::console::exec(sandbox, &argv, None).await.unwrap();
        assert_eq!(insert.code, 0, "{}", insert.stderr);

        memory
    }

    /// A store that was never made.
    fn missing() -> Memory {
        Memory::new(fresh_path("missing"))
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_finds_the_nearest_memory() {
        let sandbox = test_sandbox().await;
        let memory = filled(&sandbox, &["User drinks tea", "User switched to oat milk"]).await;

        let found = memory.search(&sandbox, "oat milk").await.unwrap();
        assert_eq!(found, ["User switched to oat milk"]);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_with_nothing_near_is_empty() {
        let sandbox = test_sandbox().await;
        let memory = filled(&sandbox, &["User drinks tea"]).await;

        let found = memory.search(&sandbox, "almond").await.unwrap();
        assert!(found.is_empty(), "nothing near the query: {found:?}");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_search_on_a_store_that_is_not_there_is_an_error() {
        let sandbox = test_sandbox().await;
        let memory = missing();

        let e = memory
            .search(&sandbox, "oat milk")
            .await
            .expect_err("there is no such store");
        assert!(e.to_string().contains(memory.memfile()), "{e}");
    }

    /// Written memories are returned in order and are actually searchable afterward.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_writes_what_it_was_given() {
        let sandbox = test_sandbox().await;
        let memory = store(&sandbox).await;

        let written = memory
            .insert(&sandbox, ["User switched to oat milk", "User drinks tea"])
            .await
            .unwrap();
        assert_eq!(written, ["User switched to oat milk", "User drinks tea"]);

        let found = memory.search(&sandbox, "oat milk").await.unwrap();
        assert_eq!(found, ["User switched to oat milk"]);
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_with_nothing_to_write_writes_nothing() {
        let sandbox = test_sandbox().await;
        let memory = store(&sandbox).await;

        let written = memory.insert(&sandbox, Vec::<String>::new()).await.unwrap();
        assert!(written.is_empty(), "no memories, so no lines: {written:?}");
    }

    /// One blank memory fails the whole insert and leaves the store unchanged.
    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_refuses_an_empty_memory() {
        let sandbox = test_sandbox().await;
        let memory = store(&sandbox).await;

        memory
            .insert(&sandbox, ["User drinks tea", "   "])
            .await
            .expect_err("a memory is a statement, and the second is empty");

        let found = memory.search(&sandbox, "tea").await.unwrap();
        assert!(found.is_empty(), "nothing was written: {found:?}");
    }

    #[test_with::executable(mem)]
    #[tokio::test]
    async fn test_insert_into_a_store_that_is_not_there_is_an_error() {
        let sandbox = test_sandbox().await;
        let memory = missing();

        let e = memory
            .insert(&sandbox, ["User drinks tea"])
            .await
            .expect_err("there is no such store");
        assert!(e.to_string().contains(memory.memfile()), "{e}");
        assert!(
            !std::path::Path::new(memory.memfile()).exists(),
            "a failed insert made nothing"
        );
    }
}
