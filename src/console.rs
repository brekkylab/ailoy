//! The sandbox a tool runs commands in, and what its tools do with it.
//!
//! A tool takes the agent's [`Sandbox`] as virtx makes it. What the built-in tools need beyond
//! it is here: a command's whole output, and a path as the files API names it.

use futures::StreamExt;
pub use virtx::{Error, Sandbox};
use virtx::{Files as _, SandboxExecArgs, SandboxJobEntry};

/// What a command printed, line by line, and how it ended.
#[derive(Clone, Debug, Default)]
pub struct ExecOutput {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
    /// The server stopped keeping its output partway.
    pub truncated: bool,
    /// It was killed for running past the timeout it was given.
    pub timed_out: bool,
}

/// Run `argv` in `sandbox`, without a shell, and wait for it to end; killed after
/// `timeout_ms`.
pub async fn exec<I, S>(
    sandbox: &Sandbox,
    argv: I,
    timeout_ms: Option<u64>,
) -> Result<ExecOutput, Error>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = SandboxExecArgs::new(argv);
    args.timeout_ms = timeout_ms;
    let mut log = sandbox.exec(&args).log();
    let mut out = ExecOutput::default();
    while let Some(entry) = log.next().await {
        match entry?.entry {
            SandboxJobEntry::Stdout(line) => push_line(&mut out.stdout, &line),
            SandboxJobEntry::Stderr(line) => push_line(&mut out.stderr, &line),
            SandboxJobEntry::Truncated {} => out.truncated = true,
            SandboxJobEntry::End(end) => {
                out.code = end.code;
                // The server kills a command past its timeout with SIGKILL.
                out.timed_out = timeout_ms.is_some() && end.code == 128 + 9;
                return Ok(out);
            }
        }
    }
    Err(Error::Call("the command's log ended before it did".into()))
}

/// The whole file at `path` in `sandbox`, as a command names it.
pub async fn read(sandbox: &Sandbox, path: &str) -> Result<Vec<u8>, Error> {
    let path = resolve(sandbox, path).await?;
    sandbox.read_file(path, 0, None).await
}

/// Write `content` as the whole file at `path` in `sandbox`, as a command names it, making its
/// directory if it is not there.
pub async fn write(sandbox: &Sandbox, path: &str, content: impl AsRef<[u8]>) -> Result<(), Error> {
    let path = resolve(sandbox, path).await?;
    if let Some((parent, _)) = path.rsplit_once('/') {
        sandbox.create_dir_all(parent).await?;
    }
    sandbox.write_file(path, content, None).await
}

/// `path` as the files API names it: from the root, without its leading `/`. A relative one is
/// from the directory commands run in, as a command would read it.
pub async fn resolve(sandbox: &Sandbox, path: &str) -> Result<String, Error> {
    let absolute = if path.starts_with('/') {
        path.to_owned()
    } else {
        let workdir = exec(sandbox, ["pwd"], Some(10_000)).await?.stdout;
        format!("{}/{path}", workdir.trim().trim_end_matches('/'))
    };
    Ok(absolute.trim_start_matches('/').to_owned())
}

fn push_line(into: &mut String, line: &str) {
    into.push_str(line);
    into.push('\n');
}
