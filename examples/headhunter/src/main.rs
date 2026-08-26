//! The headhunter example — read a posting, find candidates, write a shortlist and
//! cold-mail drafts.
//!
//! # What this example shows
//!
//! **An app builds an executable in its own domain and attaches it to the console.**
//! That is [`executable`]'s `headhunting`, and it is the agent's only way to the pool.
//! Instead of writing SQL the agent asks in the vocabulary of recruiting — find by
//! condition (`search`), read the ones you picked (`read`), and drop to free-form SQL only
//! when neither can ask it (`query`).
//!
//! # Attaching it costs no code
//!
//! cortex's delegation materializes a delegated name as a symlink on `PATH`, so once the
//! name is registered ailoy's existing `shell` tool calls it — `console.exec(["sh", "-c",
//! cmd])` is already that path. Zero lines of adapter is spec §1.2's conclusion, and
//! `cortex-console-servers/local/tests/exec_sqlite.rs` demonstrated it through a real
//! server process. So the app's work is implementing `Executable` and naming it.
//!
//! # What this file does
//!
//! Reads the CLI, prepares the tree this run will use, assembles the console and the
//! agent, consumes the stream, and checks the result. It does not query the pool — that is
//! [`executable`]'s job. What flows onto the screen and into the query log is built by
//! [`trace`], which holds the tool traffic; bringing it here would put state in two
//! places.

use ailoy::{lang_model::get_lm_providers_mut, message::Role};
use anyhow::{Context, Result, bail};
use clap::Parser;
use cortex::{
    console::{Console, stdio::StdioClient},
    exec::ExecutableSet,
    fs::Mount,
};
use futures::StreamExt;
// Not `std::process::Command`: `StdioClient` reads and writes the pipes on the runtime,
// so it takes the async one. Plan A's passing test carries the same note in the same place.
use tokio::process::Command;

mod prompt;
mod trace;

use headhunter::executable::Headhunting;

#[derive(Parser)]
#[command(about = "Read a job posting, pick the top k from the pool, draft cold mails")]
struct Args {
    /// Path to the posting, as markdown.
    ///
    /// The default is one of the four in `eval/jd`, each of which has an answer key beside
    /// it. A posting with no key runs to the end and writes artifacts that nothing can
    /// grade, which reads as working.
    #[arg(long, default_value = "eval/jd/backend-rust.md")]
    jd: std::path::PathBuf,

    /// How many to shortlist. If fewer qualify, it emits fewer and says why.
    #[arg(long, default_value_t = 3)]
    k: usize,

    /// The candidate pool.
    ///
    /// **A host path**, not a name inside the mounted tree. `headhunting` is a command
    /// the app registers, so it receives this value at registration and holds it — which
    /// is why the command line the agent writes carries no db argument.
    #[arg(long, default_value = "data/headhunter.db")]
    db: std::path::PathBuf,

    /// Where artifacts go. A subdirectory is made under the posting's name.
    #[arg(long, default_value = "artifacts")]
    out: std::path::PathBuf,

    /// The model identifier.
    ///
    /// `<provider>/<model>`; the provider is registered from environment variables. The
    /// default is Bedrock because that is what this example was run on — it reads
    /// `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION`.
    ///
    /// **On Bedrock the model name has to be an inference-profile id** — the prefixed
    /// form like `global.anthropic.claude-sonnet-5`. A bare foundation-model id is
    /// refused under on-demand throughput.
    ///
    /// Other providers work unchanged — `--model anthropic/claude-sonnet-5` reads
    /// `ANTHROPIC_API_KEY`.
    #[arg(long, default_value = "bedrock/global.anthropic.claude-sonnet-5")]
    model: String,

    /// The console server binary, built from the sibling cortex checkout.
    #[arg(
        long,
        env = "AILOY_CORTEX_CONSOLE",
        default_value = "cortex-local-console"
    )]
    console: std::path::PathBuf,

    /// The most tokens the model may emit in one response.
    ///
    /// **This example fails at the default (Anthropic's 8192).** The shortlist is written
    /// through the `write` tool, so the file body rides in the tool-call arguments and
    /// counts against response tokens. Measured: writing a 10KB shortlist ended in
    /// `FinishReason::Length` with zero artifacts — 47 turns and nothing left behind.
    #[arg(long, default_value_t = 32_000)]
    max_tokens: u64,
}

/// Checks **before the first call** that a provider for this model is registered.
///
/// Without it the first LM call dies saying "no provider found", which does not say what
/// is missing. This names the empty environment variable.
fn ensure_provider(model: &str) -> Result<()> {
    let providers = get_lm_providers_mut();
    let default = providers
        .get("default")
        .expect("the default provider is always there");
    if default.get(model).is_some() {
        return Ok(());
    }
    drop(providers);

    let (var, hint) = match model.split('/').next() {
        Some("bedrock") => (
            "AWS_BEARER_TOKEN_BEDROCK",
            "issued from the Bedrock console under API keys",
        ),
        Some("anthropic") => ("ANTHROPIC_API_KEY", "issued at console.anthropic.com"),
        Some("openai") => ("OPENAI_API_KEY", ""),
        _ => ("", ""),
    };
    if !var.is_empty() && std::env::var(var).unwrap_or_default().trim().is_empty() {
        bail!("{var} is empty. Fill in the repository root `.env`. {hint}");
    }
    bail!(
        "no provider for model '{model}'. Check that the API key is in `.env` and that \
         the prefix (`bedrock/`, `anthropic/`, …) is right"
    )
}

/// The working tree — the directory the console treats as one.
///
/// `Mount` does not mean a kernel mount. Its only required method is `mountpoint()`, and
/// `PathBuf` does not implement it, so these five lines are written by hand.
///
/// FUSE is not needed. cortex's `FuseTMount`/`FuseMount` are for when the kernel actually
/// has to answer, and they sit behind the optional `fuser` feature. Using one host
/// directory as a tree does not reach them — `Mounted` in
/// `local/tests/exec_sqlite.rs` is the same five lines, and that test passes through a
/// real server.
struct Tree(std::path::PathBuf);

impl Mount for Tree {
    fn mountpoint(&self) -> &std::path::Path {
        &self.0
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Reads the repository root `.env`. If there is none it moves on quietly and uses
    // whatever is already in the environment.
    //
    // ailoy itself calls `dotenvy::dotenv()` too, but inside `#[cfg(test)]`, so only in
    // tests. A binary has to call it, and without the call you see "no provider found"
    // with the key sitting in `.env`.
    //
    // `dotenv()` walks upward from the current directory, so running from
    // `examples/headhunter` or from the repository root picks up the same file.
    dotenvy::dotenv().ok();

    let args = Args::parse();
    ensure_provider(&args.model)?;
    let jd = std::fs::read_to_string(&args.jd)
        .with_context(|| format!("reading the posting {}", args.jd.display()))?;

    // **The pool is checked here.** Without it the first query dies, and what is left on
    // screen is the agent failing a command — which does not show that the cause is the
    // data.
    if !args.db.is_file() {
        bail!(
            "no pool at {}. Build it with `python3 sql/load.py`",
            args.db.display()
        );
    }
    // The executable opens a host path, not a name inside the mounted tree, so it is made
    // absolute here — the console server may run from a different directory.
    let db = std::fs::canonicalize(&args.db)?;

    let jd_slug = args
        .jd
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "role".to_string());
    let out_dir = args.out.join(&jd_slug);
    let workspace = prepare(&out_dir, &args.jd)?;

    let mut server = Command::new(&args.console);
    // If the server cannot start, its reason comes out on this stderr. Failing quietly
    // would leave only "the console did not attach".
    server.stderr(std::process::Stdio::inherit());

    let console = Console::builder()
        .client(StdioClient::new(server)?)
        .mount(Tree(workspace.clone()))
        .executables(ExecutableSet::new().register(
            "headhunting",
            Headhunting::summary(),
            Headhunting::new(&db),
        ))
        .build()
        .await?;

    // Only `Console::builder().build()` is awaited — `AgentBuilder::build()` is not async.
    //
    // **Without `system_tools()` the agent can do nothing.** Attaching a console and
    // registering tools are separate things: the console is where a tool runs a command,
    // and with no tools there is nobody to run it.
    //
    // This was missed once and **the failure was silent.** The instruction said "You have
    // a `shell` tool" while there was none, so with nothing to call the model imitated a
    // command in prose:
    //
    //     `headhunting search --skill rust`
    //
    // That is not a tool call, it is text. The model ended its turn, the stream ended,
    // and the run finished **in one turn with exit 0** and no error.
    //
    // `system_tools()` gives `shell`, `read`, `write`, `edit`, `glob`, and `grep`.
    // `shell` calls `headhunting` (delegation put it on `PATH`), `read` reads the posting
    // and the schema from the tree, and `write` writes the artifacts. They are not
    // attached individually because the list differs by model family — openai models get
    // `apply_patch`.
    //
    // Keeping those tools from seeing outside the tree is what [`prepare`] does.
    let mut agent = ailoy::agent::AgentBuilder::new(&args.model)
        .console(console)
        .system_tools()
        .max_tokens(args.max_tokens)
        .instruction(prompt::system(args.k))
        .build()?;

    println!("  posting  {}", args.jd.display());
    println!("  model    {}", args.model);
    println!("  console  {}", args.console.display());
    // The paths as given, not the canonicalized ones the mount and the executable hold.
    // Absolute here would put whoever ran it into a record that gets committed.
    println!("  pool     {}", args.db.display());
    println!("  tree     {}", out_dir.display());
    println!("  max      {} tokens/response\n", args.max_tokens);

    // `run` returns a stream, not a single future. Turns advance as it is consumed.
    //
    // **The finish reason is held onto.** With no artifacts it is the only clue to why it
    // stopped: `Stop` means it said it would write and ended the turn, `Length` means
    // max_tokens.
    //
    // What goes on screen is built by [`trace::Trace`]. Full command text goes to a file
    // rather than the screen; what flows here is the agent's words and the scale of what
    // the tools returned.
    let mut turns = 0usize;
    let mut tool_calls = 0usize;
    let mut last_finish: Option<String> = None;
    let mut trace = trace::Trace::default();
    let mut stream = agent.run(prompt::user(&jd, &args.jd));
    while let Some(output) = stream.next().await {
        let output = output?;
        turns += 1;
        if output.message.role == Role::Assistant {
            last_finish = Some(format!("{:?}", output.finish_reason));
        }
        tool_calls += output.message.tool_calls.as_ref().map_or(0, |c| c.len());
        print!("{}", trace.observe(&output));
        use std::io::Write;
        std::io::stdout().flush()?;
    }
    drop(stream);
    // The last group has no words following it, so it only folds after the stream ends.
    print!("{}", trace.finish());

    let queries = trace.queries().len();
    if queries > 0 {
        trace.write_queries(&out_dir.join(trace::QUERY_LOG), &args.jd)?;
    }

    let written = list_files(&out_dir);
    println!("--- run summary ---");
    println!(
        "{turns} turns · {queries} pool calls · {} failed · {tool_calls} tool calls · finish {}",
        trace.failures(),
        last_finish.as_deref().unwrap_or("(no assistant response)")
    );
    let usage = trace.usage();
    println!(
        "tokens  input {} · output {} · cache write {} · cache read {}",
        trace::toks(usage.input),
        trace::toks(usage.output),
        trace::toks(usage.cache_write),
        trace::toks(usage.cache_read)
    );
    // Printed only when caching took hold. Without it the two figures are equal and there
    // is nothing to say.
    //
    // **There is a reason the effective input is computed here.** The four figures are
    // priced differently (1× list, 1.25× write, 0.1× read), so merely listing them leaves
    // the reader to do the arithmetic, and nobody does. And reading `input` alone makes
    // the cost look near zero once caching takes hold, when it has only moved to
    // `cache_read`.
    if usage.cache_read > 0 || usage.cache_write > 0 {
        println!(
            "        effective input {} (uncached it would be {})",
            trace::toks(usage.effective_input()),
            trace::toks(usage.uncached_input())
        );
    }
    println!("{} artifacts", written.len());
    for path in &written {
        let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // Size comes first for alignment. With the name first, a Korean filename takes
        // twice the width and padding like `{:<28}` goes crooked.
        println!(
            "  {:>8}  {}",
            trace::size(bytes),
            path.file_name().unwrap_or_default().to_string_lossy()
        );
    }
    if queries > 0 {
        println!(
            "query log  {}  ({queries} calls, untruncated)",
            out_dir.join(trace::QUERY_LOG).display()
        );
    }

    // **What is checked is whether a shortlist came out, not whether files appeared.**
    // Counting a run that stopped partway with only mail drafts as a success would make
    // the check meaningless. And zero tool calls means the agent imitated commands in
    // prose — which has actually happened.
    // **If it reached for the answer key, this run's output cannot be used to evaluate.**
    //
    // Narrowing the tree only stops the file tools. `shell` is spawned on the host by the
    // console server, so an absolute path goes anywhere — it cannot be prevented, so what
    // is checked is whether it happened.
    if !trace.escapes().is_empty() {
        for cmd in trace.escapes() {
            eprintln!(
                "  ✗ outside the tree: {}",
                cmd.lines().next().unwrap_or(cmd)
            );
        }
        bail!(
            "the agent reached the answer key or the scoring criteria outside the tree \
             through the shell ({} times). This run's artifacts cannot be evaluated",
            trace.escapes().len()
        );
    }

    if tool_calls == 0 {
        bail!(
            "zero tool calls ({turns} turns, finish reason {}). The agent most likely \
             wrote commands as text instead of running them — check that `system_tools()` \
             is attached",
            last_finish.as_deref().unwrap_or("?")
        );
    }
    if !written
        .iter()
        .any(|p| p.file_name().is_some_and(|n| n == "00-shortlist.md"))
    {
        bail!(
            "no 00-shortlist.md ({} artifacts, finish reason {}). Length means it hit \
             max_tokens; Stop means it said it would write and ended the turn",
            written.len(),
            last_finish.as_deref().unwrap_or("?")
        );
    }

    println!("score     eval/run_eval.py --score {}", out_dir.display());
    Ok(())
}

/// Builds the directory this run uses as its tree and puts what the agent may read in it.
///
/// # Why the current directory is not mounted whole
///
/// The example directory holds things the agent must not see. `data/ground_truth.json`
/// is the **answer key**, with trap labels and verdicts; `eval/expected/*.json` is the
/// scoring criteria; `data/candidates.json` is the entire source pool. Mount the whole
/// directory and one `read` call reaches all of it.
///
/// So the tree holds only what this run needs.
///
/// # This is not a wall
///
/// Only the file tools are confined to the tree. **`shell` is not** —
/// `cortex-local-console` spawns `sh -c` on the host and only matches `current_dir` to
/// the session, so an absolute path goes anywhere. In a real run the agent wrote
/// intermediate results to `/tmp` and those files stayed on the host.
///
/// What this does, then, is **narrow the default path**. Whether anything off limits was
/// actually reached is answered at the end of the run by [`trace::Trace::escapes`].
///
/// ```text
/// <out>/<jd>/          the root of the tree and where artifacts are written
///   in/jd.md           the posting
///   in/schema.sql      table definitions. The views are not here — see below
/// ```
///
/// Inputs go under `in/` because of counting and scoring. On the same level as the
/// artifacts, [`list_files`] would count them as artifacts and `run_eval.py`'s `*.md`
/// would pick up the posting.
fn prepare(out_dir: &std::path::Path, jd: &std::path::Path) -> Result<std::path::PathBuf> {
    let inputs = out_dir.join("in");
    std::fs::create_dir_all(&inputs)
        .with_context(|| format!("creating the working directory {}", inputs.display()))?;

    std::fs::copy(jd, inputs.join("jd.md"))
        .with_context(|| format!("copying the posting {}", jd.display()))?;
    // Tables only. **`views.sql` deliberately does not go into the tree.**
    //
    // Every view is behind a command: `candidate_tenure` and `current_position` are folded
    // into `search` and `read`, `candidate_brief` is what `search` stands on, and the
    // distributions are `distribution`'s axes. Measured across three runs, the 87 free-form
    // SQL statements touched a view five times, and all five asked for figures `read`
    // already carries.
    //
    // Handing over the definitions contradicts what this example claims — that the command
    // is the only way to the pool — and `location_distribution` was worse than unused: it
    // reports `positions.location`, which drifts, while `--city` matches the normalized
    // `candidates.city`. Two runs lost turns to that mismatch.
    //
    // The schema is copied from the repository as is. A transcribed copy would let the
    // agent read stale definitions after the real schema changed, and that failure is
    // silent — asking for a column that is gone merely errors, while missing one that
    // exists leaves you unaware it was ever there.
    let from = std::path::Path::new("sql").join("schema.sql");
    std::fs::copy(&from, inputs.join("schema.sql"))
        .with_context(|| format!("copying the schema {}", from.display()))?;

    // A mount takes a host path. Left relative it would point somewhere else depending on
    // the console server's working directory.
    std::fs::canonicalize(out_dir).with_context(|| format!("resolving {}", out_dir.display()))
}

/// What the run produced, directly under the directory. It does not recurse — artifacts
/// are one level.
///
/// **Not everything in there.** A record of the run shares the directory (`queries.log`,
/// and the screen kept as `console.txt`), and so does prose written afterwards for a
/// person — `SCENARIO.md`. Counted as artifacts they blur the one figure that says what
/// came out. [`prompt`] fixes the filenames as `NN-<slug>.md`, so that shape is what
/// identifies one. `run_eval.py` narrows the same way and for the same reason.
fn list_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out: Vec<_> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(is_artifact)
        })
        .collect();
    out.sort();
    out
}

/// `NN-<slug>.md` — two digits, a hyphen, and a markdown extension.
fn is_artifact(name: &str) -> bool {
    let mut head = name.chars();
    let numbered = matches!(
        (head.next(), head.next(), head.next()),
        (Some(a), Some(b), Some('-')) if a.is_ascii_digit() && b.is_ascii_digit()
    );
    numbered && name.ends_with(".md")
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, list_files};

    /// Running with no arguments has to produce something that can be scored.
    ///
    /// A default posting with no answer key beside it reads as working — the run finishes
    /// and writes artifacts — and then `run_eval.py --score` has nothing to grade it by.
    #[test]
    fn the_default_posting_has_an_answer_key() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let jd = root.join(Args::parse_from(["headhunter"]).jd);
        assert!(jd.is_file(), "no posting at {}", jd.display());

        let key = root
            .join("eval/expected")
            .join(jd.file_stem().unwrap())
            .with_extension("json");
        assert!(key.is_file(), "no answer key at {}", key.display());
    }

    /// A run directory holds more than the run produced. `SCENARIO.md` is written after the
    /// fact for a person to read, and `console.txt` is the screen kept as a record. Counted
    /// among the artifacts they inflate the figure that says what came out.
    #[test]
    fn only_the_numbered_artifacts_are_counted() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "00-shortlist.md",
            "01-someone.md",
            "SCENARIO.md",
            "console.txt",
            "queries.log",
        ] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }

        let names: Vec<_> = list_files(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        assert_eq!(names, ["00-shortlist.md", "01-someone.md"]);
    }
}
