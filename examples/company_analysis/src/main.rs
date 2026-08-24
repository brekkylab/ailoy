//! Ask a question about a company, get an answer with its sources attached.
//!
//! ```sh
//! cargo run -p company_analysis -- --preset entity-profile --company "Samsung Electronics"
//! cargo run -p company_analysis -- --task "Who owns Alphabet's subsidiaries?"
//! ```
//!
//! The registries are not copied here. Each is a [`cortex::fs::FileSystem`] answered by
//! its API and mounted, so a `read` is a request and a listing is what the registry says
//! now. The built-in file tools are the only interface the agent gets.

mod apifs;
mod edgar;
mod gleif;
mod guard;
mod prompt;

use std::path::{Path, PathBuf};

use ailoy::{
    agent::AgentBuilder,
    console::Console,
    lang_model::get_lm_providers_mut,
    message::{Message, Part, Role, TokenUsage},
};
use anyhow::{Context, Result, bail};
use futures::StreamExt;

use crate::{
    edgar::EdgarFs,
    gleif::GleifFs,
    guard::WriteBoundary,
    prompt::{Paths, Preset},
};

const DEFAULT_MODEL: &str = "anthropic/claude-sonnet-5";

/// Commands run on the host. The mounts being read-only is the stores' doing, not this
/// process's, and everything outside them is covered by [`WriteBoundary`] after the
/// fact rather than before.
const CONSOLE_PROGRAM: &str = "cortex-local-console";

/// What the mountpoint says about itself, written beside the two mounts.
///
/// Each store serves its own catalogue at its own root, but the instruction sends the
/// agent to the catalogue above them, and that path is a plain directory — the one
/// place in this tree that no store answers for.
const LIVE_CATALOG: &str = r#"# Two registries, mounted side by side

    gleif/     the Global LEI Index — legal entities, and who owns whom
    edgar/     SEC filings — US registrants, their disclosures and XBRL facts

Each has its own `CATALOG.md`. Read the one for the registry you need: they do not
share a path grammar by accident, but they do differ in what a search gives back.

## There is no reliable key between them

GLEIF's identifier is the LEI and EDGAR's is the CIK, and neither registry carries the
other's. `submissions.json` has an `lei` field that is almost always null — fifteen
registrants sampled across the ticker list had it empty, every one. EIN does not bridge
either, because GLEIF records whatever the local registration authority issued, which
for a US entity is a state filing number or an SEC series id rather than the EIN.

So crossing between them means **matching on a name**, and a name match is a candidate
rather than a fact. Legal-form suffixes differ, holding companies share words with
their subsidiaries, and funds are named after the companies they track — a search for
NVIDIA returns a fund before it returns the manufacturer. Confirm a match against
something else (country, address, registration date) and say in the report that the two
records were joined by name.

## What neither covers

Financial statements for a non-US company, sanctions, litigation, news. A question that
needs those has no answer here, and saying so is the answer.
"#;

struct Args {
    task: Option<String>,
    preset: Option<Preset>,
    company: Option<String>,
    since: Option<PathBuf>,
    out: PathBuf,
    workspace: PathBuf,
    model: String,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        task: None,
        preset: None,
        company: None,
        since: None,
        out: PathBuf::from("./artifacts"),
        workspace: PathBuf::from("./workspace"),
        model: DEFAULT_MODEL.to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--task" => a.task = Some(value()?),
            "--task-file" => {
                let p = value()?;
                a.task = Some(std::fs::read_to_string(&p).with_context(|| format!("read {p}"))?);
            }
            "--preset" => {
                let v = value()?;
                a.preset = Some(Preset::parse(&v).ok_or_else(|| {
                    anyhow::anyhow!("unknown preset '{v}' (one of: {})", Preset::ALL.join(", "))
                })?);
            }
            "--company" => a.company = Some(value()?),
            "--since" => a.since = Some(PathBuf::from(value()?)),
            "--out" => a.out = PathBuf::from(value()?),
            "--workspace" => a.workspace = PathBuf::from(value()?),
            "--model" => a.model = value()?,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument '{other}' (see --help)"),
        }
    }
    if a.task.is_none() && a.preset.is_none() {
        bail!("one of --task or --preset is required (see --help)");
    }
    if a.preset.is_some() && a.company.is_none() {
        bail!("a preset needs --company");
    }
    Ok(a)
}

fn print_help() {
    println!(
        "\
company_analysis — an agent over two mounted company registries

  --task <sentence>    a question in your own words
  --task-file <path>   the same, read from a file
  --preset <name>      {presets}
  --company <name>     the subject, required by every preset
  --since <findings>   a previous run's findings.json; the report then covers changes
  --out <path>         default ./artifacts
  --workspace <path>   default ./workspace
  --model <id>         default {DEFAULT_MODEL}

Environment:
  ANTHROPIC_API_KEY      or whichever key the chosen model's provider wants
  SEC_USER_AGENT         required: SEC answers 403 to a User-Agent naming no contact,
                         e.g. 'company-analysis you@example.com'
  AILOY_CORTEX_CONSOLE   path to the console server binary, if it is not on PATH

Mounting needs a libfuse provider at build time; see the README.",
        presets = Preset::ALL.join(" | "),
    );
}

/// `1787539333-alphabet-entity-profile`. Unique per run, so the epoch goes in front.
/// Seconds rather than a date because a date needs a formatting crate and this does not.
fn run_slug(args: &Args) -> String {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let label = args
        .preset
        .map(|p| p.slug().to_string())
        .unwrap_or_else(|| "adhoc".to_string());
    match args.company.as_deref().map(slugify).filter(|s| !s.is_empty()) {
        Some(c) => format!("{stamp}-{c}-{label}"),
        None => format!("{stamp}-{label}"),
    }
}

/// Non-alphanumerics become `-`, and runs of them collapse to one.
fn slugify(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_alphanumeric() {
            out.extend(c.to_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// Token usage across the run.
///
/// Anthropic-shaped APIs report the whole conversation as each call's `input_tokens`,
/// so the sum is not a count of distinct tokens — it is what was billed. How large the
/// conversation grew is a separate question, which `peak_input` answers.
#[derive(Default)]
struct Usage {
    calls: usize,
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    peak_input: u64,
}

impl Usage {
    fn add(&mut self, u: Option<&TokenUsage>) {
        let Some(u) = u else { return };
        self.calls += 1;
        self.input += u.input_tokens;
        self.output += u.output_tokens;
        self.cache_read += u.cache_read_input_tokens.unwrap_or(0);
        self.cache_write += u.cache_creation_input_tokens.unwrap_or(0);
        self.peak_input = self.peak_input.max(u.input_tokens);
    }
}

impl std::fmt::Display for Usage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.calls == 0 {
            return f.write_str("tokens: unknown (the provider reported no usage)");
        }
        write!(
            f,
            "tokens ({} calls): in {} / out {}",
            self.calls,
            thousands(self.input),
            thousands(self.output)
        )?;
        if self.cache_read > 0 || self.cache_write > 0 {
            write!(
                f,
                " / cache read {} / cache write {}",
                thousands(self.cache_read),
                thousands(self.cache_write)
            )?;
        }
        write!(
            f,
            "\n  largest context {} (the conversation at its last call)",
            thousands(self.peak_input)
        )
    }
}

/// `1234567` → `1,234,567`.
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// The guards and the stores behind them. Dropping this unmounts both.
struct Live {
    _mounts: (cortex::fs::FuseTMount, cortex::fs::FuseTMount),
    gleif: std::sync::Arc<GleifFs>,
    edgar: std::sync::Arc<EdgarFs>,
}

/// Mount the two registries under `at`.
///
/// Two mounts rather than one router: a mount point is what a kernel already uses to
/// join trees, so dispatching on the first path segment would be code for something the
/// OS does. The cost is that `at` itself stays an ordinary directory — writes there are
/// caught by [`WriteBoundary`] rather than refused, which is a narrower claim than the
/// mounts make about themselves, and the run summary says so.
/// Where the registries are mounted, deliberately outside the working tree.
///
/// A file here is a paid request and anything that walks a directory opens every file in
/// it, so an editor or terminal indexing the project spends hundreds of requests that no
/// command asked for. Not a flag: a mountpoint a caller could put back under a watched
/// directory is that cost waiting to return.
fn mountpoint() -> PathBuf {
    std::env::temp_dir().join("ailoy-company-analysis")
}

fn mount_live(at: &Path) -> Result<Live> {
    let gleif_dir = at.join("gleif");
    let edgar_dir = at.join("edgar");
    std::fs::create_dir_all(&gleif_dir)?;
    std::fs::create_dir_all(&edgar_dir)?;
    std::fs::write(at.join("CATALOG.md"), LIVE_CATALOG)?;

    // Required rather than defaulted: SEC answers 403 with an HTML page to a
    // `User-Agent` that names no contact, and a default would only move the failure to
    // the first read — where it arrives as a JSON syntax error rather than as this.
    let ua = std::env::var("SEC_USER_AGENT").map_err(|_| {
        anyhow::anyhow!(
            "SEC_USER_AGENT is required. SEC refuses a User-Agent that names no contact.\n\
             Example: SEC_USER_AGENT='company-analysis you@example.com'"
        )
    })?;

    let gleif = std::sync::Arc::new(GleifFs::new());
    let edgar = std::sync::Arc::new(EdgarFs::new(&ua));
    let mounts = (
        cortex::fs::FuseTMount::try_new(gleif.clone(), &gleif_dir)
            .with_context(|| format!("mounting {}", gleif_dir.display()))?,
        cortex::fs::FuseTMount::try_new(edgar.clone(), &edgar_dir)
            .with_context(|| format!("mounting {}", edgar_dir.display()))?,
    );
    Ok(Live {
        _mounts: mounts,
        gleif,
        edgar,
    })
}

/// Built-in tools do not run without a console server.
///
/// The binary lives in the `cortex` checkout rather than on `PATH`, so its location can
/// be named. No mount is declared here: this server takes its own working directory as
/// the session's, and that is already the tree.
async fn console() -> Result<Console> {
    let program =
        std::env::var("AILOY_CORTEX_CONSOLE").unwrap_or_else(|_| CONSOLE_PROGRAM.to_string());
    let mut console = Console::builder()
        .stdio_client(&[&program])
        .build()
        .await
        .with_context(|| {
            format!(
                "could not start the console server '{program}'. Build it in the cortex \
                 checkout with `cargo build -p {CONSOLE_PROGRAM}` and point \
                 $AILOY_CORTEX_CONSOLE at the binary."
            )
        })?;
    console.start().await.context("starting the console")?;
    Ok(console)
}

/// Fail before assembly rather than at the first call, where the cause is harder to see.
fn ensure_provider(model: &str) -> Result<()> {
    let providers = get_lm_providers_mut();
    let default = providers
        .get("default")
        .expect("the default provider always exists");
    if default.get(model).is_some() {
        return Ok(());
    }
    bail!(
        "no provider is registered for model '{model}'. Set the API key its provider \
         reads (ANTHROPIC_API_KEY for anthropic/*) in the repository's .env or the \
         environment."
    )
}

fn build_task(args: &Args) -> Result<String> {
    let mut task = match (&args.task, args.preset) {
        (Some(t), _) => t.clone(),
        (None, Some(p)) => p.task(args.company.as_deref().unwrap_or("")),
        (None, None) => unreachable!("parse_args rejects this"),
    };
    if let Some(since) = &args.since {
        let prev = std::fs::read_to_string(since)
            .with_context(|| format!("reading the previous findings {}", since.display()))?;
        task.push_str("\n\n# The previous run's findings — report what changed\n\n```json\n");
        task.push_str(&prev);
        task.push_str("\n```\n");
    }
    Ok(task)
}

fn list_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(list_files(&p));
        } else {
            out.push(p);
        }
    }
    out.sort();
    out
}

#[tokio::main]
async fn main() -> Result<()> {
    // The crate reads `.env` only under `#[cfg(test)]`, so a run has to ask.
    dotenvy::dotenv().ok();
    let args = parse_args()?;
    ensure_provider(&args.model)?;

    // Only what the run writes. The console's root is a place to stand rather than a
    // confinement, so the read-only mounts are reachable from outside it — see
    // [`mountpoint`].
    let tree = std::env::current_dir()?.canonicalize()?;
    for (label, path) in [("--out", &args.out), ("--workspace", &args.workspace)] {
        if !tree.join(path).starts_with(&tree) {
            bail!(
                "{label} {} is outside the session tree {}",
                path.display(),
                tree.display()
            );
        }
    }

    let mountpoint = mountpoint();
    std::fs::create_dir_all(&mountpoint)?;
    // Held to the end of `main`: the guard is the mount, and dropping it unmounts.
    let live = mount_live(&mountpoint)?;

    let slug = run_slug(&args);
    let artifacts_dir = args.out.join(&slug);
    let workspace_dir = args.workspace.join(&slug);
    std::fs::create_dir_all(&artifacts_dir)?;
    std::fs::create_dir_all(&workspace_dir)?;

    let boundary = WriteBoundary::new([artifacts_dir.clone(), workspace_dir.clone()]);
    boundary.check(&artifacts_dir.join("report.md"))?;

    let task = build_task(&args)?;
    let paths = Paths {
        data: &mountpoint.to_string_lossy(),
        workspace: &args.workspace.to_string_lossy(),
        artifacts: &args.out.to_string_lossy(),
    };
    let instruction = prompt::instruction(&paths, args.preset, &slug);

    let mut agent = AgentBuilder::new(&args.model)
        .instruction(instruction)
        .system_tools()
        .python_repl_tool()
        .console(console().await?)
        .build()
        .context("assembling the agent")?;

    println!("model    {}", args.model);
    println!("console  {CONSOLE_PROGRAM} (on the host — writes outside the mounts are detected, not refused)");
    println!("mounts   {}/{{gleif,edgar}}", mountpoint.display());
    println!("tree     {}", tree.display());
    println!("run      {slug}\n");

    let mut turns = 0usize;
    let mut tool_calls = 0usize;
    // Requests already billed, so each call can be charged for its own.
    let mut spent = (0usize, 0usize);
    let mut usage = Usage::default();
    // The loop ends the moment a finish reason is not a tool call. When nothing was
    // produced, this is the only clue as to why.
    let mut last_finish = None;
    let mut stream = agent.run(Message::new(Role::User).with_contents([Part::text(task)]));
    while let Some(output) = stream.next().await {
        let output = output?;
        turns += 1;
        if output.message.role == Role::Assistant {
            last_finish = Some(format!("{:?}", output.finish_reason));
        }
        for part in &output.message.contents {
            if let Part::Text { text } = part {
                if output.message.role == Role::Assistant {
                    println!("{text}");
                }
            }
        }
        if let Some(calls) = output.message.tool_calls.as_ref() {
            for c in calls {
                let ailoy::message::Part::Function { function: f, .. } = c else {
                    continue;
                };
                // The id names nothing the reader needs and crowds out the arguments.
                let args = f
                    .arguments
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .map(|(k, v)| match v.as_str() {
                                Some(t) => format!("{k}={t}"),
                                None => format!("{k}={v:?}"),
                            })
                            .collect::<Vec<_>>()
                            .join("  ")
                    })
                    .unwrap_or_else(|| format!("{:?}", f.arguments));
                let one = args.replace('\n', " ⏎ ");
                let shown = match one.char_indices().nth(200) {
                    Some((i, _)) => format!("{}…", &one[..i]),
                    None => one,
                };
                // Charged per call, so the cost lands on the command that caused it.
                let (g, e) = (live.gleif.calls(), live.edgar.calls());
                let (dg, de) = (g - spent.0, e - spent.1);
                spent = (g, e);
                let cost = match (dg, de) {
                    (0, 0) => String::new(),
                    (a, 0) => format!(" [gleif +{a}]"),
                    (0, b) => format!(" [edgar +{b}]"),
                    (a, b) => format!(" [gleif +{a} edgar +{b}]"),
                };
                println!("  → {:<12} {shown}{cost}", f.name);
            }
        }
        tool_calls += output.message.tool_calls.as_ref().map_or(0, |c| c.len());
        usage.add(output.usage.as_ref());
    }
    drop(stream);

    let written = list_files(&artifacts_dir);
    println!("\n--- run summary ---");
    println!("turns {turns} / tool calls {tool_calls}");
    println!("{usage}");
    println!(
        "requests  gleif {} / edgar {}",
        live.gleif.calls(),
        live.edgar.calls()
    );
    // A total says a run was expensive; this says what it bought.
    for (store, rows) in [
        ("gleif", live.gleif.breakdown()),
        ("edgar", live.edgar.breakdown()),
    ] {
        if rows.is_empty() {
            continue;
        }
        let detail: Vec<String> = rows.iter().map(|(k, n)| format!("{k} ×{n}")).collect();
        println!("  {store:<6}  {}", detail.join(", "));
        let (hot, distinct) = match store {
            "gleif" => (live.gleif.hot_keys(3), live.gleif.distinct_keys()),
            _ => (live.edgar.hot_keys(3), live.edgar.distinct_keys()),
        };
        println!("          {distinct} distinct paths; heaviest:");
        for (k, n) in hot {
            println!("            {n:>4} × {k}");
        }
    }
    println!(
        "finish    {}",
        last_finish.as_deref().unwrap_or("(no assistant message)")
    );
    println!("artifacts {}:", written.len());
    for p in &written {
        println!("  {}", p.display());
    }

    let escaped: Vec<_> = written.iter().filter(|p| !boundary.permits(p)).collect();
    if !escaped.is_empty() {
        println!("write boundary: **violated** {escaped:?}");
        bail!("a write left the allowed directories");
    }

    // Not "did anything appear" but "did the report appear". A run that left query
    // files and stopped is not a run that answered.
    if !written
        .iter()
        .any(|p| p.file_name().is_some_and(|n| n == "report.md"))
    {
        bail!(
            "no report.md ({} artifacts, finish {}). `Length` means the cap was hit; \
             `Stop` means the turn ended on a promise to write rather than a write.",
            written.len(),
            last_finish.as_deref().unwrap_or("?")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_collapses_runs_and_keeps_letters() {
        assert_eq!(slugify("Acme Materials Co., Ltd."), "acme-materials-co-ltd");
        // GLEIF records a legal name in its own script, so a company argument often is
        // not Latin. Dropping those characters would leave runs named after their
        // punctuation.
        assert_eq!(slugify("金河化学工业有限公司"), "金河化学工业有限公司");
        assert_eq!(slugify("株式会社 柏商事"), "株式会社-柏商事");
        assert_eq!(slugify("---"), "");
    }

    #[test]
    fn thousands_groups_from_the_right() {
        for (n, want) in [
            (0, "0"),
            (7, "7"),
            (100, "100"),
            (1_000, "1,000"),
            (1_234_567, "1,234,567"),
        ] {
            assert_eq!(thousands(n), want, "{n}");
        }
    }

    #[test]
    fn usage_sums_what_was_billed_and_tracks_the_peak() {
        let mut u = Usage::default();
        u.add(Some(&TokenUsage {
            input_tokens: 1_000,
            output_tokens: 50,
            cache_creation_input_tokens: None,
            cache_read_input_tokens: Some(900),
        }));
        u.add(Some(&TokenUsage {
            input_tokens: 12_345,
            output_tokens: 60,
            cache_creation_input_tokens: Some(11),
            cache_read_input_tokens: None,
        }));
        u.add(None); // a tool result carries no usage and is not a call
        assert_eq!(u.calls, 2);
        assert_eq!(u.input, 13_345);
        assert_eq!(u.output, 110);
        assert_eq!(u.cache_read, 900);
        assert_eq!(u.cache_write, 11);
        // Not the sum: the conversation was this big once, not thirteen thousand times.
        assert_eq!(u.peak_input, 12_345);
        assert!(u.to_string().contains("13,345"));
    }

    #[test]
    fn a_missing_key_names_the_variable() {
        // The message has to say which variable, or a first run stalls on guesswork.
        let e = ensure_provider("nosuchprovider/model").unwrap_err().to_string();
        assert!(e.contains("ANTHROPIC_API_KEY"), "{e}");
    }
}
