//! Ask questions about someone else's repository with an agent that reads its code and its
//! history, and builds and runs it, in a console.
//!
//! ```sh
//! cargo run --example repo_analysis -- https://github.com/pallets/click "How are nested command groups resolved?"
//! cargo run --example repo_analysis -- https://github.com/tokio-rs/mini-redis "Run the tests and explain any that fail"
//! cargo run --example repo_analysis -- ../some/local/checkout "Draw a map of the modules and what depends on what"
//! ```
//!
//! The first argument is the repository, a URL to clone or a directory on this machine, and
//! the rest is the question; both are required. A URL is cloned on the host into `repos/`,
//! once: a second run on the same URL reuses the clone as it is.
//!
//! There is no skill: reading code and its history is what the agent already knows. The image
//! is Alpine with git, and the agent installs with `apk` whatever else a repository needs.
//!
//! * the repository at `/repo`, read-only — the agent copies it out to build it.
//! * `artifacts/` at `/artifacts`, writable — where the reports and diagrams go.
//!
//! What the agent runs is a stranger's code: its build scripts, its tests and whatever they
//! install. The console is where that is safe to do. It sees the repository, which it cannot
//! change, and `artifacts/`, and nothing else of the host. It does reach the public internet,
//! to install the project's dependencies; `NetworkAccess::none()` makes it read-only analysis.
//!
//! Environment:
//!
//! * `REPO_ANALYSIS_DEPTH` — a depth to clone at, the whole history by default. A shallow
//!   clone is faster for a big repository, and leaves `git log` and `git blame` cut off.
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//!
//! Read from `.env` as well.

use std::{
    io::Write as _,
    path::{Path, PathBuf},
};

use ailoy::{
    agent::AgentBuilder,
    console::Console,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
use cortex::{console::NetworkAccess, image::Image};
use futures::StreamExt as _;

const USAGE: &str = "usage: cargo run --example repo_analysis -- <repository> <question>";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let mut args = std::env::args().skip(1);
    let repo = args.next().context(USAGE)?;
    let prompt = args.collect::<Vec<_>>().join(" ");
    anyhow::ensure!(!prompt.is_empty(), "no question given\n{USAGE}");

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/repo_analysis");
    for dir in ["repos", "artifacts"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let repo_path = prepare(&project_path, &repo).await?;

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "# Repository\n\n",
        "Path: /repo\n\n",
        "The repository the user is asking about, with its git history. It is read-only: to ",
        "build it or run its tests, copy it out first (`cp -a /repo /tmp/work`). The machine ",
        "is Alpine with git and little else, and the network is open: install what the ",
        "repository needs with `apk add`, as its manifests and CI say rather than guessing at ",
        "it. It is someone else's code, so read a script before you run it.\n\n",
        "Cite what an answer rests on as `path/to/file:line`, relative to the repository, ",
        "and say which commit it is about. Keep apart what you read in the code, what you saw ",
        "when you ran it, and what you infer.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the files the user asks for go, such as a report, a map of the modules or a ",
        "diagram. Write them here and name them so the user can tell what they are. A result ",
        "that is only in your reply is not delivered as a file.",
    ))
    .system_tools()
    .console(
        Console::builder()
            .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(
                Image::new()
                    .base("alpine:3.22")
                    .step("apk add --no-cache git")
                    // The repository is the host user's, and git refuses to read one owned
                    // by someone else without this.
                    .step("git config --system --add safe.directory '*'"),
            )
            .mount_readonly(repo_path.clone(), "/repo")
            .mount(project_path.join("artifacts"), "/artifacts")
            // To install what the repository depends on. `none` keeps it to reading.
            .network(NetworkAccess::public())
            .vcpus(4)
            .memory_mib(4096)
            .build()
            .await
            .with_context(|| "starting the console")?,
    )
    .build()
    .await?;

    println!("  repo   {}", repo_path.display());
    println!("  query  {prompt}\n");

    let query = Message::new(Role::User).with_contents([Part::text(prompt)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let message = output?.message;
        match message.role {
            Role::Assistant => {
                for text in message.contents.iter().filter_map(Part::as_text) {
                    println!("{text}");
                }
                for call in message.tool_calls.iter().flatten() {
                    if let Some((_, name, args)) = call.as_function() {
                        println!("  → {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            Role::Tool => {
                for part in &message.contents {
                    println!("  ← {}", serde_json::to_string_pretty(part)?);
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }

    Ok(())
}

/// The repository as a directory on the host, cloned into `project/repos` when it is a URL
async fn prepare(project: &Path, repo: &str) -> anyhow::Result<PathBuf> {
    if Path::new(repo).is_dir() {
        return std::fs::canonicalize(repo).with_context(|| format!("resolving {repo}"));
    }

    // `owner__name`, so that two repositories named alike do not share a clone.
    let name = repo
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .rsplit(['/', ':'])
        .take(2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join("__");
    let dest = project.join("repos").join(&name);
    if dest.join(".git").is_dir() {
        return Ok(dest);
    }

    println!("  clone  {repo}");
    let mut git = tokio::process::Command::new("git");
    git.args(["clone", "--quiet"]);
    if let Ok(depth) = std::env::var("REPO_ANALYSIS_DEPTH") {
        git.args(["--depth", &depth]);
    }
    let status = git
        .arg(repo)
        .arg(&dest)
        .status()
        .await
        .with_context(|| "running `git`. Install git.")?;
    anyhow::ensure!(status.success(), "cloning {repo}: {status}");
    Ok(dest)
}
