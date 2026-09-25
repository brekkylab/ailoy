//! Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
//! Python, in a console, against the data it is given as context.
//!
//! ```sh
//! cargo run --example offshore_leaks
//! cargo run --example offshore_leaks -- "Which South Korean officers appear in more than one leak?"
//! cargo run --example offshore_leaks -- "Map who is behind the entities Mossack Fonseca set up in Niue"
//! ```
//!
//! The [Offshore Leaks database](https://offshoreleaks.icij.org) is the graph ICIJ published
//! from the Offshore Leaks, the Panama, Paradise and Pandora Papers and the Bahamas Leaks:
//! about 2 million offshore companies, people, intermediaries and addresses, and the 3.3
//! million relationships between them. `prepare_data.py` downloads ICIJ's CSVs and loads them
//! into one DuckDB file, which the agent queries in place.
//!
//! The skill is `SKILL.md` and `oldb.py`, mounted at `/skills/offshore-leaks` from memory. It
//! is what the agent knows of the data before it looks: the tables, which way a relationship
//! points, and what a match on a name does not show. For more than a query, the agent writes
//! Python of its own and runs it with its `shell` tool.
//!
//! * `context/` at `/context`, read-only — `offshore_leaks.duckdb`, and whatever else the
//!   request is about, such as a list of names to look for.
//! * `artifacts/` at `/artifacts`, writable — where the reports, tables and charts go.
//!
//! What the agent runs is its own code over 2 million records, and the console is where that
//! is safe to do: it sees the two directories and nothing else of the host, and the database
//! is read-only to it.
//!
//! The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA.
//! Being in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.
//!
//! Environment:
//!
//! * `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads, ICIJ's latest by default.
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//! * `UV` — the `uv` binary `prepare_data.py` runs with, `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//!
//! Read from `.env` as well.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::Console,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
use cortex::{console::NetworkAccess, fs::Directory, image::Image};
// One host binding per platform, each mounting on `try_new` and unmounting on `Drop`, so
// the tree below is written once. Three arms and not `not(windows)` because the guards are
// three distinct types: cortex's default `mount` feature compiles the one binding its target
// has — Dokany on Windows, `fuser` on Linux, FUSE-T on macOS — and names the guard after it.
#[cfg(windows)]
use cortex::fs::DokanMount as HostMount;
#[cfg(target_os = "linux")]
use cortex::fs::FuseMount as HostMount;
#[cfg(target_os = "macos")]
use cortex::fs::FuseTMount as HostMount;
use futures::StreamExt as _;

/// The request when none is given.
const QUERY: &str = "Which intermediaries set up the most entities in the Panama Papers, and \
    in which jurisdictions? Write a short report with a chart.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        QUERY.to_string()
    } else {
        prompt
    };

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/offshore_leaks");
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    prepare(&project_path).await?;

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "openai/gpt-6-astra".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: /context\n\n",
        "Holds what you were given to work on: `offshore_leaks.duckdb`, ICIJ's Offshore Leaks ",
        "database, and anything else the request is about, such as a list of names to look ",
        "for. List it before you start. This folder is read-only.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the files the user asks for go, such as a report, a table of what you found or ",
        "a chart. Write them here and name them so the user can tell what they are. A result ",
        "that is only in your reply is not delivered as a file.",
    ))
    .system_tools()
    .console(
        Console::builder()
            .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(Image::new().base("python:3.12-slim-trixie").step(
                // The DuckDB `prepare_data.py` wrote the file with, in pyproject.toml: an
                // older one may not read it.
                "pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx",
            ))
            .mount_readonly(
                HostMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("oldb.py", include_str!("oldb.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/offshore-leaks",
            )
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            .network(NetworkAccess::none())
            .vcpus(2)
            .memory_mib(2048)
            .build()
            .await
            .with_context(|| "starting the console")?,
    )
    .skill("/skills/offshore-leaks")
    .build()
    .await?;

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
                        println!("→ {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            Role::Tool => {
                for part in &message.contents {
                    println!("← {}", serde_json::to_string_pretty(part)?);
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }

    Ok(())
}

/// Download the database and load it into `project/context`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_data.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the data: {status}");
    Ok(())
}
