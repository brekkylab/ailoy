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
//! million relationships between them. `prepare_data.py` loads ICIJ's CSVs into one DuckDB
//! file, which the agent queries in place.
//!
//! The skill (`SKILL.md`, `oldb.py`) is mounted from memory at `/skills/offshore-leaks`. It
//! is what the agent knows of the data before it looks: the tables, which way a relationship
//! points, and what a match on a name does not show.
//!
//! * `context/` at `/context`, read-only — `offshore_leaks.duckdb`, and whatever else the
//!   request is about, such as a list of names to look for.
//! * `artifacts/` at `/artifacts`, writable — where the reports, tables and charts go.
//!
//! The agent's code runs in the console, which sees nothing of the host but these two
//! directories.
//!
//! The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA.
//! Being in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.
//!
//! Environment, also read from `.env`:
//!
//! * `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
//! * `UV` — the `uv` binary `prepare_data.py` runs with; `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's API
//!   key has to be set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
// One host binding per platform (mounts on `try_new`, unmounts on `Drop`), so the tree
// below is written once. Three arms, not `not(windows)`: virtx's default `mount` feature
// compiles only its target's binding, each a distinct guard type.
use futures::StreamExt as _;
#[cfg(windows)]
use virtx::fs::DokanMount as HostMount;
#[cfg(target_os = "linux")]
use virtx::fs::FuseMount as HostMount;
#[cfg(target_os = "macos")]
use virtx::fs::FuseTMount as HostMount;
use virtx::{fs::Directory, image::Recipe};

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
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("offshore_leaks/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("offshore_leaks/rust");
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    prepare(&shared_path, &project_path).await?;

    // The console server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none.
    virtx::ensure_virtx().await?;
    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL")
            .unwrap_or_else(|_| "openai/gpt-6-astra".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: `/context`\n\n",
        "This folder holds the data and context the user wants to share with you. ",
        "When the user refers to something whose context you cannot figure out, the files in this folder might help. ",
        "The information that settles the answer may be here too, and so may hints toward it, so look through this folder for them.\n\n",
        "# Artifacts\n\n",
        "Path: `/artifacts`\n\n",
        "This folder is where what the user asked for goes. ",
        "Write every result here, such as a report, a figure, or the file the user came for. ",
        "Everything in this folder is collected and handed back to the user, and a result left anywhere else is not delivered.",
    ))
    .max_tokens(64000)
    .system_tools()
    .console(
        ConsoleClient::builder()
            .image(Recipe::new("python:3.12-slim-trixie").step(
                // The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml); an
                // older one may not read it.
                "pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx",
            ))
            .mount_readonly(
                HostMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("../shared/SKILL.md").as_bytes())?
                        .with_file("oldb.py", include_str!("../shared/oldb.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/offshore-leaks",
            )
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            .network(false)
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
async fn prepare(shared: &Path, project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_data.py"])
        .arg(project.join("data"))
        .arg(project.join("context/offshore_leaks.duckdb"))
        .current_dir(shared)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the data: {status}");
    Ok(())
}
