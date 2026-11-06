//! Analyze ICIJ's Offshore Leaks database with an agent that writes and runs its own SQL and
//! Python, in a sandbox, against the data it is given as context.
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
//! file in `context/` beside this file, which the agent queries in place.
//!
//! Three volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-offshore-leaks-skill` at `/skills/offshore-leaks`, read-only — the skill
//!   (`SKILL.md`, `oldb.py` from `../shared/`), put there each run. It is what the agent knows
//!   of the data before it looks: the tables, which way a relationship points, and what a
//!   match on a name does not show.
//! * `ailoy-offshore-leaks-context` at `/context`, read-only — `context/` beside this file,
//!   every file of it, put there each run: `offshore_leaks.duckdb`, and whatever else the
//!   request is about, such as a list of names to look for.
//! * `ailoy-offshore-leaks-artifacts` at `/artifacts`, writable — where the reports, tables and
//!   charts go. It stays on the server between runs, and is copied to `artifacts/` beside this
//!   file at the end.
//!
//! The agent's code runs in the sandbox, which sees nothing of the host but these three
//! volumes.
//!
//! The data is ICIJ's, under the Open Database License, and its contents under CC BY-SA.
//! Being in it is not evidence of wrongdoing, as ICIJ says and the skill tells the agent.
//!
//! Environment, also read from `.env`:
//!
//! * `OFFSHORE_LEAKS_URL` — the archive `prepare_data.py` downloads; ICIJ's latest by default.
//! * `UV` — the `uv` binary `prepare_data.py` runs with; `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to
//!   be set up as `docs/src/content/docs/guide/model-providers.mdx` says.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    message::{Message, Part, Role},
};
use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use virtx::{Client, Files as _, ImageJobEntry, Recipe, SandboxCreateArgs, SandboxVolume};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-offshore-leaks:1";

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

    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("offshore_leaks/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("offshore_leaks/rust");
    for dir in ["context", "artifacts"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    prepare(&shared_path, &project_path).await?;

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once: Python, and what the agent's code reads the data and draws with.
    // The server answers a tag in its full form, as `docker.io/library/ailoy-offshore-leaks:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        let recipe = Recipe::new("python:3.12-slim-trixie").step(
            // The DuckDB `prepare_data.py` writes the file with (pinned in pyproject.toml); an
            // older one may not read it.
            "pip install --no-cache-dir duckdb==1.5.5 pandas matplotlib networkx",
        );
        let mut log = client.build_image(recipe, IMAGE).log();
        while let Some(entry) = log.next().await {
            if let ImageJobEntry::End(end) = entry?.entry
                && let Some(error) = end.error
            {
                bail!("building {IMAGE}: {error}");
            }
        }
    }

    // The volumes, made the first time and kept on the server: `409` says one is there already.
    let mut volumes = Vec::new();
    for name in [
        "ailoy-offshore-leaks-skill",
        "ailoy-offshore-leaks-context",
        "ailoy-offshore-leaks-artifacts",
    ] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [skill, context, artifacts] = <[_; 3]>::try_from(volumes).unwrap();

    // The skill onto the first.
    for name in ["SKILL.md", "oldb.py"] {
        skill
            .write_file(name, std::fs::read(shared_path.join(name))?, None)
            .await?;
    }

    // `context/`, every file of it, onto the second.
    let mut dirs = vec![(project_path.join("context"), String::new())];
    while let Some((dir, at)) = dirs.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let to = format!("{at}{}", entry.file_name().to_string_lossy());
            if entry.file_type()?.is_dir() {
                context.create_dir_all(to.clone()).await?;
                dirs.push((entry.path(), format!("{to}/")));
            } else {
                context
                    .write_file(to, std::fs::read(entry.path())?, None)
                    .await?;
            }
        }
    }

    let mut args = SandboxCreateArgs::new(IMAGE);
    args.volumes = vec![
        SandboxVolume {
            volume: skill.name().to_string(),
            path: "/skills/offshore-leaks".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: context.name().to_string(),
            path: "/context".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: artifacts.name().to_string(),
            path: "/artifacts".to_string(),
            readonly: false,
        },
    ];
    args.network = false;
    args.vcpus = Some(2);
    args.memory_mib = Some(2048);
    let sandbox = client.create_sandbox(&args).await?;

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
    .system_tools()
    .sandbox(sandbox)
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

    // The artifacts, every file of them, to `artifacts/` here.
    let mut dirs = vec![String::new()];
    while let Some(at) = dirs.pop() {
        std::fs::create_dir_all(project_path.join("artifacts").join(&at))?;
        for entry in artifacts.read_dir(at.clone()).await? {
            let from = format!("{at}{}", entry.name);
            if entry.metadata.is_dir {
                dirs.push(format!("{from}/"));
            } else {
                let bytes = artifacts.read_file(from.clone(), 0, None).await?;
                std::fs::write(project_path.join("artifacts").join(&from), bytes)?;
            }
        }
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
