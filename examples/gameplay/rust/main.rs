//! An agent that plays OpenTTD, running in a sandbox, while you watch it over VNC.
//!
//! ```sh
//! cargo run --example gameplay
//! cargo run --example gameplay -- "Connect every town over 1000 people by air"
//! ```
//!
//! [OpenTTD](https://www.openttd.org) is the open-source game after Transport Tycoon Deluxe,
//! and with OpenGFX, OpenSFX and OpenMSX, everything it needs is free: there is nothing to
//! bring. The sandbox's image installs it all from Debian, and the sandbox runs the game as a
//! dedicated server, in which the agent runs a company: it looks at towns and industries, builds stations, roads
//! and airports, buys vehicles and lets time pass, and sees from the company's books what
//! earns.
//!
//! **One sandbox**: the game and the agent's commands run side by side in it, the game as one
//! job that runs for as long as it does.
//!
//! **How it plays**: the skill's `ttd.py` speaks to the server's admin port, and the
//! `AiloyBridge` Game Script inside the game takes what it says as commands and carries them
//! out for the company. The script finds roads and places for stations itself, so the agent
//! decides what to connect and not which tile each piece goes on. The company is opened by
//! the `AiloyCompany` AI, which does nothing else: an AI is the one way to open a company
//! on a dedicated server. The game stands paused between the agent's commands, and runs
//! while one is carried out and while the agent waits.
//!
//! The agent plays in rounds: each ends when it stops to answer, and the next begins with
//! the date, until the game has run for `OPENTTD_YEARS` years. Older command output drops out
//! of its context as it goes, and its notes in `artifacts/notes.md` carry what it knows.
//!
//! **Ports**: `ttd.py` reaches the game's admin port and `shot.py`, which takes screenshots,
//! at `127.0.0.1` in the sandbox. Only the VNC server's port is opened, and forwarded to an
//! address on this machine's loopback, which the program prints.
//!
//! **Watching**: open that address as `vnc://127.0.0.1:PORT` (Screen Sharing on macOS, or any
//! VNC viewer) with the password `openttd`: a client of the
//! game there spectates, and its view follows what the agent builds. You can start a company
//! of your own from its menu and play against the agent.
//!
//! Two volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-gameplay` at `/skills/openttd`, read-only — `../shared/`, put there each run: the
//!   skill (`SKILL.md`, `ttd.py`, `admin.py`) and, in `server/`, `start.sh`, the Game Script,
//!   the AI and `shot.py`.
//! * `ailoy-gameplay-artifacts` at `/artifacts`, writable — the agent's notes, the screenshots
//!   it took, the saved games in `user/openttd/save/` (the last as `ailoy-final.sav`), and the
//!   logs. It stays on the server between runs, and is copied to `artifacts/` beside this file
//!   at the end.
//!
//! Environment, also read from `.env`:
//!
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its org's backend
//!   has to be set up as `docs/src/content/docs/guide/model-providers.mdx` says. It reads
//!   screenshots when it looks, so it should take images.
//! * `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game
//!   month takes a minute to pass.
//! * `OPENTTD_SAVE` — a saved game in `user/openttd/save/` of the artifacts to go on with, such
//!   as `ailoy-final.sav`, rather than a new game.
//! * `OPENTTD_SEED`, `OPENTTD_YEAR`, `OPENTTD_MAP_X`, `OPENTTD_MAP_Y` — the new game's random
//!   seed, its first year (`1950`), and its size as powers of two (`8`, 256 tiles).
//! * `OPENTTD_SIZE` — the viewer's display, `1280x720` by default.
//! * `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::{Agent, AgentBuilder, ContextManager},
    message::{FinishReason, Message, Part, Role},
    tool::exec,
};
use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use virtx::{
    Client, Files as _, ImageJobEntry, PortForwarder, Recipe, SandboxCreateArgs, SandboxExecArgs,
    SandboxJobEntry, SandboxVolume,
};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-gameplay:1";

/// The VNC server's port in the sandbox.
const VNC_PORT: u16 = 5900;

/// The goal when none is given.
const GOAL: &str = "Make Ailoy Transport as valuable as you can: build routes that earn, \
    grow the ones that work, and have the loan paid back by the end if you can.";

/// Rounds at most, so that an agent that answers at once each time does not go on forever.
const MAX_ROUNDS: usize = 200;

/// The settings `start.sh` reads, passed on to it when they are set here.
const GAME_SETTINGS: [&str; 5] = [
    "OPENTTD_SEED",
    "OPENTTD_YEAR",
    "OPENTTD_MAP_X",
    "OPENTTD_MAP_Y",
    "OPENTTD_ADMIN_PASSWORD",
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let goal = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let goal = if goal.is_empty() {
        GOAL.to_string()
    } else {
        goal
    };
    let years: i32 = std::env::var("OPENTTD_YEARS")
        .ok()
        .map(|y| y.parse())
        .transpose()
        .with_context(|| "OPENTTD_YEARS is a number of years")?
        .unwrap_or(3);
    let size = std::env::var("OPENTTD_SIZE").unwrap_or_else(|_| "1280x720".to_string());
    let password = std::env::var("OPENTTD_VNC_PASSWORD").unwrap_or_else(|_| "openttd".to_string());
    let save = std::env::var("OPENTTD_SAVE").unwrap_or_default();
    let here = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/gameplay");

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once: the game, and the free graphics, sounds and music it needs, a
    // display for the viewer, Python for `ttd.py`, and ImageMagick for the agent's screenshots.
    // The server answers a tag in its full form, as `docker.io/library/ailoy-gameplay:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        let recipe = Recipe::new("debian:trixie-slim").step(
            "apt-get update \
            && apt-get install -y --no-install-recommends \
                openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 \
                imagemagick \
            && rm -rf /var/lib/apt/lists/*",
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
    for name in ["ailoy-gameplay", "ailoy-gameplay-artifacts"] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [files, artifacts] = <[_; 2]>::try_from(volumes).unwrap();

    // `shared/`, every file of it, onto the first.
    let mut dirs = vec![(here.join("shared"), String::new())];
    while let Some((dir, at)) = dirs.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let to = format!("{at}{}", entry.file_name().to_string_lossy());
            if entry.file_type()?.is_dir() {
                files.create_dir_all(to.clone()).await?;
                dirs.push((entry.path(), format!("{to}/")));
            } else {
                files
                    .write_file(to, std::fs::read(entry.path())?, None)
                    .await?;
            }
        }
    }

    let mut args = SandboxCreateArgs::new(IMAGE);
    args.volumes = vec![
        SandboxVolume {
            volume: files.name().to_string(),
            path: "/skills/openttd".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: artifacts.name().to_string(),
            path: "/artifacts".to_string(),
            readonly: false,
        },
    ];
    args.ports = vec![VNC_PORT];
    args.vcpus = Some(2);
    args.memory_mib = Some(2048);
    let sandbox = client.create_sandbox(&args).await?;

    // An exec takes no environment of its own, so the settings go through `env`.
    let mut start = vec!["env".to_string()];
    for name in GAME_SETTINGS {
        if let Ok(value) = std::env::var(name) {
            start.push(format!("{name}={value}"));
        }
    }
    start.extend([
        "sh".to_string(),
        "/skills/openttd/server/start.sh".to_string(),
        size,
        password.clone(),
        save,
    ]);
    // The game is one job, running for as long as the game does; `start.sh` says `ready` once
    // it is up. Leaving its log leaves the job running.
    let mut game = sandbox.exec(&SandboxExecArgs::new(start)).log();
    while let Some(entry) = game.next().await {
        match entry?.entry {
            SandboxJobEntry::Stdout(line) if line == "ready" => break,
            SandboxJobEntry::Stdout(line) | SandboxJobEntry::Stderr(line) => println!("{line}"),
            SandboxJobEntry::End(end) => bail!("OpenTTD did not start (exit {})", end.code),
            SandboxJobEntry::Truncated {} => {}
        }
    }
    let viewer = PortForwarder::open(&client, &sandbox.info().ports[0].url)?;
    println!(
        "OpenTTD is running. Watch it at vnc://{} (password: {password}).",
        viewer.address()
    );

    let mut date = game_date(&sandbox).await?;
    let until = year_of(&date)? + years;

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "You play OpenTTD: a game is running in your sandbox, and one company in it is yours. ",
        "Read the openttd skill before anything else, and play with its `ttd.py`.\n\n",
        "You play in rounds. In each, look at how the company is doing, decide what to build or ",
        "change, do it, and let time pass with `wait`, a month or two at a time, checking the ",
        "report each time. End a round with a few lines on what you did and how it is going; the ",
        "next begins with the date.\n\n",
        "# Artifacts\n\n",
        "Path: `/artifacts`\n\n",
        "Keep your notes in `/artifacts/notes.md`: the routes, their stations, depots and ",
        "vehicles, what each earns, and your plan. Update them as you go. Command output from ",
        "earlier rounds drops out of what you remember; the notes do not.",
    ))
    .sandbox(sandbox.clone())
    .skill("/skills/openttd")
    // Many commands a round, each with its output: keep the last two rounds' results whole,
    // and of the screenshots the agent reads, the latest ten once more than thirty have
    // built up.
    .context_manager(ContextManager {
        max_input_tokens: 80_000,
        preserve_recent_turns: 2,
        ..Default::default()
    })
    .build()
    .await?;

    let mut prompt = format!(
        "It is {date}, and the game is yours until {until}-01-01. {goal}\n\n\
        Start by reading the skill, then look at the company, the largest towns and the \
        industries, and build a first route that will earn."
    );
    for round in 1..=MAX_ROUNDS {
        println!("\n=== Round {round}: {date} ===\n");
        play(&mut agent, &prompt).await?;

        date = game_date(&sandbox).await?;
        if year_of(&date)? >= until {
            println!("\n=== The game has reached {date} ===");
            break;
        }
        prompt = format!(
            "It is {date}; the game is yours until {until}-01-01. Go on: read your notes, check \
            the report, then fix what needs fixing, grow what earns, and let time pass."
        );
    }

    let out = exec(
        &sandbox,
        ["python3", "/skills/openttd/ttd.py", "save", "ailoy-final"],
        Some(60_000),
    )
    .await?;
    print!("{}", out.stdout);

    // The artifacts, every file of them, to `artifacts/` here.
    let mut dirs = vec![String::new()];
    while let Some(at) = dirs.pop() {
        std::fs::create_dir_all(here.join("rust/artifacts").join(&at))?;
        for entry in artifacts.read_dir(at.clone()).await? {
            let from = format!("{at}{}", entry.name);
            if entry.metadata.is_dir {
                dirs.push(format!("{from}/"));
            } else {
                let bytes = artifacts.read_file(from.clone(), 0, None).await?;
                std::fs::write(here.join("rust/artifacts").join(&from), bytes)?;
            }
        }
    }
    println!("The game is still there to look at. Press Enter to stop.");
    tokio::task::spawn_blocking(|| std::io::stdin().read_line(&mut String::new())).await??;
    Ok(())
}

/// Run one round of the agent, printing what it says and does.
async fn play(agent: &mut Agent, prompt: &str) -> anyhow::Result<()> {
    let query = Message::new(Role::User).with_contents([Part::text(prompt)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let output = output?;
        // The run ends on any other reason as well, and without this it ends in silence.
        if matches!(output.finish_reason, FinishReason::Length {}) {
            eprintln!("(the reply was cut off at the token limit)");
        }
        let message = output.message;
        match message.role {
            Role::Assistant => {
                for text in message.contents.iter().filter_map(Part::as_text) {
                    println!("{text}");
                }
                for call in message.tool_calls.iter().flatten() {
                    if let Some((_, name, args)) = call.as_function() {
                        println!("→ {name} {}", serde_json::to_string(args)?);
                    }
                }
            }
            // A screenshot is an image part: its bytes are no use on a terminal. And the
            // rest only as much as shows what came back.
            Role::Tool => {
                for part in &message.contents {
                    if part.is_image() {
                        println!("← [image]");
                    } else {
                        let text = serde_json::to_string(part)?;
                        let cut: String = text.chars().take(600).collect();
                        let more = if cut.len() < text.len() { " …" } else { "" };
                        println!("← {cut}{more}");
                    }
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }
    Ok(())
}

/// The game's date, as `YYYY-MM-DD`.
async fn game_date(sandbox: &virtx::Sandbox) -> anyhow::Result<String> {
    let out = exec(
        sandbox,
        ["python3", "/skills/openttd/ttd.py", "--json", "status"],
        Some(60_000),
    )
    .await?;
    if out.code != 0 {
        bail!("asking the game its date:\n{}", out.stderr);
    }
    let status: serde_json::Value =
        serde_json::from_str(&out.stdout).with_context(|| "reading the game's status")?;
    status["date"]
        .as_str()
        .map(str::to_string)
        .context("the game's status has no date")
}

fn year_of(date: &str) -> anyhow::Result<i32> {
    date.split('-')
        .next()
        .and_then(|y| y.parse().ok())
        .with_context(|| format!("{date} is not a date"))
}
