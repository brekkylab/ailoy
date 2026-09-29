//! An agent that plays OpenTTD, running in a console, while you watch it over VNC.
//!
//! ```sh
//! cargo run --example openttd
//! cargo run --example openttd -- "Connect every town over 1000 people by air"
//! ```
//!
//! [OpenTTD](https://www.openttd.org) is the open-source game after Transport Tycoon Deluxe,
//! and with OpenGFX, OpenSFX and OpenMSX, everything it needs is free: there is nothing to
//! bring. A console installs it all from Debian and runs the game as a dedicated server, in
//! which the agent runs a company: it looks at towns and industries, builds stations, roads
//! and airports, buys vehicles and lets time pass, and sees from the company's books what
//! earns.
//!
//! **Two consoles**: the game runs in one of its own, and the agent in another. An agent
//! stops its console after each batch of tool calls, which would stop the game with it, so
//! the game's console is this program's and runs until the end.
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
//! **Tunnels**: nothing can connect into a console, so the game's `tunnel.py` connects out to
//! this program, once for each port in there something here reaches in to, and this joins
//! each to a port here: the VNC server to `localhost:5901`, and the admin port and
//! `shot.py`, which takes screenshots, to two ports the agent's console is granted.
//!
//! **Watching**: open `vnc://localhost:5901`
//! (Screen Sharing on macOS, or any VNC viewer) with the password `openttd`: a client of the
//! game there spectates, and its view follows what the agent builds. You can start a company
//! of your own from its menu and play against the agent.
//!
//! * `server/` at `/example` in the game's console, read-only — `start.sh`, the Game Script,
//!   the AI, `shot.py` and `tunnel.py`.
//! * `skill/` at `/skills/openttd` in both, read-only — `SKILL.md`, `ttd.py` and the admin
//!   port.
//! * `artifacts/` at `/artifacts` in both, writable — the agent's notes, the screenshots it
//!   took, the saved games in `user/openttd/save/` (the last as `ailoy-final.sav`), and the
//!   logs.
//!
//! Environment:
//!
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). It reads screenshots
//!   when it looks, so it should take images.
//! * `OPENTTD_YEARS` — how many years of the game the agent plays, `3` by default. A game
//!   month takes a minute to pass.
//! * `OPENTTD_SAVE` — a saved game in `artifacts/user/openttd/save/` to go on with, such as
//!   `ailoy-final.sav`, rather than a new game.
//! * `OPENTTD_SEED`, `OPENTTD_YEAR`, `OPENTTD_MAP_X`, `OPENTTD_MAP_Y` — the new game's random
//!   seed, its first year (`1950`), and its size as powers of two (`8`, 256 tiles).
//! * `OPENTTD_SIZE` — the viewer's display, `1280x720` by default.
//! * `OPENTTD_VNC_PASSWORD` — the viewer's password, `openttd` by default.
//!
//! Read from `.env` as well.

use std::{collections::VecDeque, io::Write as _, path::Path, sync::Arc};

use ailoy::{
    agent::{Agent, AgentBuilder, ContextManager},
    console::ConsoleClient,
    message::{FinishReason, Message, Part, Role},
};
use anyhow::{Context as _, bail};
use cortex::{image::Recipe, protocol::NetworkAccess};
use futures::StreamExt as _;
use tokio::{
    io::{AsyncReadExt as _, AsyncWriteExt as _},
    net::{TcpListener, TcpStream},
    sync::Mutex,
};

/// The ports on this machine the game console's tunnels connect to, the ones it is granted:
/// for its VNC server, its admin port and its screenshots.
const VNC_TUNNEL: u16 = 5500;
const ADMIN_TUNNEL: u16 = 5501;
const SHOT_TUNNEL: u16 = 5502;

/// The port a VNC viewer connects to here.
const VIEWER_PORT: u16 = 5901;

/// The ports the agent's console is granted, joined to the game's admin port and to its
/// screenshots. `admin.py` names the same.
const ADMIN_PORT: u16 = 5511;
const SHOT_PORT: u16 = 5512;

/// What the console sees this machine as: its gateway, which opens a granted port on
/// loopback here.
const HOST_FROM_CONSOLE: &str = "10.0.2.1";

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

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/openttd");
    let artifacts = project_path.join("artifacts");
    std::fs::create_dir_all(&artifacts)
        .with_context(|| format!("creating {}", artifacts.display()))?;

    // Bound before the consoles start, so the tunnels have somewhere to connect to at once.
    for (from_game, to) in [
        (VNC_TUNNEL, VIEWER_PORT),
        (ADMIN_TUNNEL, ADMIN_PORT),
        (SHOT_TUNNEL, SHOT_PORT),
    ] {
        let from_console = TcpListener::bind(("127.0.0.1", from_game))
            .await
            .with_context(|| format!("listening on {from_game} for the game's console"))?;
        let clients = TcpListener::bind(("127.0.0.1", to))
            .await
            .with_context(|| format!("listening on {to}"))?;
        tokio::spawn(tunnel(from_console, clients));
    }

    let mut game = ConsoleClient::builder()
        .image(
            // All of it in `main`: the game, and the free graphics, sounds and music it needs,
            // a display for the viewer, and ImageMagick for the agent's screenshots of it.
            Recipe::new("debian:trixie-slim").step(
                "apt-get update \
                && apt-get install -y --no-install-recommends \
                    openttd openttd-opengfx openttd-opensfx openttd-openmsx xvfb x11vnc python3 \
                    imagemagick \
                && rm -rf /var/lib/apt/lists/*",
            ),
        )
        .mount_readonly(project_path.join("server"), "/example")
        .mount_readonly(project_path.join("skill"), "/skills/openttd")
        .mount(artifacts.clone(), "/artifacts")
        // Nothing outside, and of this machine only the tunnels' ports.
        .network(NetworkAccess::host().with_host_ports([VNC_TUNNEL, ADMIN_TUNNEL, SHOT_TUNNEL]))
        .vcpus(2)
        .memory_mib(2048)
        .build()
        .await
        .with_context(|| "starting the game's console")?;

    // An exec takes no environment of its own, so the settings go through `env`.
    let mut start = vec!["env".to_string()];
    for name in GAME_SETTINGS {
        if let Ok(value) = std::env::var(name) {
            start.push(format!("{name}={value}"));
        }
    }
    start.extend([
        "sh".to_string(),
        "/example/start.sh".to_string(),
        size,
        password.clone(),
        format!("{HOST_FROM_CONSOLE}:{VNC_TUNNEL}"),
        format!("{HOST_FROM_CONSOLE}:{ADMIN_TUNNEL}"),
        format!("{HOST_FROM_CONSOLE}:{SHOT_TUNNEL}"),
        save,
    ]);
    let out = game.exec(&start, Some(180_000)).await?;
    print!("{}", String::from_utf8_lossy(&out.stdout));
    if out.code != 0 {
        bail!(
            "starting OpenTTD:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    println!("OpenTTD is running. Watch it at vnc://localhost:{VIEWER_PORT} (password: {password}).");

    let mut date = game_date(&mut game).await?;
    let until = year_of(&date)? + years;

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "You play OpenTTD: a game is running in the console, and one company in it is yours. ",
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
    .max_tokens(32000)
    .system_tools()
    // Python for `ttd.py`, and of this machine only the ports joined to the game.
    .console(
        ConsoleClient::builder()
            .image(Recipe::new("python:3.12-slim-trixie"))
            .mount_readonly(project_path.join("skill"), "/skills/openttd")
            .mount(artifacts, "/artifacts")
            .network(NetworkAccess::host().with_host_ports([ADMIN_PORT, SHOT_PORT]))
            .build()
            .await
            .with_context(|| "starting the agent's console")?,
    )
    .skill("/skills/openttd")
    // Many commands a round, each with its output: keep the last two rounds whole.
    .context_manager(ContextManager {
        max_input_tokens: 80_000,
        preserve_recent_turns: 2,
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

        date = game_date(&mut game).await?;
        if year_of(&date)? >= until {
            println!("\n=== The game has reached {date} ===");
            break;
        }
        prompt = format!(
            "It is {date}; the game is yours until {until}-01-01. Go on: read your notes, check \
            the report, then fix what needs fixing, grow what earns, and let time pass."
        );
    }

    let out = game
        .exec(
            ["python3", "/skills/openttd/ttd.py", "save", "ailoy-final"],
            Some(60_000),
        )
        .await?;
    print!("{}", String::from_utf8_lossy(&out.stdout));
    println!("The game is still there to look at. Press Enter to stop.");
    tokio::task::spawn_blocking(|| std::io::stdin().read_line(&mut String::new())).await??;

    // Dropping the consoles tears them down, game and all.
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
async fn game_date(console: &mut ConsoleClient) -> anyhow::Result<String> {
    let out = console
        .exec(
            ["python3", "/skills/openttd/ttd.py", "--json", "status"],
            Some(60_000),
        )
        .await?;
    if out.code != 0 {
        bail!(
            "asking the game its date:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let status: serde_json::Value = serde_json::from_slice(&out.stdout)
        .with_context(|| "reading the game's status")?;
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

/// Join viewers here to the VNC server in the console, which no one here can connect to.
///
/// The console's `tunnel.py` keeps one connection open to `from_console`, saying `CTRL`. For
/// each viewer that connects to `viewers`, this writes `OPEN` on that connection, the tunnel
/// connects again saying `DATA` with the server on its other end, and the two are joined.
async fn tunnel(from_console: TcpListener, viewers: TcpListener) {
    #[derive(Default)]
    struct Tunnel {
        ctrl: Option<TcpStream>,
        waiting: VecDeque<TcpStream>,
    }
    let tunnel = Arc::new(Mutex::new(Tunnel::default()));

    let shared = tunnel.clone();
    tokio::spawn(async move {
        while let Ok((viewer, _)) = viewers.accept().await {
            let mut t = shared.lock().await;
            t.waiting.push_back(viewer);
            // With no control connection yet, the viewer waits for one: it asks for them all.
            if let Some(ctrl) = t.ctrl.as_mut()
                && ctrl.write_all(b"OPEN\n").await.is_err()
            {
                t.ctrl = None;
            }
        }
    });

    while let Ok((mut conn, _)) = from_console.accept().await {
        let tunnel = tunnel.clone();
        tokio::spawn(async move {
            let mut hello = [0u8; 5];
            if conn.read_exact(&mut hello).await.is_err() {
                return;
            }
            let mut t = tunnel.lock().await;
            match &hello {
                b"CTRL\n" => {
                    for _ in 0..t.waiting.len() {
                        if conn.write_all(b"OPEN\n").await.is_err() {
                            return;
                        }
                    }
                    t.ctrl = Some(conn);
                }
                b"DATA\n" => {
                    if let Some(mut viewer) = t.waiting.pop_front() {
                        drop(t);
                        let _ = tokio::io::copy_bidirectional(&mut viewer, &mut conn).await;
                    }
                }
                _ => {}
            }
        });
    }
}
