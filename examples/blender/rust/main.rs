//! Build 3D scenes in a live Blender that you watch over VNC, with an agent that drives it,
//! renders what it built, looks at it and fixes it.
//!
//! ```sh
//! cargo run --example blender
//! cargo run --example blender -- "A low-poly fox sitting on a mossy rock, for a game"
//! cargo run --example blender -- "A product shot of a matte black espresso cup on a saucer"
//! ```
//!
//! Blender 5.2 runs on a VNC display in a console of its own, published here at
//! `localhost:5901` (password `ailoy`): open `vnc://localhost:5901` in macOS Screen Sharing, or
//! any VNC viewer, and watch the scene being built. The skill's `server.py` runs inside that
//! Blender and takes Python at port 8765, also published here. The agent works in a second
//! console and sends its scripts there with `run.py`, through its `shell` tool; each object it
//! adds shows up in the viewport as it goes. To see what it made, it renders the scene — flat
//! views from four sides, or EEVEE through the camera — and reads the pictures with its
//! `imgread` tool.
//!
//! **Two consoles**: the agent's console is stopped between its tool calls, and Blender with
//! it would be, so Blender has a console the agent does not hold. Every console reaches a
//! published port at `127.0.0.1`, as a program here does, so `run.py` finds Blender there.
//!
//! Blender draws with Vulkan. Where the host GPU has what Blender asks of a device it renders
//! on it; elsewhere, as on a Mac, on Mesa's llvmpipe in the guest's CPUs, slower but the same.
//!
//! The skill (`SKILL.md`, `server.py`, `run.py`, `start.sh`) is mounted from memory at
//! `/skills/blender`, and these at the same paths, in both consoles:
//!
//! * `context/` at `/context`, read-only — what the request is about, when it is not in the
//!   prompt: a reference picture, a concept sketch, a `.blend` to start from.
//! * `artifacts/` at `/artifacts`, writable — the scripts, `scene.blend` and `scene.glb`, the
//!   renders and the turntable.
//!
//! Environment, also read from `.env`:
//!
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to
//!   be set up as `docs/guide/model-providers.md` says. It has to take images, or it cannot see
//!   what it built.

use std::{io::Write as _, path::Path, sync::Arc};

use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{FinishReason, Message, Part, Role},
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
use virtx::{fs::Directory, image::Recipe, protocol::Port};

/// The request when none is given.
const QUERY: &str = "Build a small isometric diorama of a ramen stall at night: a square base \
    about 4 m across, the stall with a counter, three stools and a noren curtain over the front, \
    paper lanterns that glow, a steaming bowl on the counter and a vending machine beside it. \
    Warm light from inside the stall, cool moonlight from above. Frame it from the front \
    three-quarters, render it, and give me a turntable.";

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
    // What a run reads and writes, beside this file.
    let project_path = examples.join("blender/rust");
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    // The console server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none.
    virtx::ensure_virtx().await?;
    // One mount, shared: both consoles see the same skill.
    let skill = Arc::new(
        HostMount::try_new(
            Directory::new()
                .with_file("SKILL.md", include_str!("../shared/SKILL.md").as_bytes())?
                .with_file("server.py", include_str!("../shared/server.py").as_bytes())?
                .with_file("run.py", include_str!("../shared/run.py").as_bytes())?
                .with_file("start.sh", include_str!("../shared/start.sh").as_bytes())?,
            &project_path.join("skill"),
        )
        .with_context(|| "mounting the skill")?,
    );
    let mut blender = ConsoleClient::builder()
        .image(
            Recipe::new("debian:sid-slim")
                // Blender 5.2 is in sid for every architecture. `mesa-vulkan-drivers` carries the
                // guest's venus ICD and llvmpipe, `libvulkan1` the loader; Xvnc is the display.
                .step(
                    "apt-get update && apt-get install -y --no-install-recommends \
                    blender python3-numpy mesa-vulkan-drivers libvulkan1 \
                    tigervnc-standalone-server tigervnc-tools && rm -rf /var/lib/apt/lists/*",
                )
                // Saved preferences skip the first-run splash, and renders open no window.
                .step(
                    "blender -b --factory-startup --python-expr \"import bpy; \
                    v = bpy.context.preferences.view; v.show_splash = False; \
                    v.render_display_type = 'NONE'; bpy.ops.wm.save_userpref()\"",
                ),
        )
        .mount_readonly(skill.clone(), "/skills/blender")
        .mount_readonly(project_path.join("context"), "/context")
        .mount(project_path.join("artifacts"), "/artifacts")
        // What comes in is the viewer, at 5901 here to leave 5900 to macOS Screen Sharing, and
        // `run.py` from the agent's console.
        .network(true)
        .ports([Port::new(5901, 5900)?, Port::new(8765, 8765)?])
        .gpu(true)
        .vcpus(4)
        .memory_mib(8192)
        .gpu_memory_mib(8192)
        .build()
        .await
        .with_context(|| "starting Blender's console")?;

    // Blender up before the agent's first turn, so the person can watch from the start.
    let started = blender
        .exec(["sh", "/skills/blender/start.sh"], Some(180_000))
        .await?;
    anyhow::ensure!(
        started.code == 0,
        "starting Blender: {}",
        String::from_utf8_lossy(&started.stderr)
    );
    println!("Blender is up: open vnc://localhost:5901 (password: ailoy) to watch.\n");

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "openai/gpt-6-astra".to_string()),
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
    // Python for `run.py`, and a network to reach the port Blender's console published.
    .console(
        ConsoleClient::builder()
            .image(Recipe::new("python:3.12-slim-trixie"))
            .mount_readonly(skill, "/skills/blender")
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            .network(true)
            .build()
            .await
            .with_context(|| "starting the agent's console")?,
    )
    .skill("/skills/blender")
    .build()
    .await?;

    let query = Message::new(Role::User).with_contents([Part::text(prompt)]);
    let mut stream = agent.run(query);
    while let Some(output) = stream.next().await {
        let output = output?;
        // A token-limit cutoff also ends the run; without this it ends silently.
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
                        println!("→ {name} {}", serde_json::to_string_pretty(args)?);
                    }
                }
            }
            // A `read` of a picture is an image part: its bytes are no use on a terminal.
            Role::Tool => {
                for part in &message.contents {
                    if part.is_image() {
                        println!("← [image]");
                    } else {
                        println!("← {}", serde_json::to_string_pretty(part)?);
                    }
                }
            }
            _ => {}
        }
        std::io::stdout().flush()?;
    }
    drop(stream);

    // Blender's console, and Blender in it, goes when this returns.
    println!("\nBlender stays open at vnc://localhost:5901. Press Enter to close it.");
    std::io::stdin().read_line(&mut String::new())?;
    Ok(())
}
