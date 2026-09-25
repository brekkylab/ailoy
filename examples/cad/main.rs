//! Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in
//! a console.
//!
//! ```sh
//! cargo run --example cad
//! cargo run --example cad -- "A wall mount for a 60 mm fan, with four M3 screw holes"
//! cargo run --example cad -- "A twisted vase, 150 mm tall, hexagonal at the base and round at the top"
//! ```
//!
//! [CadQuery](https://cadquery.readthedocs.io) is a Python library for parametric CAD on the
//! OpenCascade kernel: a model is a script, and what it builds is exact solids, not meshes. The
//! agent writes that script, and the skill's `render.py` runs it, checks each part — valid,
//! one solid, closed, not overlapping another — and draws it from four sides. The agent reads
//! the pictures with its `read` tool, so it sees the hole it put on the wrong face, and goes
//! round again until the model is what was asked for.
//!
//! The skill is `SKILL.md` and `render.py`, mounted at `/skills/cad` from memory.
//!
//! * `context/` at `/context`, read-only — what the request is about, when it is not in the
//!   prompt: a sketch, a photo of the thing it has to fit, the STEP of a part to mate with.
//! * `artifacts/` at `/artifacts`, writable — the script, the STEP and STL files, a GLB for a
//!   viewer, and the pictures: four views, exploded, cut and turning.
//!
//! There is no model to download: `render.py` draws with a small rasterizer of its own, on the
//! CPU, so the console needs no GPU. OpenCascade's wheel still links libGL and libX11, which
//! the slim image leaves out.
//!
//! Environment:
//!
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …). It has to take images,
//!   or it cannot see what it built.
//!
//! Read from `.env` as well.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::Console,
    message::{FinishReason, Message, Part, Role},
};
use anyhow::Context as _;
use cortex::{
    fs::{Directory, FuseTMount},
    image::Image,
};
use futures::StreamExt as _;

/// The request when none is given.
const QUERY: &str = "Design a gear bearing that prints in one piece, already assembled: a \
    sun, five planets and a ring, all with herringbone teeth so the planets cannot slide out, \
    and enough clearance that it turns when it comes off the bed. About 60 mm across and 15 mm \
    tall, with a hexagonal hole through the sun for a key. Show it assembled, cut in half and \
    turning.";

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
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cad");
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "openai/gpt-6-astra".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: /context\n\n",
        "Holds what you were given to design from, when it is not in the request: sketches, ",
        "photos or drawings of what the part has to fit, or models of the parts it mates with. ",
        "List it before you start. This folder is read-only.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the model goes: its script, the STEP and STL files and the pictures of it, in a ",
        "folder named so the user can tell what it is. A result that is only in your reply is ",
        "not delivered as a file.",
    ))
    // A whole model script is one `write`, and the model thinks before it, which counts
    // against the same limit: far more than the 8192 tokens a reply gets by default.
    .max_tokens(64000)
    .system_tools()
    .console(
        Console::builder()
            .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(
                Image::new()
                    .base("python:3.12-slim-trixie")
                    .step(
                        "apt-get update && apt-get install -y --no-install-recommends \
                        libgl1 libx11-6 && rm -rf /var/lib/apt/lists/*",
                    )
                    .step("pip install --no-cache-dir vtk==9.6.2")
                    .step("pip install --no-cache-dir cadquery-ocp==7.9.3.1.1")
                    .step(
                        "pip install --no-cache-dir --no-deps cadquery==2.8.0 \
                        && pip install --no-cache-dir casadi ezdxf multimethod nlopt pyparsing \
                        runtype scipy typing_extensions trimesh numpy pillow",
                    ),
            )
            .mount_readonly(
                FuseTMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("render.py", include_str!("render.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/cad",
            )
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            .vcpus(4)
            .memory_mib(4096)
            .build()
            .await
            .with_context(|| "starting the console")?,
    )
    .skill("/skills/cad")
    .build()
    .await?;

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

    Ok(())
}
