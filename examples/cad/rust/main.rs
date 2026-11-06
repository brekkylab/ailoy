//! Design 3D parts with an agent that writes CadQuery, looks at what it built and fixes it, in
//! a sandbox.
//!
//! ```sh
//! cargo run --example cad
//! cargo run --example cad -- "A wall mount for a 60 mm fan, with four M3 screw holes"
//! cargo run --example cad -- "A twisted vase, 150 mm tall, hexagonal at the base and round at the top"
//! ```
//!
//! [CadQuery](https://cadquery.readthedocs.io) is a Python library for parametric CAD on the
//! OpenCascade kernel: a model is a script, and what it builds is exact solids, not meshes. The
//! agent writes that script; the skill's `render.py` runs it, checks each part — valid, one
//! solid, closed, not overlapping another — and draws it from four sides. The agent reads the
//! pictures with its `imgread` tool and goes round again until the model is what was asked for.
//! `render.py` rasterizes on the CPU, so there is no model to download and no GPU needed.
//!
//! Three volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-cad-skill` at `/skills/cad`, read-only — the skill (`SKILL.md`, `render.py`) from
//!   `../shared/`, put there each run.
//! * `ailoy-cad-context` at `/context`, read-only — `context/` beside this file, put there each
//!   run: what the request is about, when it is not in the prompt: a sketch, a photo of the
//!   thing it has to fit, the STEP of a part to mate with.
//! * `ailoy-cad-artifacts` at `/artifacts`, writable — the script, the STEP, STL and GLB files,
//!   and the pictures. It stays on the server between runs, and is copied to `artifacts/`
//!   beside this file at the end.
//!
//! Environment, also read from `.env`:
//!
//! * `AILOY_MODEL` — the agent's model, `openai/gpt-6-astra` by default; its org's backend has to
//!   be set up as `docs/src/content/docs/guide/model-providers.mdx` says. It has to take
//!   images, or it cannot see what it built.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    message::{FinishReason, Message, Part, Role},
};
use anyhow::{Context as _, bail};
use futures::StreamExt as _;
use virtx::{Client, Files as _, ImageJobEntry, Recipe, SandboxCreateArgs, SandboxVolume};

/// The image the sandbox boots, built once on the virtx server and kept under this tag.
const IMAGE: &str = "ailoy-cad:1";

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

    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill.
    let shared = examples.join("cad/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("cad/rust");
    let context = project_path.join("context");
    std::fs::create_dir_all(&context).with_context(|| format!("creating {}", context.display()))?;

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once. The server answers a tag in its full form, as
    // `docker.io/library/ailoy-cad:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        let recipe = Recipe::new("python:3.12-slim-trixie")
            // OpenCascade's wheel links libGL and libX11, which the slim image leaves out.
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
        "ailoy-cad-skill",
        "ailoy-cad-context",
        "ailoy-cad-artifacts",
    ] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [skill, context_volume, artifacts] = <[_; 3]>::try_from(volumes).unwrap();

    // The skill onto the first.
    for name in ["SKILL.md", "render.py"] {
        skill
            .write_file(name, std::fs::read(shared.join(name))?, None)
            .await?;
    }

    // `context/`, every file of it, onto the second.
    let mut dirs = vec![(context.clone(), String::new())];
    while let Some((dir, at)) = dirs.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let to = format!("{at}{}", entry.file_name().to_string_lossy());
            if entry.file_type()?.is_dir() {
                context_volume.create_dir_all(to.clone()).await?;
                dirs.push((entry.path(), format!("{to}/")));
            } else {
                context_volume
                    .write_file(to, std::fs::read(entry.path())?, None)
                    .await?;
            }
        }
    }

    let mut args = SandboxCreateArgs::new(IMAGE);
    args.volumes = vec![
        SandboxVolume {
            volume: skill.name().to_string(),
            path: "/skills/cad".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: context_volume.name().to_string(),
            path: "/context".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: artifacts.name().to_string(),
            path: "/artifacts".to_string(),
            readonly: false,
        },
    ];
    args.vcpus = Some(4);
    args.memory_mib = Some(4096);
    let sandbox = client
        .create_sandbox(&args)
        .await
        .with_context(|| "starting the sandbox")?;

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
    .sandbox(sandbox)
    .skill("/skills/cad")
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
