//! Segment images and videos with SAM3 through ncnn on the guest's Vulkan device, as an agent's
//! skill.
//!
//! ```sh
//! cargo run --example sam3
//! cargo run --example sam3 -- "Find every cat in the photos in context and mask them"
//! ```
//!
//! [SAM3](https://huggingface.co/facebook/sam3) is one ViT backbone with three heads, all
//! converted from the checkpoint by `prepare_model.py`:
//!
//! * the detector, which finds every instance of a text prompt or of example boxes;
//! * the tracker, SAM 2's mask decoder, which segments one object from points, a box or a mask;
//! * the video tracker, the same decoder on a memory of earlier frames, which follows objects
//!   through a video from a prompt on one of its frames.
//!
//! Four volumes on the virtx server are mounted into the sandbox:
//!
//! * `ailoy-sam3-skill` at `/skills/sam3`, read-only — the skill (`SKILL.md`, `run_sam3.py`)
//!   from `../shared/`, put there each run.
//! * `ailoy-sam3-models` at `/models`, read-only — `data/ncnn/` beside this file, the converted
//!   models, put there each run.
//! * `ailoy-sam3-context` at `/context`, read-only — `context/` beside this file, put there each
//!   run: the images and frames to segment, when they are not in the prompt.
//! * `ailoy-sam3-artifacts` at `/artifacts`, writable — where what the agent hands back goes.
//!   It stays on the server between runs, and is copied to `artifacts/` beside this file at the
//!   end.
//!
//! Environment, also read from `.env`:
//!
//! * `UV` — the `uv` binary `prepare_model.py` runs with; `uv` on `PATH` by default.
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
const IMAGE: &str = "ailoy-sam3:1";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");

    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    // What the Rust, Python and Node sides share: the skill and the scripts that fetch its data.
    let shared_path = examples.join("sam3/shared");
    // What a run reads and writes, beside this file.
    let project_path = examples.join("sam3/rust");
    prepare(&shared_path, &project_path).await?;
    let context = project_path.join("context");
    std::fs::create_dir_all(&context).with_context(|| format!("creating {}", context.display()))?;

    // The virtx server, fetched into virtx's cache the first time: a host that installed only
    // ailoy has none. The sandboxes it holds go when this program does.
    let client = tokio::task::spawn_blocking(|| Client::spawn_virtx([] as [&str; 0])).await??;

    // The image, built once. The server answers a tag in its full form, as
    // `docker.io/library/ailoy-sam3:1`.
    let images = client.list_image().await?;
    if !images
        .iter()
        .any(|i| i.reference.rsplit('/').next() == Some(IMAGE))
    {
        println!("Building {IMAGE}, once; it takes a few minutes.");
        // Debian, not Alpine: PyPI's ncnn wheels are manylinux (glibc) only.
        let recipe = Recipe::new("python:3.12-slim-trixie")
            // `mesa-vulkan-drivers` carries the guest's venus ICD, `libvulkan1` the loader the
            // wheel opens. Mesa from backports: venus passes VK_KHR_shader_bfloat16 and
            // VK_KHR_cooperative_matrix through from 26.0 on, and trixie itself has 25.0.
            .step(
                "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                > /etc/apt/sources.list.d/backports.list \
                && apt-get update && apt-get install -y --no-install-recommends \
                -t trixie-backports mesa-vulkan-drivers \
                && apt-get install -y --no-install-recommends libvulkan1 \
                && rm -rf /var/lib/apt/lists/*",
            )
            // Headless OpenCV: `opencv-python` needs libGL, which the slim image lacks. `ncnn`
            // asks for it by name, and both wheels install the same `cv2` package, so naming the
            // headless one alongside `ncnn` is not enough -- whichever lands second wins. Hence
            // `--no-deps` on `ncnn` and its own dependencies spelled out, opencv aside.
            .step(
                "pip install --no-cache-dir av numpy opencv-python-headless pillow \
                portalocker requests tokenizers tqdm \
                && pip install --no-cache-dir --no-deps ncnn",
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
        "ailoy-sam3-skill",
        "ailoy-sam3-models",
        "ailoy-sam3-context",
        "ailoy-sam3-artifacts",
    ] {
        volumes.push(match client.create_volume(Some(name)).await {
            Err(e) if e.status() == Some(409) => client.get_volume(name).await?,
            made => made?,
        });
    }
    let [skill, models, context_volume, artifacts] = <[_; 4]>::try_from(volumes).unwrap();

    // The skill onto the first.
    for name in ["SKILL.md", "run_sam3.py"] {
        skill
            .write_file(name, std::fs::read(shared_path.join(name))?, None)
            .await?;
    }

    // `data/ncnn/`, every file of it, onto the second.
    let mut dirs = vec![(project_path.join("data/ncnn"), String::new())];
    while let Some((dir, at)) = dirs.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let to = format!("{at}{}", entry.file_name().to_string_lossy());
            if entry.file_type()?.is_dir() {
                models.create_dir_all(to.clone()).await?;
                dirs.push((entry.path(), format!("{to}/")));
            } else {
                models
                    .write_file(to, std::fs::read(entry.path())?, None)
                    .await?;
            }
        }
    }

    // `context/`, every file of it, onto the third.
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
            volume: models.name().to_string(),
            path: "/models".to_string(),
            readonly: true,
        },
        SandboxVolume {
            volume: skill.name().to_string(),
            path: "/skills/sam3".to_string(),
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
    args.gpu = Some(true);
    args.vcpus = Some(2);
    args.memory_mib = Some(4096);
    args.gpu_memory_mib = Some(12288);
    let sandbox = client
        .create_sandbox(&args)
        .await
        .with_context(|| "starting the sandbox")?;

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
    .web_fetch_tool()
    .web_search_tool(vec![])
    .sandbox(sandbox)
    .skill("/skills/sam3")
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
            // A run prints a summary of what it found and where the masks went, and ncnn's
            // device log: shown whole.
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

/// Download and convert the models into `project/data`
async fn prepare(shared: &Path, project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .arg(project.join("data"))
        .current_dir(shared)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the models: {status}");
    Ok(())
}
