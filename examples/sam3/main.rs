//! Segment images and videos with SAM3 through ncnn on the guest's Vulkan device, as an agent's
//! skill.
//!
//! ```sh
//! cargo run --example sam3
//! cargo run --example sam3 -- "Find every cat in the photos in context and mask them"
//! ```
//!
//! [SAM3](https://huggingface.co/facebook/sam3) is one ViT backbone and three heads on it, all
//! of them here, converted from the checkpoint by `prepare_model.py`:
//!
//! * the detector, which finds every instance of a text prompt or of example boxes;
//! * the tracker, SAM 2's mask decoder, which segments one object from points, a box or a mask;
//! * the video tracker, the same decoder on a memory of earlier frames, which follows objects
//!   through a video from a prompt on one of its frames.
//!
//! The skill is `SKILL.md` and `run_sam3.py`, mounted at `/skills/sam3` from memory, which the
//! agent runs with its `shell` tool.
//!
//! * `context/` at `/context`, read-only — the images and frames to segment, when they are not
//!   in the prompt.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! The image is Debian rather than Alpine because PyPI's ncnn wheels are manylinux
//! (glibc) only. `mesa-vulkan-drivers` carries the venus ICD the guest needs — from
//! trixie-backports, for bf16 — and `libvulkan1` the loader the wheel opens.
//!
//! Environment:
//!
//! * `UV` — the `uv` binary `prepare_model.py` runs with, `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
//!   API key has to be set (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …).
//!
//! Read from `.env` as well.

use std::{io::Write as _, path::Path};

use ailoy::{
    agent::AgentBuilder,
    console::ConsoleClient,
    message::{Message, Part, Role},
};
use anyhow::Context as _;
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
use cortex::{fs::Directory, image::Recipe, protocol::NetworkAccess};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/sam3");
    prepare(&project_path).await?;
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }

    // `build()` below builds the image before it starts the console, which takes a while.
    println!("building the image ...");
    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL")
            .unwrap_or_else(|_| "bedrock/global.openai.gpt-6-astra".to_string()),
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
    .console(
        ConsoleClient::builder()
            .image(
                Recipe::new("python:3.12-slim-trixie")
                    // Mesa from backports: venus passes VK_KHR_shader_bfloat16 and
                    // VK_KHR_cooperative_matrix through from 26.0 on, and trixie itself has 25.0.
                    .step(
                        "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                        > /etc/apt/sources.list.d/backports.list \
                        && apt-get update && apt-get install -y --no-install-recommends \
                        -t trixie-backports mesa-vulkan-drivers \
                        && apt-get install -y --no-install-recommends libvulkan1 \
                        && rm -rf /var/lib/apt/lists/*",
                    )
                    // Headless OpenCV: `opencv-python` needs libGL, which the slim image lacks.
                    // `ncnn` asks for it by name, and both wheels install the same `cv2`
                    // package, so naming the headless one alongside `ncnn` is not enough --
                    // whichever lands second wins. Hence `--no-deps` on `ncnn` and its own
                    // dependencies spelled out, opencv aside.
                    .step(
                        "pip install --no-cache-dir av numpy opencv-python-headless pillow \
                        portalocker requests tokenizers tqdm \
                        && pip install --no-cache-dir --no-deps ncnn",
                    ),
            )
            .mount_readonly(project_path.join("data/ncnn"), "/models")
            .mount_readonly(
                HostMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("run_sam3.py", include_str!("run_sam3.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/sam3",
            )
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            // The build's `apt-get` and `pip` run with the session's reach.
            .network(NetworkAccess::public())
            .gpu(true)
            .vcpus(2)
            .memory_mib(4096)
            .gpu_memory_mib(12288)
            .build()
            .await
            .with_context(|| format!("starting the console"))?,
    )
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

    Ok(())
}

/// Download and convert the models into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the models: {status}");
    Ok(())
}
