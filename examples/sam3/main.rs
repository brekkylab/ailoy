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
//! * `AILOY_CORTEX_CONSOLE` — the console server binary, `cortex-krun` by default.
//!   It has to be built with the `gpu` feature, or the session is refused with
//!   `UNSUPPORTED_MACHINE`.
//! * `UV` — the `uv` binary `prepare_model.py` runs with, `uv` on `PATH` by default.
//! * `AILOY_MODEL` — the agent's model, `anthropic/claude-sonnet-5` by default; its provider's
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
// One host binding per platform, each mounting on `try_new` and unmounting on `Drop`, so
// the tree below is written once. Three arms and not `not(windows)`: `fuse-t` is macOS
// only, and its build script probes pkg-config for FUSE-T on whatever host it runs, so
// naming it on Linux fails at build time. Cargo.toml splits the feature the same way.
#[cfg(windows)]
use cortex::fs::DokanMount as HostMount;
#[cfg(target_os = "linux")]
use cortex::fs::FuseMount as HostMount;
#[cfg(target_os = "macos")]
use cortex::fs::FuseTMount as HostMount;
use cortex::{console::NetworkAccess, fs::Directory, image::Image};
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

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
    )
    .instruction(concat!(
        "# Context\n\n",
        "Path: /context\n\n",
        "Holds what you were given to segment, such as images, and videos as folders of ",
        "frames. When the request refers to something that is not in it, look here first. ",
        "List the folder, and pass SAM3 the paths that bear on the request. ",
        "This folder is read-only.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the files the user asks for go, such as the masks and overlays SAM3 writes or a ",
        "report on them. Have SAM3 write into a folder here named so the user can tell what it ",
        "is. A result that is only in your reply is not delivered as a file.",
    ))
    .system_tools()
    .web_fetch_tool()
    .web_search_tool(vec![])
    .console(
        Console::builder()
            .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(
                Image::new()
                    .base("python:3.12-slim-trixie")
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
                    .step(
                        "pip install --no-cache-dir av ncnn numpy opencv-python-headless \
                        pillow tokenizers",
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
