//! Detect objects named in plain text with YOLOE, through ncnn on the guest's Vulkan device, as
//! an agent's skill.
//!
//! ```sh
//! cargo run --example yoloe
//! cargo run --example yoloe -- "Count the people in the images in context and draw their boxes"
//! YOLOE_CLASSES="person,bus,glasses" cargo run --example yoloe -- "..."
//! ```
//!
//! [YOLOE](https://docs.ultralytics.com/models/yoloe/) is an open-vocabulary detector: it looks
//! for whatever classes it is given as text, rather than the 80 COCO classes a YOLO is trained
//! on. The text goes through MobileCLIP once, on the host, into embeddings YOLOE folds into its
//! head, so the model in the guest is as fast as a plain YOLO and finds those classes, and no
//! others: the agent cannot change them. `prepare_model.py` converts it again when they change.
//!
//! The skill is `SKILL.md` and `run_yoloe.py`, mounted at `/skills/yoloe` from memory, which
//! the agent runs with its `shell` tool.
//!
//! * `context/` at `/context`, read-only — what to run on, when it is not in the prompt. The
//!   image the reference was taken on is put there when it is empty.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! The image is Debian rather than Alpine because PyPI's ncnn wheels are manylinux (glibc)
//! only. `mesa-vulkan-drivers` carries the venus ICD the guest needs — from trixie-backports,
//! for bf16 — and `libvulkan1` the loader the wheel opens. The ncnn wheel goes in without its
//! dependencies, which would bring in `opencv-python` and the X and GL libraries it needs; the
//! headless OpenCV does the letterboxing instead.
//!
//! YOLOE and its weights are Ultralytics', under AGPL-3.0.
//!
//! Environment:
//!
//! * `YOLOE_CLASSES` — the classes to look for, comma-separated; see `prepare_model.py` for
//!   the default.
//! * `YOLOE_IMAGE` — the image the reference is taken on, Ultralytics' `bus.jpg` by default.
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
use cortex::{console::NetworkAccess, fs::Directory, image::Image};
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
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");

    // Absolute, because a mount is named to the server as a `file://` URL.
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/yoloe");
    prepare(&project_path).await?;
    // `skill` too, which is empty on the host: it is where the skill is mounted from memory.
    for dir in ["context", "artifacts", "skill"] {
        let dir = project_path.join(dir);
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    // Something to look in, for a first run.
    let context = project_path.join("context");
    if std::fs::read_dir(&context)?.next().is_none() {
        std::fs::copy(
            project_path.join("data/ncnn/image.jpg"),
            context.join("image.jpg"),
        )
        .with_context(|| "copying the reference image into context")?;
    }

    let mut agent = AgentBuilder::new(
        std::env::var("AILOY_MODEL").unwrap_or_else(|_| "anthropic/claude-sonnet-5".to_string()),
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
        Console::builder()
            .stdio_client(&[&std::env::var("AILOY_CORTEX_CONSOLE")
                .unwrap_or_else(|_| "cortex-krun".to_string())])
            .image(
                Image::new()
                    .base("python:3.12-slim-trixie")
                    // Mesa from backports: venus passes VK_KHR_shader_bfloat16 through from 26.0
                    // on, and trixie itself has 25.0.
                    .step(
                        "echo 'deb http://deb.debian.org/debian trixie-backports main' \
                        > /etc/apt/sources.list.d/backports.list \
                        && apt-get update && apt-get install -y --no-install-recommends \
                        -t trixie-backports mesa-vulkan-drivers \
                        && apt-get install -y --no-install-recommends libvulkan1 \
                        && rm -rf /var/lib/apt/lists/*",
                    )
                    .step(
                        "pip install --no-cache-dir --no-deps ncnn \
                        && pip install --no-cache-dir numpy opencv-python-headless",
                    ),
            )
            .mount_readonly(project_path.join("data/ncnn"), "/models")
            .mount_readonly(
                HostMount::try_new(
                    Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("run_yoloe.py", include_str!("run_yoloe.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/yoloe",
            )
            .mount_readonly(project_path.join("context"), "/context")
            .mount(project_path.join("artifacts"), "/artifacts")
            // The build's `apt-get` and `pip` run with the session's reach.
            .network(NetworkAccess::public())
            .gpu(true)
            .vcpus(2)
            .memory_mib(2048)
            .build()
            .await
            .with_context(|| format!("starting the console"))?,
    )
    .skill("/skills/yoloe")
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
            // A run prints the detections and ncnn's device log: shown whole.
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

/// Download, set the classes of and convert the model into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let mut command = tokio::process::Command::new(&uv);
    // Named from where this runs, not from the project directory `uv` runs in.
    if let Some(image) = std::env::var_os("YOLOE_IMAGE") {
        let image = std::path::absolute(&image)
            .with_context(|| format!("resolving YOLOE_IMAGE {}", image.display()))?;
        command.env("YOLOE_IMAGE", image);
    }
    let status = command
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the model: {status}");
    Ok(())
}
