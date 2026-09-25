//! Answer typed questions with Laya through ncnn on the guest's Vulkan device, as an agent's
//! skill.
//!
//! ```sh
//! cargo run --example laya
//! cargo run --example laya -- "Triage this ticket: we were billed twice for March …"
//! ```
//!
//! [Laya](https://huggingface.co/convaiinnovations/laya) is a decision model: given a state
//! (a text, an email, a ticket) and typed questions — a choice, a score, a yes/no — it answers
//! each with calibrated probabilities in one forward pass, and generates no text.
//!
//! The skill is `SKILL.md` and `run_laya.py`, mounted at `/skills/laya` from memory, which the
//! agent runs with its `shell` tool.
//!
//! * `context/` at `/context`, read-only — what to decide on, when it is not in the prompt.
//! * `artifacts/` at `/artifacts`, writable — where what the agent hands back goes.
//!
//! The image is Debian rather than Alpine. PyPI's ncnn has a musllinux wheel, but it crashes
//! freeing the first `Mat` it allocates, where the manylinux (glibc) one runs laya.
//! `mesa-vulkan-drivers` carries the venus ICD the guest needs — from trixie-backports, for
//! bf16 — and `libvulkan1` the loader the wheel opens.
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
use cortex::{console::NetworkAccess, image::Image};
use futures::StreamExt as _;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    let project_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/laya");
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");

    prepare(&project_path).await?;

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
        "Holds what you were given to decide on, such as the user's messages, ",
        "tickets and documents. When the request refers to something that is not in it, look ",
        "here first. List the folder, read what bears on the request, and pass that text to Laya ",
        "as the state. This folder is read-only.\n\n",
        "# Artifacts\n\n",
        "Path: /artifacts\n\n",
        "Where the files the user asks for go. When they ask for a file, such as ",
        "a report or a table of decisions, write it here and name it so they can tell what it ",
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
                    .step("pip install --no-cache-dir ncnn numpy tokenizers"),
            )
            .mount_readonly(project_path.join("data/ncnn"), "/models")
            .mount_readonly(
                cortex::fs::FuseTMount::try_new(
                    cortex::fs::Directory::new()
                        .with_file("SKILL.md", include_str!("SKILL.md").as_bytes())?
                        .with_file("run_laya.py", include_str!("run_laya.py").as_bytes())?,
                    &project_path.join("skill"),
                )
                .with_context(|| "mounting the skill")?,
                "/skills/laya",
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
    .skill("/skills/laya")
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
            // Laya's answers are small, and they are what the reply is made from: shown whole.
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

/// Download and convert the model into `project/data`
async fn prepare(project: &Path) -> anyhow::Result<()> {
    let uv = std::env::var("UV").unwrap_or_else(|_| "uv".to_string());
    let status = tokio::process::Command::new(&uv)
        .args(["run", "prepare_model.py"])
        .current_dir(project)
        // An activated environment elsewhere is not this project's, and uv says so.
        .env_remove("VIRTUAL_ENV")
        // `transformers` probes for TensorFlow at import, which can hang model construction.
        .env("USE_TF", "0")
        .status()
        .await
        .with_context(|| format!("running `{uv}`. Install uv, or point $UV at it."))?;
    anyhow::ensure!(status.success(), "preparing the model: {status}");
    Ok(())
}
